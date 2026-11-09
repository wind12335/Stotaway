"""8-card gate: host-sync-free MoE layer (dispatch -> grouped expert MLP -> combine, fwd + bwd)
vs the Megatron-style RCCL path (count a2a + .tolist() + varlen a2a + per-expert loop + reverse a2a).

Shape: Qwen3-30B-A3B MoE layer, EP=8: 128 experts (16 local), top-8, H=2048, expert ffn 768.
Routing: real routing traces (opus_moe_routes.npz, 4 skew slices) and fresh tokens every round.
Correctness (per round, fresh x / w / upstream grad): out, dx, dw_router, dW1, dW2 of ours vs the
RCCL path, both vs an fp32 reference computed from the SAME inputs. Pass = ours' error <= max(2 x RCCL's
error, 1e-2) for every tensor and round. Timing: forward and forward+backward, median over ITERS, max
over ranks; the sync-free path runs ROUNDS rounds back to back with no barrier.
Env: SG_T tokens per rank (default 4096 = MBS1 seq4096), SG_ITERS, SG_ROUNDS, SG_B (virtual group rows).
"""
import ctypes, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_moesf_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")
import numpy as np  # noqa: E402
import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm); R = pHost.mpi_comm_size(mpi, comm)
hipl = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so")); hipl.hipSetDevice.restype = ctypes.c_int
assert hipl.hipSetDevice(rank) == 0
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
torch.cuda.set_device(rank)
import torch.distributed as dist  # noqa: E402
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29961")
dist.init_process_group("nccl", rank=rank, world_size=R)
_x = torch.ones(1, device="cuda")
for _ in range(3):
    dist.all_reduce(_x)
torch.cuda.synchronize()
from stowaway.kernels.common_ops import init_dcu  # noqa: E402
host = pHost.DushmemHost(preload_bootstrap=False); host.init_mpi(comm)
init_dcu(mpi, comm, rank, R, host=host)
from stowaway.kernels.moe_sf import SFMoEContext, sf_moe  # noqa: E402
from stowaway.kernels.moe_expert_mlp import get_plan, expert_linear, _ptr_table  # noqa: E402

H, E, K, FE = 2048, 128, 8, 768
EPR = E // R
T = int(os.environ.get("SG_T", "4096"))
ITERS = int(os.environ.get("SG_ITERS", "20"))
ROUNDS = int(os.environ.get("SG_ROUNDS", "6"))
BV = int(os.environ.get("SG_B", "2048"))
DT = torch.bfloat16
MAXROWS = int(os.environ.get("SG_MAXROWS", str(R * T * K // R * 3)))   # 3x the mean rows per expert rank
z = np.load(os.path.join(HERE, "data", "qwen3_30b_a3b_routes.npz"))
IDX, SKEW = z["idx"], z["skew"]
TR = IDX.shape[1] // R
assert T <= TR * 4, "routing trace too short"

torch.manual_seed(1234 + rank)
W1 = [(torch.randn(2 * FE, H, device="cuda", dtype=DT) * 0.02).requires_grad_() for _ in range(EPR)]
W2 = [(torch.randn(H, FE, device="cuda", dtype=DT) * 0.02).requires_grad_() for _ in range(EPR)]
t1, t2 = _ptr_table(W1), _ptr_table(W2)
p1 = get_plan(EPR, H, 2 * FE, MAXROWS, BV)
p2 = get_plan(EPR, FE, H, MAXROWS, BV)
sf = SFMoEContext(T, K, H, E, rank, R, MAXROWS)


def experts_gg(xp, offs):
    h = expert_linear(xp, offs, W1, p1, t1)
    a, b = h.chunk(2, dim=-1)
    return expert_linear((F.silu(a) * b).contiguous(), offs, W2, p2, t2)


def ours(x, idx, logits):
    w = torch.softmax(logits, -1)                          # [T, K] routing weights (fp32), grad to router
    return sf_moe(x, idx, w, sf, experts_gg)


# ---------------------------------------------------------------- RCCL path (Megatron alltoall style)
class _A2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sp_out, sp_in):
        ctx.sp_out, ctx.sp_in = sp_out, sp_in
        y = torch.empty(sum(sp_out), x.shape[1], dtype=x.dtype, device=x.device)
        dist.all_to_all_single(y, x.contiguous(), sp_out, sp_in)
        return y

    @staticmethod
    def backward(ctx, g):
        y = torch.empty(sum(ctx.sp_in), g.shape[1], dtype=g.dtype, device=g.device)
        dist.all_to_all_single(y, g.contiguous(), ctx.sp_in, ctx.sp_out)
        return y, None, None


def rccl(x, idx, logits, w1=None, w2=None, fp32=False):
    w1 = w1 or W1; w2 = w2 or W2
    w = torch.softmax(logits, -1)
    dst = idx.long() // EPR
    order = torch.argsort(dst, stable=True)
    c = torch.bincount(dst, minlength=R)
    rc = torch.empty_like(c); dist.all_to_all_single(rc, c)
    sp_in, sp_out = c.tolist(), rc.tolist()                              # host sync (as Megatron)
    xs = x.repeat_interleave(K, 0)[order]
    rx = _A2A.apply(xs, sp_out, sp_in)
    lid_s = (idx.long() % EPR)[order].to(torch.float32).unsqueeze(1)
    rl = torch.empty(sum(sp_out), 1, dtype=torch.float32, device="cuda")
    dist.all_to_all_single(rl, lid_s.contiguous(), sp_out, sp_in)
    lid = rl.squeeze(1).long()
    o2 = torch.argsort(lid, stable=True)
    tpe = torch.bincount(lid, minlength=EPR).tolist()                    # host sync (as Megatron)
    xe = rx[o2]
    outs = []
    for e, xi in enumerate(torch.split(xe, tpe)):
        if fp32:
            xi = xi.float()
            h = xi @ w1[e].float().t(); a, b = h.chunk(2, -1); outs.append((F.silu(a) * b) @ w2[e].float().t())
        else:
            h = xi @ w1[e].t(); a, b = h.chunk(2, -1); outs.append((F.silu(a) * b) @ w2[e].t())
    ye = torch.cat(outs)
    yr = torch.empty_like(ye); yr[o2] = ye
    if fp32:
        yr = yr.float()
    back = _A2A.apply(yr, sp_in, sp_out)
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device="cuda")
    yk = back[inv].view(T, K, H)
    return (yk.float() * w.unsqueeze(-1)).sum(1).to(torch.float32 if fp32 else DT)


def fwd_bwd(fn, x0, idx, lg0, g, **kw):
    for p in W1 + W2:
        p.grad = None
    x = x0.clone().requires_grad_(); lg = lg0.clone().requires_grad_()
    out = fn(x, idx, lg, **kw)
    (out.float() * g.float()).sum().backward()
    return out.detach(), x.grad, lg.grad, [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in W1], \
        [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in W2]


def ref32(x0, idx, lg0, g):
    w1 = [p.detach().float().requires_grad_() for p in W1]; w2 = [p.detach().float().requires_grad_() for p in W2]
    x = x0.float().requires_grad_(); lg = lg0.clone().requires_grad_()
    out = rccl(x, idx, lg, w1=w1, w2=w2, fp32=True)
    (out * g.float()).sum().backward()
    return out.detach(), x.grad, lg.grad, [p.grad for p in w1], [p.grad for p in w2]


def rel(a, b):
    b = b.float()
    return float((a.float() - b).abs().max() / b.abs().max().clamp_min(1e-12))


def allmax(v):
    t = torch.tensor([float(v)], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX); return t.item()


def tm(fn, it=ITERS):
    for _ in range(3):
        fn(); torch.cuda.synchronize(); dist.barrier()
    ts = []
    for _ in range(it):
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(allmax((time.perf_counter() - t0) * 1e3))
    return statistics.median(ts)


st = torch.cuda.current_stream()
for si in range(IDX.shape[0]):
    full = IDX[si]                                   # [R*TR*K]? or [R*TR, K]
    full = full.reshape(-1, K)
    reps = (R * T + full.shape[0] - 1) // full.shape[0]
    full = np.concatenate([full] * reps, 0)[: R * T]
    idx = torch.from_numpy(full[rank * T:(rank + 1) * T]).to(torch.int32).cuda().reshape(-1).contiguous()
    names = ["out", "dx", "dlogit", "dW1", "dW2"]
    worst = {n: [0.0, 0.0] for n in names}
    fails = 0
    for r_ in range(ROUNDS):
        torch.manual_seed(10007 * si + 101 * r_ + rank)
        x0 = (torch.randn(T, H, device="cuda") * 0.5).to(DT)
        lg0 = torch.randn(T, K, device="cuda")
        g = torch.randn(T, H, device="cuda").to(DT)
        a = fwd_bwd(ours, x0, idx, lg0, g)
        b = fwd_bwd(rccl, x0, idx, lg0, g)
        c = ref32(x0, idx, lg0, g)
        errs = []
        for n, ta, tb, tc in zip(names, a, b, c):
            if isinstance(ta, list):
                ea = max(rel(u, v) for u, v in zip(ta, tc)); eb = max(rel(u, v) for u, v in zip(tb, tc))
            else:
                ea, eb = rel(ta, tc), rel(tb, tc)
            ea, eb = allmax(ea), allmax(eb)
            worst[n] = [max(worst[n][0], ea), max(worst[n][1], eb)]
            errs.append(ea <= max(2 * eb, 1e-2))
        fails += 0 if all(errs) else 1
    ovf = allmax(sf.overflow.item())
    x0 = (torch.randn(T, H, device="cuda") * 0.5).to(DT); lg0 = torch.randn(T, K, device="cuda"); g = torch.randn(T, H, device="cuda").to(DT)

    def f_ours():
        with torch.no_grad():
            ours(x0, idx, lg0)

    def f_rccl():
        with torch.no_grad():
            rccl(x0, idx, lg0)
    t_of, t_rf = tm(f_ours), tm(f_rccl)
    t_ob = tm(lambda: fwd_bwd(ours, x0, idx, lg0, g)); t_rb = tm(lambda: fwd_bwd(rccl, x0, idx, lg0, g))
    if rank == 0:
        print(f"[moesf] slice{si} skew={SKEW[si]:.3f} T={T} | correctness {ROUNDS - fails}/{ROUNDS} PASS overflow={ovf:g} | "
              + " ".join(f"{n} ours {worst[n][0]:.2e}/rccl {worst[n][1]:.2e}" for n in names), flush=True)
        print(f"[moesf] slice{si} fwd: ours {t_of:.3f} rccl {t_rf:.3f} -> x{t_rf / t_of:.3f} | fwd+bwd: ours {t_ob:.3f} "
              f"rccl {t_rb:.3f} -> x{t_rb / t_ob:.3f} ms", flush=True)
dist.barrier()
try:
    mpi.MPI_Finalize()
except Exception:
    pass
