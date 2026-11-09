"""Operator-level bench: backward of a row-parallel site under SP (= the AG half of fbag):
all-gather of the output gradient fused with dgrad (stowaway.kernels.gemm_rs_dx.dx_ag_bwd_op),
vs stock AG + dgrad + wgrad (TN).
Env: FG_M rows, FG_N = hidden, FG_KS = comma list of local K (fc2 = ffn/R, o_proj = hidden/R),
     FG_OPS correctness ops (fresh input every op; full-gather result must match bit-exactly).
"""
import ctypes, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_opus_bg_{os.environ.get('OMPI_COMM_WORLD_RANK','0')}")
import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm); R = pHost.mpi_comm_size(mpi, comm)
hipl = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so")); hipl.hipSetDevice.restype = ctypes.c_int
assert hipl.hipSetDevice(rank) == 0
import torch  # noqa: E402
torch.cuda.set_device(rank)
import torch.distributed as dist  # noqa: E402
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29985")
dist.init_process_group("nccl", rank=rank, world_size=R)
_x = torch.ones(1, device="cuda")
for _ in range(3):
    dist.all_reduce(_x)
torch.cuda.synchronize()
import stowaway  # noqa: E402
from stowaway.kernels.common_ops import init_dcu  # noqa: E402
from stowaway.kernels.gemm_rs_dx import DXAGBwdContext, dx_ag_bwd_op  # noqa: E402
host = pHost.DushmemHost(preload_bootstrap=False); host.init_mpi(comm)
init_dcu(mpi, comm, rank, R, host=host)
if rank == 0:
    print(f"[bg] stowaway from {stowaway.__file__}", flush=True)

M, N = int(os.environ.get("FG_M", "32768")), int(os.environ.get("FG_N", "5120"))
KS = [int(x) for x in os.environ.get("FG_KS", "2176,640").split(",")]
OPS = int(os.environ.get("FG_OPS", "64"))
BURST = int(os.environ.get("FG_BURST", "4"))
DT = torch.bfloat16
mp = M // R
ctx = DXAGBwdContext(M, N, DT, rank, R)
W = {}
for K in KS:
    torch.manual_seed(500 + K + rank)
    W[K] = torch.randn(N, K, dtype=DT, device="cuda") * 0.05


def allmax(v):
    t = torch.tensor([float(v)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def stock(g, w, a):
    full = torch.empty(M, N, dtype=DT, device="cuda")
    dist.all_gather_into_tensor(full, g)
    return torch.matmul(full, w), full.t().matmul(a), full


g = torch.Generator(device="cuda")
fails = 0
w_gi = w_full = w_gw = 0.0
op = 0
while op < OPS:
    batch = []
    for b in range(BURST):
        K = KS[(op + b) % len(KS)]
        g.manual_seed(20000 * (op + b) + rank)
        gs = (torch.randn(mp, N, generator=g, dtype=torch.float32, device="cuda") * 0.05).to(DT)
        a = (torch.randn(M, K, generator=g, dtype=torch.float32, device="cuda") * 0.05).to(DT)
        batch.append((K, gs, a))
    torch.cuda.synchronize()
    outs = [dx_ag_bwd_op(gs, W[K], a, ctx) for K, gs, a in batch]   # back-to-back
    torch.cuda.synchronize()
    for (K, gs, a), (gi, gw, full) in zip(batch, outs):
        rgi, rgw, rfull = stock(gs, W[K], a)
        d_full = allmax((full.float() - rfull.float()).abs().max().item())
        d_gw = allmax((gw.float() - rgw.float()).abs().max().item())
        d_gi = allmax(((gi.float() - rgi.float()).abs() / rgi.float().abs().clamp_min(1.0)).max().item())
        w_full, w_gw, w_gi = max(w_full, d_full), max(w_gw, d_gw), max(w_gi, d_gi)
        if d_full != 0 or d_gi >= 2e-2:
            fails += 1
            if rank == 0 and fails <= 5:
                print(f"[bg] FAIL K={K} full {d_full:g} gw {d_gw:g} gi {d_gi:.3e}", flush=True)
    op += BURST
if rank == 0:
    print(f"[bg] correctness ops={op} fails={fails} max|full-ref|={w_full:g} (must be 0) "
          f"max|gw-ref|={w_gw:g} rel(gi)={w_gi:.3e} -> {'PASS' if fails == 0 else 'FAIL'}", flush=True)

st = torch.cuda.current_stream()


def tm(fn, it=25):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(st); fn(); e.record(st); torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ts.append(t.item())
    return statistics.median(ts)


for K in KS:
    gs = torch.randn(mp, N, dtype=DT, device="cuda") * 0.05
    a = torch.randn(M, K, dtype=DT, device="cuda") * 0.05
    fb = torch.empty(M, N, dtype=DT, device="cuda")
    t_st = tm(lambda: stock(gs, W[K], a))
    t_dx = tm(lambda: dx_ag_bwd_op(gs, W[K], a, ctx, full=fb))
    t_st16 = tm(lambda: [stock(gs, W[K], a) for _ in range(16)], it=8) / 16
    t_dx16 = tm(lambda: [dx_ag_bwd_op(gs, W[K], a, ctx, full=fb) for _ in range(16)], it=8) / 16
    if rank == 0:
        print(f"[bg] K={K} stock(AG+dgrad+wgradTN) {t_st:.3f} | dx-ag-bwd {t_dx:.3f} -> x{t_st / t_dx:.3f} "
              f"| b2b/op stock {t_st16:.3f} dx {t_dx16:.3f} x{t_st16 / t_dx16:.3f}", flush=True)
dist.barrier()
try:
    mpi.MPI_Finalize()
except Exception:
    pass
