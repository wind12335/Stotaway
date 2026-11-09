"""Operator-level bench: GEMM+RS (row-parallel output, o_proj / fc2 call sites), PRODUCTION op
stowaway.kernels.gemm_rs_dx.dx_gemm_rs_op (+ the standalone engine RS dxv_op).

  stock = vendor GEMM + RCCL reduce_scatter_tensor (serial)
  ours  = DX overlap: GEMM in row blocks, permutation-order direct push of each block to its owner
          while the next block computes, ticket-gated fp32 sum, no barrier
Env: FG_M rows, FG_N = hidden, FG_KS = comma list of local K (fc2 = ffn/R, o_proj = hidden/R),
     FG_OPS correctness ops (fresh input every op), FG_NOBWD=1 skips the backward-piece timings.
Correctness ref: RCCL reduce_scatter of the full bf16 matmul (+ an fp32 ref).
Driver: bench/run_rs_matrix.sh.
"""
import ctypes, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_opus_fg2_{os.environ.get('OMPI_COMM_WORLD_RANK','0')}")
import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm); R = pHost.mpi_comm_size(mpi, comm)
hipl = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so")); hipl.hipSetDevice.restype = ctypes.c_int
assert hipl.hipSetDevice(rank) == 0
import torch  # noqa: E402
torch.cuda.set_device(rank)
import torch.distributed as dist  # noqa: E402
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29983")
dist.init_process_group("nccl", rank=rank, world_size=R)
_x = torch.ones(1, device="cuda")
for _ in range(3):
    dist.all_reduce(_x)
torch.cuda.synchronize()
import stowaway  # noqa: E402
from stowaway.kernels.common_ops import init_dcu  # noqa: E402
from stowaway.kernels.gemm_rs_dx import DXGemmRSContext, dx_gemm_rs_op, DXRSContext, dxv_op  # noqa: E402
host = pHost.DushmemHost(preload_bootstrap=False); host.init_mpi(comm)
init_dcu(mpi, comm, rank, R, host=host)
if rank == 0:
    print(f"[fg] stowaway from {stowaway.__file__}", flush=True)

M, N = int(os.environ.get("FG_M", "32768")), int(os.environ.get("FG_N", "5120"))
KS = [int(x) for x in os.environ.get("FG_KS", "2176,640").split(",")]
OPS = int(os.environ.get("FG_OPS", "0"))
BURST = int(os.environ.get("FG_BURST", "4"))
DT = torch.bfloat16
mp = M // R
ctx = DXGemmRSContext(M, N, DT, rank, R)
rsctx = DXRSContext(M * N, DT, rank, R, k_slots=4)
W = {}
for K in KS:
    torch.manual_seed(500 + K + rank)
    W[K] = torch.randn(N, K, dtype=DT, device="cuda") * 0.05


def rel(x, ref):
    v = ((x.float() - ref.float()).abs() / ref.float().abs().clamp_min(1.0)).max()
    t = v.reshape(1)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


worst_rccl = worst_ours = worst_vs_rccl = 0.0
fails = 0
g = torch.Generator(device="cuda")
op = 0
while op < OPS:
    batch = []
    for b in range(BURST):
        K = KS[(op + b) % len(KS)]
        g.manual_seed(10000 * (op + b) + rank)
        a = torch.randn(M, K, generator=g, dtype=torch.float32, device="cuda").to(DT) * 0.05
        out = torch.empty(mp, N, dtype=DT, device="cuda")
        batch.append((K, a, out))
    torch.cuda.synchronize()
    for K, a, out in batch:            # back-to-back, no sync between ops
        dx_gemm_rs_op(a, W[K], out, ctx)
    torch.cuda.synchronize()
    for K, a, out in batch:
        full = torch.matmul(a, W[K].t())
        ref = torch.empty(mp, N, dtype=DT, device="cuda")
        dist.reduce_scatter_tensor(ref, full)
        full32 = torch.matmul(a.float(), W[K].float().t())
        ref32 = torch.empty(mp, N, dtype=torch.float32, device="cuda")
        dist.reduce_scatter_tensor(ref32, full32)
        r1 = rel(out, ref)
        r2 = rel(ref, ref32)
        r3 = rel(out, ref32)
        worst_vs_rccl, worst_rccl, worst_ours = max(worst_vs_rccl, r1), max(worst_rccl, r2), max(worst_ours, r3)
        if r1 >= 2e-2:
            fails += 1
            if rank == 0 and fails <= 5:
                print(f"[fg] FAIL op K={K} rel_vs_rccl={r1:.3e}", flush=True)
    op += BURST
if rank == 0:
    print(f"[fg] correctness ops={op} fails={fails} rel(ours,rccl)={worst_vs_rccl:.3e} "
          f"rel(rccl,fp32)={worst_rccl:.3e} rel(ours,fp32)={worst_ours:.3e} -> "
          f"{'PASS' if fails == 0 else 'FAIL'}", flush=True)

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
    a = torch.randn(M, K, dtype=DT, device="cuda") * 0.05
    full = torch.empty(M, N, dtype=DT, device="cuda")
    o1 = torch.empty(mp, N, dtype=DT, device="cuda")
    o2 = torch.empty_like(o1)
    t_stock = tm(lambda: (torch.matmul(a, W[K].t(), out=full), dist.reduce_scatter_tensor(o1, full)))
    t_gemm = tm(lambda: torch.matmul(a, W[K].t(), out=full))
    t_rs = tm(lambda: dist.reduce_scatter_tensor(o1, full))
    t_dx = tm(lambda: dx_gemm_rs_op(a, W[K], o2, ctx))
    t_dxrs = tm(lambda: dxv_op(full, o1, rsctx))
    t_gdx = tm(lambda: (torch.matmul(a, W[K].t(), out=full), dxv_op(full, o1, rsctx)))
    # back-to-back throughput (no per-op sync): 16 ops, wall/16
    def b2b(fn, n=16):
        return lambda: [fn() for _ in range(n)]
    t_dx16 = tm(b2b(lambda: dx_gemm_rs_op(a, W[K], o2, ctx)), it=8) / 16
    t_gdx16 = tm(b2b(lambda: (torch.matmul(a, W[K].t(), out=full), dxv_op(full, o1, rsctx))), it=8) / 16
    # 10-31: stock back-to-back too, so the b2b speedup vs stock (RCCL) exists (matches the AG gate)
    t_st16 = tm(b2b(lambda: (torch.matmul(a, W[K].t(), out=full), dist.reduce_scatter_tensor(o1, full))), it=8) / 16
    if rank == 0:
        print(f"[fg2] K={K} stock {t_stock:.3f} (gemm {t_gemm:.3f} + rccl-rs {t_rs:.3f}) | dx-rs {t_dxrs:.3f} "
              f"| gemm+dx-rs {t_gdx:.3f} | dx-fused {t_dx:.3f} -> fused/gemm+dx x{t_gdx / t_dx:.3f} "
              f"| b2b/op gemm+dx {t_gdx16:.3f} fused {t_dx16:.3f} x{t_gdx16 / t_dx16:.3f} "
              f"| b2b/op stock {t_st16:.3f} fused {t_dx16:.3f} x{t_st16 / t_dx16:.3f}", flush=True)
# backward forms at the same shapes (grad_out [M/R, N] -> AG -> dgrad + wgrad)
for K in (KS if os.environ.get("FG_NOBWD") != "1" else []):
    a = torch.randn(M, K, dtype=DT, device="cuda") * 0.05
    g_sh = torch.randn(mp, N, dtype=DT, device="cuda") * 0.05
    full = torch.empty(M, N, dtype=DT, device="cuda")
    dist.all_gather_into_tensor(full, g_sh)
    t_ag = tm(lambda: dist.all_gather_into_tensor(full, g_sh))
    t_dg = tm(lambda: torch.matmul(full, W[K]))
    t_wg_nn = tm(lambda: torch.matmul(full.transpose(-2, -1).contiguous(), a))
    t_wg_tn = tm(lambda: full.t().matmul(a))
    t_tr = tm(lambda: full.transpose(-2, -1).contiguous())
    if rank == 0:
        print(f"[fg2-bwd] K={K} ag {t_ag:.3f} | dgrad {t_dg:.3f} | wgrad NN(contig copy) {t_wg_nn:.3f} "
              f"(copy alone {t_tr:.3f}) | wgrad TN(view, stock) {t_wg_tn:.3f} -> NN/TN x{t_wg_nn / t_wg_tn:.2f}",
              flush=True)
dist.barrier()
try:
    mpi.MPI_Finalize()
except Exception:
    pass
