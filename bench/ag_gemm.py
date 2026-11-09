"""Operator-level bench: AG+GEMM (column-parallel input, QKV / fc1 call sites), PRODUCTION op
stowaway.kernels.gemm_rs_dx.dx_ag_gemm_op (the same function the Megatron hook calls).

  stock = RCCL all_gather_into_tensor + vendor GEMM (serial)
  ours  = DX overlap: blocked engine push in permutation order + per-block vendor GEMM, ticket-gated
Env: AGF_M rows (seq x MBS), AGF_K = hidden, AGF_NCS = comma list of local output widths
     (fc1 = 2*ffn/R, qkv = (heads + 2 kv) * head_dim / R), AGF_OPS = correctness ops (fresh input
     every op, back-to-back bursts of AGF_BURST without sync), AGF_STRESS=1 stress mode.
Prints '[agf] correctness ...' and '[agf] Nc=... stock ... | dx-ag-gemm ...' (synced and
back-to-back x16 timings, slowest rank). Driver: bench/run_ag_matrix.sh.
"""
import ctypes, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_opus_agf_{os.environ.get('OMPI_COMM_WORLD_RANK','0')}")
import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm); R = pHost.mpi_comm_size(mpi, comm)
hipl = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so")); hipl.hipSetDevice.restype = ctypes.c_int
assert hipl.hipSetDevice(rank) == 0
import torch  # noqa: E402
torch.cuda.set_device(rank)
import torch.distributed as dist  # noqa: E402
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29987")
dist.init_process_group("nccl", rank=rank, world_size=R)
_x = torch.ones(1, device="cuda")
for _ in range(3):
    dist.all_reduce(_x)
torch.cuda.synchronize()
import stowaway  # noqa: E402
from stowaway.kernels.common_ops import init_dcu  # noqa: E402
from stowaway.kernels.gemm_rs_dx import DXAGGemmContext, dx_ag_gemm_op  # noqa: E402
host = pHost.DushmemHost(preload_bootstrap=False); host.init_mpi(comm)
init_dcu(mpi, comm, rank, R, host=host)
if rank == 0:
    print(f"[agf] stowaway from {stowaway.__file__}", flush=True)

M, K = int(os.environ.get("AGF_M", "32768")), int(os.environ.get("AGF_K", "5120"))
NCS = [int(x) for x in os.environ.get("AGF_NCS", "4352,896").split(",")]   # fc1 (swiglu), qkv
OPS = int(os.environ.get("AGF_OPS", "64"))
BURST = int(os.environ.get("AGF_BURST", "4"))
DT = torch.bfloat16
mp = M // R
ctx = DXAGGemmContext(M, K, DT, rank, R)
W = {}
for Nc in NCS:
    torch.manual_seed(700 + Nc + rank)
    W[Nc] = torch.randn(Nc, K, dtype=DT, device="cuda") * 0.05


def allmax(v):
    t = torch.tensor([float(v)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def stock(a_sh, w):
    full = torch.empty(M, K, dtype=DT, device="cuda")
    dist.all_gather_into_tensor(full, a_sh)
    return torch.matmul(full, w.t()), full


g = torch.Generator(device="cuda")
STRESS = os.environ.get("AGF_STRESS", "") == "1"
abuf = torch.empty(mp, K, dtype=DT, device="cuda")
dly = torch.randn(4096, 4096, dtype=DT, device="cuda")
fails = 0
w_full = w_out = w_blk = 0.0
op = 0
while op < OPS:
    batch = []
    for b in range(BURST):
        Nc = NCS[(op + b) % len(NCS)]
        g.manual_seed(30000 * (op + b) + rank)
        a = (torch.randn(mp, K, generator=g, dtype=torch.float32, device="cuda") * 0.05).to(DT)
        batch.append((Nc, a))
    torch.cuda.synchronize()
    if STRESS:
        # ordering stress (11-01): every op's input is written into ONE reused buffer on the
        # caller stream right before the op, behind a delay GEMM, with no sync. A fork that is
        # not ordered after the producer pushes stale data; a join that does not hold the
        # caller until the pushes are done lets the next overwrite corrupt in-flight pushes.
        outs = []
        for Nc, a in batch:
            torch.matmul(dly, dly)
            abuf.copy_(a)
            o, f = dx_ag_gemm_op(abuf, W[Nc], ctx, need_full=True)
            outs.append((o, f))
    else:
        outs = [dx_ag_gemm_op(a, W[Nc], ctx, need_full=True) for Nc, a in batch]   # back-to-back
    torch.cuda.synchronize()
    for (Nc, a), (out, full) in zip(batch, outs):
        rout, rfull = stock(a, W[Nc])
        # block-GEMM reference (same row blocks as ours) -> must be bit-exact
        rblk = torch.cat([torch.matmul(rfull[i * mp:(i + 1) * mp], W[Nc].t()) for i in range(R)])
        d_full = allmax((full.float() - rfull.float()).abs().max().item())
        d_blk = allmax((out.float() - rblk.float()).abs().max().item())
        d_out = allmax(((out.float() - rout.float()).abs() / rout.float().abs().clamp_min(1.0)).max().item())
        w_full, w_blk, w_out = max(w_full, d_full), max(w_blk, d_blk), max(w_out, d_out)
        if d_full != 0 or d_blk != 0:
            fails += 1
            if rank == 0 and fails <= 5:
                print(f"[agf] FAIL Nc={Nc} full {d_full:g} blk {d_blk:g}", flush=True)
    op += BURST
if rank == 0:
    print(f"[agf] correctness ops={op} fails={fails} max|full-ref|={w_full:g} max|out-blockref|={w_blk:g} "
          f"(both must be 0) rel(out, full-GEMM ref)={w_out:.3e} -> {'PASS' if fails == 0 else 'FAIL'}", flush=True)

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


for Nc in NCS:
    a = torch.randn(mp, K, dtype=DT, device="cuda") * 0.05
    full = torch.empty(M, K, dtype=DT, device="cuda")
    t_ag = tm(lambda: dist.all_gather_into_tensor(full, a))
    t_mm = tm(lambda: torch.matmul(full, W[Nc].t()))
    t_st = tm(lambda: stock(a, W[Nc]))
    t_dx = tm(lambda: dx_ag_gemm_op(a, W[Nc], ctx))
    t_st16 = tm(lambda: [stock(a, W[Nc]) for _ in range(16)], it=8) / 16
    t_dx16 = tm(lambda: [dx_ag_gemm_op(a, W[Nc], ctx) for _ in range(16)], it=8) / 16
    if rank == 0:
        print(f"[agf] Nc={Nc} stock {t_st:.3f} (ag {t_ag:.3f} + gemm {t_mm:.3f}) | dx-ag-gemm {t_dx:.3f} "
              f"-> x{t_st / t_dx:.3f} | b2b/op stock {t_st16:.3f} dx {t_dx16:.3f} x{t_st16 / t_dx16:.3f}",
              flush=True)
dist.barrier()
try:
    mpi.MPI_Finalize()
except Exception:
    pass
