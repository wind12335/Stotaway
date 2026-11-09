"""Opus PERM rig -- is the 8-card 'unicast fan-out wall' a real wall, or
just per-rank injection bandwidth divided across flows?

All 8 ranks act simultaneously; engine hipMemcpyAsync pushes, 64MB chunks.
  shift_j   : every rank pushes to (r+j)%R   (a permutation), j=1..7
  seq7      : every rank pushes 7 chunks back-to-back on ONE stream to
              peers r+1..r+7 (= pairwise-exchange RS transport order)
  seq7_perm : same but step j targets (r+j): identical to seq7 (sanity)
  fan7      : 7 chunks to 7 peers concurrently on 7 streams (fan-out)
  ring7     : 7 chunks back-to-back to r+1 only (ring transport, no adds)
Reports per-rank GB/s = bytes_sent / device time.  No correctness here
(transport only; correctness of SM/engine pushes is opus_smx).
"""
import ctypes, os, statistics, sys
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_opus_perm_{os.environ.get('OMPI_COMM_WORLD_RANK','0')}")
import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
R = pHost.mpi_comm_size(mpi, comm)
hipl = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so"))
hipl.hipSetDevice.restype = ctypes.c_int
assert hipl.hipSetDevice(rank) == 0
import torch  # noqa: E402
torch.cuda.set_device(rank)
import torch.distributed as dist  # noqa: E402
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29971")
dist.init_process_group("nccl", rank=rank, world_size=R)
_x = torch.ones(1, device="cuda")
for _ in range(3):
    dist.all_reduce(_x)
torch.cuda.synchronize()
from stowaway.kernels.common_ops import init_dcu, memcpy_async  # noqa: E402
host = pHost.DushmemHost(preload_bootstrap=False)
host.init_mpi(comm)
init_dcu(mpi, comm, rank, R, host=host)

CH = int(os.environ.get("PERM_MB", "64")) << 20
src = host.malloc(CH)
dst = host.malloc(R * CH)                  # slot per source
peer_dst = [host.ptr(dst, d) for d in range(R)]
st = torch.cuda.current_stream()
streams = [torch.cuda.Stream() for _ in range(R)]


def slot(d, s):
    return peer_dst[d] + s * CH


def run(fn, reps=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        dist.barrier()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(st)
        fn()
        e1.record(st)
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def shift(j):
    d = (rank + j) % R
    return lambda: memcpy_async(slot(d, rank), src, CH, st.cuda_stream)


def seq7():
    for j in range(1, R):
        d = (rank + j) % R
        memcpy_async(slot(d, rank), src, CH, st.cuda_stream)


def ring7():
    d = (rank + 1) % R
    for j in range(1, R):
        memcpy_async(slot(d, j), src, CH, st.cuda_stream)


ev = torch.cuda.Event()


def fan7():
    ev.record(st)
    for j in range(1, R):
        streams[j].wait_event(ev)
        d = (rank + j) % R
        memcpy_async(slot(d, rank), src, CH, streams[j].cuda_stream)
    for j in range(1, R):
        st.wait_stream(streams[j])


def seq7_incast():
    """the R12e a2a destination order: j = 0..R-1 skipping self -> at
    step j every other rank targets the SAME rank j (7-to-1 incast)"""
    for d in range(R):
        if d != rank:
            memcpy_async(slot(d, rank), src, CH, st.cuda_stream)


def incast_only():
    """all ranks except 0 push to rank 0 at once (pure 7-to-1)"""
    if rank != 0:
        memcpy_async(slot(0, rank), src, CH, st.cuda_stream)


res = {}
if os.environ.get("PERM_ONLY_INCAST"):
    res["seq7_incast"] = (run(seq7_incast), R - 1)
    res["incast_only"] = (run(incast_only), 1 if rank else 0)
    res["seq7"] = (run(seq7), R - 1)
for j in range(1, R) if not os.environ.get("PERM_ONLY_INCAST") else []:
    res[f"shift_{j}"] = (run(shift(j)), 1)
if not os.environ.get("PERM_ONLY_INCAST"):
    res["seq7"] = (run(seq7), R - 1)
    res["ring7"] = (run(ring7), R - 1)
    res["fan7"] = (run(fan7), R - 1)
mine = {k: round(max(n, 1) * CH / t / 1e6, 1) for k, (t, n) in res.items()}
allr = [None] * R
dist.all_gather_object(allr, mine)
if rank == 0:
    for k in mine:
        v = [a[k] for a in allr]
        print(f"[perm] {k:8s} per-rank GB/s min/med/max "
              f"{min(v)}/{statistics.median(v)}/{max(v)}", flush=True)
dist.barrier()
try:
    mpi.MPI_Finalize()
except Exception:
    pass
