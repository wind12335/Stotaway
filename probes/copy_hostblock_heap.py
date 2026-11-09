"""11-01 probe 2: the copy-cap probe (opus_copycap.py / cc4) used ordinary torch memory on the peer
GPU as destination and found that a value-gated fork stops the host block. In training the fused
AG pushes into the PEER'S dushmem symmetric heap (host.ptr VA) and still block (batch F:
push_data 3.2-4.4 ms/call with OPUS_AG_FORK=wv). This probe repeats the measurement with heap
destinations. 2 ranks (mpirun -np 2), rank 0 measures, rank 1 idles.

Rank 0, for each variant: 400 ms sleep on stream S, then 40 ops, each: kernel on S, value-gated
fork S -> lane, 7 x 1 MB copies + 7 x 4 B tickets on the lane. Host time per category.
  heap_peer : destination = rank 1's heap (host.ptr(buf, 1))
  heap_own  : destination = rank 0's own heap
  plain_own : destination = ordinary torch memory on rank 0
and the same three with the old event-gated fork.
"""
import collections, ctypes, os, sys, time
import pydushmem.host as pHost
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm); R = pHost.mpi_comm_size(mpi, comm)
hipl = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so")); hipl.hipSetDevice.restype = ctypes.c_int
assert hipl.hipSetDevice(rank) == 0
import torch
torch.cuda.set_device(rank)
from stowaway.kernels.common_ops import init_dcu, hip
host = pHost.DushmemHost(preload_bootstrap=False); host.init_mpi(comm)
init_dcu(mpi, comm, rank, R, host=host)
h = hip()
h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
h.hipEventCreateWithFlags.restype = ctypes.c_int
NB = int(os.environ.get("HP_MB", "1")) << 20
heap = host.malloc(400 * NB + 4096)
from pydushmem.ipc import IpcPtrList
ipcl = IpcPtrList(300 * NB + 4096, mpi, comm, R, rank)
assert heap
torch.cuda.synchronize(); host.barrier_all()
if rank == 0:
    src = torch.empty(NB, dtype=torch.uint8, device="cuda")
    tk = torch.arange(1024, dtype=torch.int32, device="cuda")
    plain = torch.empty(400 * NB + 4096, dtype=torch.uint8, device="cuda")
    flag = torch.zeros(64, dtype=torch.int32, device="cuda")
    x = torch.zeros(1024, device="cuda")
    S = torch.cuda.Stream(); Ln = torch.cuda.Stream()
    sh = ctypes.c_void_p(S.cuda_stream); lh = ctypes.c_void_p(Ln.cuda_stream)
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.cuda.stream(S):
        torch.cuda._sleep(10_000_000)
    torch.cuda.synchronize(); cpm = 10_000_000 / ((time.perf_counter() - t) * 1e3)
    evs = []
    for _ in range(64):
        e = ctypes.c_void_p(); assert h.hipEventCreateWithFlags(ctypes.byref(e), 2) == 0; evs.append(e)
    dst_base = {"heap_peer": host.ptr(heap, 1), "heap_own": host.ptr(heap, 0), "plain_own": plain.data_ptr(), "ipc_peer": ipcl.ptrs[1]}

    class PA(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("device", ctypes.c_int), ("devicePointer", ctypes.c_void_p),
                    ("hostPointer", ctypes.c_void_p), ("isManaged", ctypes.c_int), ("allocationFlags", ctypes.c_uint)]
    h.hipPointerGetAttributes.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipPointerGetAttributes.restype = ctypes.c_int
    for nm, p in (("plain_own", plain.data_ptr()), ("heap_own", host.ptr(heap, 0)), ("heap_peer", host.ptr(heap, 1))):
        a = PA(); rc = h.hipPointerGetAttributes(ctypes.byref(a), ctypes.c_void_p(p))
        print(f"[hp] attr {nm}: rc={rc} type={a.type} device={a.device} dptr={a.devicePointer} hptr={a.hostPointer}", flush=True)

    def T(f):
        t0 = time.perf_counter(); f(); return (time.perf_counter() - t0) * 1e3

    def run(dname, fork, sleep_ms=400, nops=40):
        global MODE
        torch.cuda.synchronize(); flag.zero_(); torch.cuda.synchronize()
        with torch.cuda.stream(S):
            torch.cuda._sleep(int(cpm * sleep_ms))
        base = dst_base[dname]
        acc = collections.defaultdict(float); tall = time.perf_counter()
        for o in range(nops):
            with torch.cuda.stream(S):
                x.add_(1.0)
            if fork == "wv":
                acc["fork"] += T(lambda: (h.hipStreamWriteValue32(sh, ctypes.c_void_p(flag.data_ptr()), ctypes.c_uint32(o + 1), 0),
                                          h.hipStreamWaitValue32(lh, ctypes.c_void_p(flag.data_ptr()), ctypes.c_uint32(o + 1), 0, 0xFFFFFFFF)))
            else:
                e = evs[o % 64]
                acc["fork"] += T(lambda: (h.hipEventRecord(e, sh), h.hipStreamWaitEvent(lh, e, 0)))
            for k in range(7):
                i = (o * 7 + k) % 280
                if MODE == "peerapi":
                    acc["data"] += T(lambda: h.hipMemcpyPeerAsync(ctypes.c_void_p(base + i * NB), 1, ctypes.c_void_p(src.data_ptr()), 0, ctypes.c_size_t(NB), lh))
                elif MODE == "pull":
                    acc["data"] += T(lambda: h.hipMemcpyAsync(ctypes.c_void_p(plain.data_ptr() + i * NB), ctypes.c_void_p(base + i * NB), NB, 3, lh))
                elif MODE == "default":
                    acc["data"] += T(lambda: h.hipMemcpyAsync(ctypes.c_void_p(base + i * NB), ctypes.c_void_p(src.data_ptr()), NB, 4, lh))
                elif MODE == "dtod":
                    acc["data"] += T(lambda: h.hipMemcpyDtoDAsync(ctypes.c_void_p(base + i * NB), ctypes.c_void_p(src.data_ptr()), ctypes.c_size_t(NB), lh))
                else:
                    acc["data"] += T(lambda: h.hipMemcpyAsync(ctypes.c_void_p(base + i * NB), ctypes.c_void_p(src.data_ptr()), NB, 3, lh))
                acc["tick"] += T(lambda: h.hipMemcpyAsync(ctypes.c_void_p(base + 290 * NB + 4 * k), ctypes.c_void_p(tk.data_ptr() + 4 * k), 4, 3, lh))
        tot = (time.perf_counter() - tall) * 1e3
        torch.cuda.synchronize()
        print(f"[hp] {dname:9s} fork={fork} | host {tot:7.1f} ms | " + " ".join(f"{k}={v:.1f}" for k, v in acc.items()), flush=True)

    h.hipMemcpyDtoDAsync.restype = ctypes.c_int
    h.hipMemcpyPeerAsync.restype = ctypes.c_int
    pa = ctypes.c_int(); h.hipDeviceCanAccessPeer.restype = ctypes.c_int
    h.hipDeviceCanAccessPeer(ctypes.byref(pa), 0, 1); print(f"[hp] canAccessPeer(0,1)={pa.value} enablePeer rc={h.hipDeviceEnablePeerAccess(1, 0)}", flush=True)
    MODE = "d2d"; run("ipc_peer", "wv", 100, 10)
    for MODE in ("d2d", "peerapi"):
        for sl in (100, 400):
            print(f"[hp] ---- mode {MODE} sleep {sl} ms, 40 ops x 7 x {NB >> 20} MB", flush=True)
            for d in ("heap_peer", "ipc_peer"):
                run(d, "wv", sl, 40)
torch.cuda.synchronize(); host.barrier_all()
ipcl.free()
try:
    mpi.MPI_Finalize()
except Exception:
    pass
