"""11-01 mechanism probe: does hipMemcpyAsync block the host once too many engine copies are
queued but not yet executed (in-flight copy cap)?

Why: in training, a fused AG call's 7 peer pushes cost 4-5 ms of host time (vs 0.06 ms isolated),
the cost scales with how far the host runs ahead of the GPU, and it does not change with the slot
plane count (C, P=2/4) nor with moving the ack waits off the push stream (D, OPUS_AG_GATE=ev).
A per-device cap on queued-but-unexecuted copies would explain all three.

Method (2 GPUs, no MPI): put a long sleep kernel on stream S of GPU 0 so nothing behind it runs,
then enqueue N copies and time each hipMemcpyAsync on the host. The first call whose host time
jumps to ~the sleep length marks the cap. Variants:
  peer   : 1 MB copies GPU0 -> GPU1 on S (behind the sleep)
  peer_l : same on a second stream L that waits on an event recorded after the sleep
  local  : 1 MB copies GPU0 -> GPU0 on S
  tiny   : 4 B peer copies on S
  kern   : small elementwise kernels on S (control: launch queue cap)
Output: per variant, index of first blocked call and its host time.
"""
import ctypes, os, time
import torch

h = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so"))
for f in ("hipMemcpyAsync", "hipDeviceEnablePeerAccess", "hipSetDevice", "hipEventRecord",
          "hipStreamWaitEvent", "hipEventCreateWithFlags"):
    getattr(h, f).restype = ctypes.c_int
N = int(os.environ.get("CC_N", "400"))
SLEEP_MS = float(os.environ.get("CC_SLEEP_MS", "400"))

torch.cuda.set_device(0)
torch.cuda.init()
h.hipSetDevice(0)
rc = h.hipDeviceEnablePeerAccess(1, 0)
print("enable peer rc", rc, flush=True)
src = torch.empty(1 << 20, dtype=torch.uint8, device="cuda:0")
dst1 = torch.empty(N << 20, dtype=torch.uint8, device="cuda:1")
dst0 = torch.empty(N << 20, dtype=torch.uint8, device="cuda:0")
x = torch.zeros(1024, device="cuda:0")

# calibrate _sleep cycles -> ms
S = torch.cuda.Stream(device=0)
L = torch.cuda.Stream(device=0)
torch.cuda.synchronize()
t = time.perf_counter()
with torch.cuda.stream(S):
    torch.cuda._sleep(10_000_000)
torch.cuda.synchronize()
cyc_per_ms = 10_000_000 / ((time.perf_counter() - t) * 1e3)
CYC = int(cyc_per_ms * SLEEP_MS)
ev = ctypes.c_void_p()
assert h.hipEventCreateWithFlags(ctypes.byref(ev), 2) == 0   # disable timing


def run(name):
    torch.cuda.synchronize()
    with torch.cuda.stream(S):
        torch.cuda._sleep(CYC)
    st = ctypes.c_void_p(S.cuda_stream)
    if name == "peer_l":
        assert h.hipEventRecord(ev, st) == 0
        assert h.hipStreamWaitEvent(ctypes.c_void_p(L.cuda_stream), ev, 0) == 0
        st = ctypes.c_void_p(L.cuda_stream)
    ts = []
    t_all = time.perf_counter()
    for i in range(N):
        t0 = time.perf_counter()
        if name in ("peer", "peer_l"):
            rc = h.hipMemcpyAsync(ctypes.c_void_p(dst1.data_ptr() + (i << 20)), ctypes.c_void_p(src.data_ptr()),
                                  1 << 20, 3, st)
        elif name == "local":
            rc = h.hipMemcpyAsync(ctypes.c_void_p(dst0.data_ptr() + (i << 20)), ctypes.c_void_p(src.data_ptr()),
                                  1 << 20, 3, st)
        elif name == "tiny":
            rc = h.hipMemcpyAsync(ctypes.c_void_p(dst1.data_ptr() + 4 * i), ctypes.c_void_p(src.data_ptr()), 4, 3, st)
        else:
            with torch.cuda.stream(S):
                x.add_(1.0)
            rc = 0
        assert rc == 0
        ts.append((time.perf_counter() - t0) * 1e3)
        if (time.perf_counter() - t_all) * 1e3 > 0.8 * SLEEP_MS:
            break
    torch.cuda.synchronize()
    blk = next((i for i, v in enumerate(ts) if i > 0 and v > 5.0), None)
    pre = sorted(ts[1:blk] if blk else ts[1:])
    med = pre[len(pre) // 2] if pre else float("nan")
    tot = sum(ts)
    print(f"[cc] {name:6s} sleep {SLEEP_MS:.0f} ms | calls issued {len(ts)} | first blocked call (>5 ms, excl. #0): "
          f"{blk} ({ts[blk] if blk else 0:.1f} ms) | median host/call before it {med:.3f} ms | "
          f"call#0 {ts[0]:.2f} ms | max {max(ts[1:]):.1f} ms | total {tot:.1f} ms", flush=True)


for name in ("kern", "local", "peer", "peer_l", "tiny", "peer"):
    run(name)
