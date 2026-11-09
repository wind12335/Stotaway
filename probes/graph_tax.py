"""Single-GPU probe (10-31, candidate (b) pre-gate): graph-replay tax at the
DX node count. Op = DX RS skeleton without the cross-card protocol:
8 block vendor GEMMs (permutation order, own block last) on the caller
stream + 7 block D2D copies on a lane stream, each waiting its block's
event, + an fp32 8-way sum. Eager vs torch.cuda.graph replay (non-default
stream capture, T2c law). Gate: graph replay <= eager DEVICE time + 0.1 ms
(the per-node tax measured in T18 was +1.0-1.35 ms at a comparable count).
"""
import os, statistics, sys, time
import torch
dev = int(os.environ.get("GT_DEV", "0"))
torch.cuda.set_device(dev)
R = 8
SHAPES = {"llama3-70b": (8192, 3584), "qwen3-8b": (4096, 1536), "llama31-405b": (16384, 6656)}
MS = [int(x) for x in os.environ.get("GT_MS", "2048,4096,8192").split(",")]
dt = torch.float16


def run(key, M):
    N, K = SHAPES[key]
    mp = M // R
    a = torch.randn(M, K, dtype=dt, device="cuda") * 0.05
    b = torch.randn(N, K, dtype=dt, device="cuda") * 0.05
    part = torch.empty(M, N, dtype=dt, device="cuda")
    slots = torch.empty(R, mp, N, dtype=dt, device="cuda")
    out = torch.empty(mp, N, dtype=dt, device="cuda")
    lane = torch.cuda.Stream()
    me = 0

    def op():
        cur = torch.cuda.current_stream()
        evs = []
        for j in range(1, R + 1):
            d = (me + j) % R
            torch.matmul(a[d * mp:(d + 1) * mp], b.t(), out=part[d * mp:(d + 1) * mp])
            e = torch.cuda.Event()
            e.record(cur)
            evs.append((d, e))
        lane.wait_stream(cur) if False else None
        for d, e in evs[:-1]:
            lane.wait_event(e)
            with torch.cuda.stream(lane):
                slots[d].copy_(part[d * mp:(d + 1) * mp], non_blocking=True)
        cur.wait_stream(lane)
        torch.sum(slots.float(), dim=0, out=None).to(dt, copy=False)

    def gemm1():
        torch.matmul(a, b.t(), out=part)

    def tmeas(fn, n=25):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ds, hs = [], []
        for _ in range(n):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            t0 = time.perf_counter()
            fn()
            hs.append((time.perf_counter() - t0) * 1e3)
            e.record()
            torch.cuda.synchronize()
            ds.append(s.elapsed_time(e))
        return statistics.median(ds), statistics.median(hs)

    cap = torch.cuda.Stream()
    with torch.cuda.stream(cap):
        eager_dev, eager_host = tmeas(op)
        g1_dev, _ = tmeas(gemm1)
        # device-only time of the eager op = back-to-back throughput (host hidden)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(16):
            op()
        e.record()
        torch.cuda.synchronize()
        eager_b2b = s.elapsed_time(e) / 16
        for _ in range(2):
            op()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cap):
            op()
        torch.cuda.synchronize()
        gr_dev, gr_host = tmeas(g.replay)
        s.record()
        for _ in range(16):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        gr_b2b = s.elapsed_time(e) / 16
    print(f"[gt] {key} M{M} gemm1 {g1_dev:.3f} | eager synced {eager_dev:.3f} (host {eager_host:.3f}) b2b {eager_b2b:.3f} "
          f"| graph synced {gr_dev:.3f} (host {gr_host:.3f}) b2b {gr_b2b:.3f} | tax(graph-eager_b2b) {gr_b2b - eager_b2b:+.3f}",
          flush=True)


for key in os.environ.get("GT_MODELS", "qwen3-8b,llama3-70b,llama31-405b").split(","):
    for M in MS:
        run(key, M)
print("[gt] done")
