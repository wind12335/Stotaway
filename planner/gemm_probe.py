"""Stage 2 (single card): vendor GEMM time of every forward TP call site of the 27B and 70B
training shapes. This is the only plan-time input the cost model needs besides the shape;
it builds no symmetric-heap context, so it cannot disturb anything (unlike the online probe, F44).
Megatron TP=8 + SP, seq 4096 x MBS 8 -> M = 32768, bf16.
Timing: back-to-back (16 ops in a row, total / 16), median of 8 repeats; also single-shot median of 25.
Output: gemm_probe.json (M32768) or $GP_OUT, e.g. GP_M=16384 GP_OUT=gemm_probe_m16384.json
"""
import json, os, statistics, time
import torch

R, M = 8, int(os.environ.get("GP_M", "32768"))
SHAPES = {   # hidden, ffn, heads, kv groups (head_dim = hidden / heads)
    "27b": (5120, 17408, 40, 8),
    "70b": (8192, 28672, 64, 8),
}
dt = torch.bfloat16


def sites(h, f, nh, kv):
    hd = h // nh
    return [  # (name, op, M, K, N) ; AG sites: A gathered [M, K] x W [K, N]; RS sites: [M, K] x [K, N] then RS
        ("qkv", "ag_up", M, h, (nh + 2 * kv) * hd // R),
        ("o_proj", "rs", M, h // R, h),
        ("fc1_gate_up", "ag_gu", M, h, 2 * f // R),
        ("fc2", "rs", M, f // R, h),
    ]


def bench(fn, n=25):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    single = statistics.median(ts)
    bb = []
    for _ in range(8):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(16):
            fn()
        e.record(); torch.cuda.synchronize()
        bb.append(s.elapsed_time(e) / 16)
    return single, statistics.median(bb)


out = []
for model, (h, f, nh, kv) in SHAPES.items():
    for name, op, m, k, n in sites(h, f, nh, kv):
        a = torch.randn(m, k, dtype=dt, device="cuda") * 0.05
        w = torch.randn(k, n, dtype=dt, device="cuda") * 0.05
        c = torch.empty(m, n, dtype=dt, device="cuda")
        g1, gb = bench(lambda: torch.matmul(a, w, out=c))
        B = (m * n * 2 if op == "rs" else m * k * 2) / 1e6
        out.append(dict(model=model, site=name, op=op, M=m, K=k, N=n, B=round(B, 3),
                        G_single=round(g1, 4), G_b2b=round(gb, 4),
                        TF=round(2 * m * k * n / (gb * 1e-3) / 1e12, 1)))
        print(out[-1], flush=True)
        del a, w, c
        torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", os.environ.get("GP_OUT", "gemm_probe.json")), "w"), indent=1)
print("done", time.strftime("%T"))
