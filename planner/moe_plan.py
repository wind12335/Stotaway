"""Predictor, MoE side (single card, seconds, no communication context): choose the expert-MLP path per shape.

Candidates for the local expert MLP (fc1 -> SwiGLU x probs -> fc2, forward + backward):
  stock    Megatron SequentialMLP (tokens_per_expert.tolist() + one GEMM pair per expert)
  gg[B]    sync-free dropless grouped GEMM, virtual group size B (src/stowaway/kernels/moe_expert_mlp.py)
B matters because hipBLASLt picks its algorithm from the grid shape (11-01: B=512/1024 up to 1.8x slower
than B=2048 on Qwen3-30B-A3B expert shapes). The routing skew barely matters for gg (fixed grid, F53), so
the plan is made on a skewed synthetic routing and holds for any routing.

usage: python3 moe_plan.py --experts 16 --hidden 2048 --ffn 768 --rows 32768 [--out moe_plan.json]
"""
import argparse, json, os, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def timeit(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=768)
    ap.add_argument("--rows", type=int, default=32768)
    ap.add_argument("--skew", type=float, default=0.8)
    ap.add_argument("--bs", default="1024,2048,4096")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "moe_plan.json"))
    a = ap.parse_args()
    from stowaway.kernels.moe_expert_mlp import get_plan, expert_linear, _ptr_table
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    E, H, Fe, T = a.experts, a.hidden, a.ffn, a.rows
    w_ = torch.distributions.LogNormal(0, a.skew).sample((E,))
    s = (w_ / w_.sum() * T).round().long()
    s[-1] += T - s.sum()
    offs = torch.cumsum(s, 0).to(dev)
    W1 = [(torch.randn(2 * Fe, H, device=dev, dtype=dt) * 0.02).requires_grad_() for _ in range(E)]
    W2 = [(torch.randn(H, Fe, device=dev, dtype=dt) * 0.02).requires_grad_() for _ in range(E)]
    x0 = torch.randn(T, H, device=dev, dtype=dt) * 0.5
    probs = torch.rand(T, device=dev)
    g = torch.randn(T, H, device=dev, dtype=dt)
    t1, t2 = _ptr_table(W1), _ptr_table(W2)

    def run_gg(B):
        p1, p2 = get_plan(E, H, 2 * Fe, T, B), get_plan(E, Fe, H, T, B)

        def f():
            for w in W1 + W2:
                w.grad = None
            x = x0.clone().requires_grad_()
            h = expert_linear(x, offs, W1, p1, t1)
            u, v = h.chunk(2, dim=-1)
            y = expert_linear((F.silu(u) * v * probs[:, None]).to(dt).contiguous(), offs, W2, p2, t2)
            (y.float() * g.float()).sum().backward()
        return timeit(f)

    def run_stock():
        def f():
            for w in W1 + W2:
                w.grad = None
            x = x0.clone().requires_grad_()
            l = offs.diff(prepend=offs.new_zeros(1)).tolist()
            outs = []
            for e, (xe, pe) in enumerate(zip(torch.split(x, l), torch.split(probs, l))):
                h = xe @ W1[e].t()
                u, v = h.chunk(2, dim=-1)
                outs.append((F.silu(u) * v * pe[:, None]).to(dt) @ W2[e].t())
            (torch.cat(outs).float() * g.float()).sum().backward()
        return timeit(f)

    res = {"stock": run_stock()}
    for B in [int(b) for b in a.bs.split(",")]:
        res[f"gg{B}"] = run_gg(B)
    best = min(res, key=res.get)
    plan = dict(shape=dict(experts=E, hidden=H, ffn=Fe, rows=T), times_ms=res, choice=best,
                speedup_vs_stock=res["stock"] / res[best])
    print(json.dumps(plan, indent=1))
    db = json.load(open(a.out)) if os.path.exists(a.out) else {}
    db[f"E{E}_H{H}_F{Fe}_T{T}"] = plan
    json.dump(db, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
