"""Summarise bench/results/opus_vllm_sp: per arm median prefill time per cell, speedup vs stock and vs
the SP-form original (rccl), and the output gate vs stock.  usage: opus_vllm_sp_report.py <dir> <M>"""
import csv, json, os, statistics, sys

d, M = sys.argv[1], int(sys.argv[2])
rows = [r for r in csv.DictReader(open(os.path.join(d, "vsp.csv"))) if int(r["M"]) == M]
arms = [a for a in ("stock", "rccl", "dx", "plan") if any(r["arm"] == a for r in rows)]
cells = [c for c in ("ttft", "ttft2", "thru") if any(r["cell"] == c for r in rows)]
med = {(a, c): statistics.median(float(r["wall_s"]) for r in rows if r["arm"] == a and r["cell"] == c)
       for a in arms for c in cells if any(r["arm"] == a and r["cell"] == c for r in rows)}
print(f"## vLLM llama3-70B prefill, M = {M}（每格为所有遍的中位，单位 s；括号内 = 原版 / 该臂）\n")
print("| 臂 | " + " | ".join(cells) + " |")
print("|---|" + "---|" * len(cells))
for a in arms:
    cs = []
    for c in cells:
        if (a, c) in med:
            sp = med[("stock", c)] / med[(a, c)] if ("stock", c) in med else float("nan")
            cs.append(f"{med[(a, c)]:.3f}（{sp:.3f}×）")
        else:
            cs.append("—")
    print(f"| {a} | " + " | ".join(cs) + " |")
if ("rccl", "ttft") in med:
    print("\n相对序列并行原版（rccl）：" + "；".join(
        f"{a} {c} {med[('rccl', c)] / med[(a, c)]:.3f}×" for a in ("dx", "plan") for c in cells if (a, c) in med))
g0 = os.path.join(d, f"gate_stock_M{M}.json")
if os.path.exists(g0):
    ref = json.load(open(g0))
    print("\n输出检查（贪心 32 个 token，对 stock）：")
    for a in arms[1:]:
        p = os.path.join(d, f"gate_{a}_M{M}.json")
        if not os.path.exists(p):
            continue
        g = json.load(open(p))
        div = next((i for i, (x, y) in enumerate(zip(ref["ids"], g["ids"])) if x != y), None)
        n = div if div is not None else len(ref["ids"])
        dl = [abs(x - y) for x, y in zip(ref["logprobs"][:n], g["logprobs"][:n]) if x is not None and y is not None]
        print(f"- {a}: 第一个不同的 token 位置 = {div if div is not None else '无（32 个全同）'}，"
              f"相同前缀上 max|logprob 差| = {max(dl) if dl else float('nan'):.4f}")
