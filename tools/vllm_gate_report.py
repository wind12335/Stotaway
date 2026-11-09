"""Compare natural-text gate outputs (e2e/vllm/vllm_gate.py) of every arm against stock.

  python3 tools/vllm_gate_report.py <dir> <M>

Per arm and prompt: tokens identical to stock out of OSL, first divergence, and on the common prefix
the max |logprob diff| of the token stock chose (when it is in the arm's top 5). The first generated
token comes purely from the M-token prefill (the SP forward), so its logprob diff is reported separately.
"""
import glob, json, os, sys

d, M = sys.argv[1], int(sys.argv[2])
arms = {}
for p in sorted(glob.glob(os.path.join(d, f"gatetext_*_M{M}.json"))):
    j = json.load(open(p))
    arms[j["arm"]] = j
if "stock" not in arms:
    sys.exit("no stock gate file")
ref = arms["stock"]["prompts"]
print(f"## 自然文本输出检查（M = {M}，每臂 {len(ref)} 个提示 × {arms['stock']['osl']} 个贪心 token，对 stock）\n")
print("| 臂 | 与 stock 完全相同的 token | 各提示第一个不同位置 | 首 token logprob 差（只来自 prefill） | 相同前缀上 max logprob 差 |")
print("|---|---|---|---|---|")
for a, j in arms.items():
    if a == "stock":
        continue
    same = tot = 0
    firsts, d0, dmax = [], 0.0, 0.0
    for r, x in zip(ref, j["prompts"]):
        n = len(r["ids"])
        tot += n
        f = next((i for i, (p, q) in enumerate(zip(r["ids"], x["ids"])) if p != q), None)
        firsts.append("无" if f is None else str(f))
        same += sum(p == q for p, q in zip(r["ids"], x["ids"]))
        for i in range(n if f is None else f):
            t = str(r["ids"][i])
            if t in x["top5"][i]:
                diff = abs(r["top5"][i][t] - x["top5"][i][t])
                dmax = max(dmax, diff)
                if i == 0:
                    d0 = max(d0, diff)
    print(f"| {a} | {same}/{tot} | {' / '.join(firsts)} | {d0:.4f} | {dmax:.4f} |")
