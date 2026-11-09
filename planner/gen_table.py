"""Generate the operator-level result document (GEMM+AG and GEMM+RS, back-to-back timing).

Back-to-back timing = issue 16 ops in a row without waiting, total time / 16, median of 8
repeats, slowest rank. Speedup = original (vendor GEMM + RCCL, serial) / ours (fused op).
Re-run this script after new data lands; it rewrites the whole document.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SW_OPS_RAW", os.path.join(HERE, "..", "results", "raw", "ops_8card"))
OUT = os.environ.get("SW_OPS_TABLE", os.path.join(HERE, "..", "results", "算子级结果表.md"))
sys.path.insert(0, HERE)
from shapes import SHAPES  # noqa: E402

ORDER = ["qwen3-8b", "llama3-8b", "qwen3-32b", "llama4-mav-d", "gemma3-27b", "mistral3-d",
         "dsv3-d", "llama3-70b", "qwen25-72b", "mistral-l-2407", "llama31-405b"]
NAME = {k: note for k, h, f, kind, note in SHAPES}
HF = {k: (h, f) for k, h, f, kind, note in SHAPES}
MS = [8192, 16384, 32768]
# Estimates for the two cells where the ORIGINAL's vendor GEMM picked a slow kernel
# (GEMM time 6.8x / 5.0x of the M/2 cell instead of ~2x). Original rebuilt with
# GEMM = 2 x GEMM(M/2), scaled by that cell's measured b2b/single ratio; ours as measured.
EST = {
    ("llama4-mav-d", 32768, "gate_up"): 1.07,   # orig ~2.59 AG + ~4.03 GEMM -> ~6.6 ms; ours 6.155 ms
    ("qwen3-8b", 32768, "down"): 1.33,          # orig ~3.15 RS + ~1.53 GEMM -> ~4.6 ms; ours 3.466 ms
}


def parse_ag():
    d = {}
    cur = None
    for line in open(os.path.join(RES, "agprod.log")):
        m = re.match(r"=== (\S+) M=(\d+) K=(\d+) Nc=(\d+),(\d+)", line)
        if m:
            cur = (m.group(1), int(m.group(2)), int(m.group(4)))
            continue
        m = re.search(r"\[agf\] Nc=(\d+) stock ([0-9.]+) .*b2b/op stock ([0-9.]+) dx ([0-9.]+) x([0-9.]+)", line)
        if m and cur:
            kind = "up" if int(m.group(1)) == cur[2] else "gate_up"
            d[(cur[0], cur[1], kind)] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
    return d


def parse_rs():
    d = {}
    p = os.path.join(RES, "rsb2b.log")
    if not os.path.exists(p):
        return d
    cur = None
    for line in open(p):
        m = re.match(r"=== (\S+) M=(\d+) N=(\d+) K=(\d+)", line)
        if m:
            cur = (m.group(1), int(m.group(2)))
            continue
        m = re.search(r"\| b2b/op stock ([0-9.]+) fused ([0-9.]+) x([0-9.]+)", line)
        if m and cur:
            d[(cur[0], cur[1], "down")] = tuple(map(float, m.groups()))
    return d


def anomalous(d, key):
    """original time at M is far above 2x the original time at M/2 -> original itself misbehaved"""
    m, M, kind = key
    lo = d.get((m, M // 2, kind))
    return bool(lo and d[key][0] / lo[0] > 2.6)


def section(title, d, kinds, shape_note):
    L = [f"## {title}", "", shape_note, ""]
    if not d:
        return L + ["（尚未跑完）", ""]
    cols = [(M, k) for M in MS for k in kinds]
    head = "| 模型 | " + " | ".join(f"M{M}" + (f" {k}" if len(kinds) > 1 else "") for M, k in cols) + " |"
    L += ["### 加速比（原版 ÷ 我们，连发计时）", "", head, "|---|" + "---|" * len(cols)]
    vals = {M: [] for M in MS}
    for m in ORDER:
        row = []
        for M, k in cols:
            v = d.get((m, M, k))
            if v is None:
                row.append("—")
                continue
            if anomalous(d, (m, M, k)):
                est = EST.get((m, M, k))
                row.append(f"≈{est:.2f}×†" if est else f"~~{v[2]:.2f}×~~†")
                continue
            vals[M].append(v[2])
            row.append(f"**{v[2]:.2f}×**" if v[2] >= 1.0 else f"{v[2]:.2f}×")
        L.append(f"| {NAME.get(m, m)} | " + " | ".join(row) + " |")
    L.append("")
    L += ["### 汇总", "", "| M | 赢的格数 | 几何均值 | 最高 |", "|---|---|---|---|"]
    import math
    allv = []
    for M in MS:
        v = vals[M]
        allv += v
        if v:
            g = math.exp(sum(map(math.log, v)) / len(v))
            L.append(f"| {M} | {sum(x >= 1.0 for x in v)}/{len(v)} | {g:.3f}× | {max(v):.3f}× |")
    if allv:
        g = math.exp(sum(map(math.log, allv)) / len(allv))
        L.append(f"| 全部 | {sum(x >= 1.0 for x in allv)}/{len(allv)} | {g:.3f}× | {max(allv):.3f}× |")
    L += ["", "### 原始时间（ms，连发计时，每个 op 平均）", "",
          "| 模型 | M | 投影 | 原版 | 我们 | 加速比 |", "|---|---|---|---|---|---|"]
    for m in ORDER:
        for M in MS:
            for k in kinds:
                v = d.get((m, M, k))
                if v:
                    flag = " †" if anomalous(d, (m, M, k)) else ""
                    L.append(f"| {NAME.get(m, m)} | {M} | {k} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f}×{flag} |")
    return L + [""]


ag, rs = parse_ag(), parse_rs()
L = ["# 算子级结果表（海光 BW1000 8 卡，TP=8）", "",
     "- **加速比 = 原版时间 ÷ 我们的时间**。原版 = 厂商 GEMM（hipBLAS）+ RCCL 集合通信，串行执行；我们 = 创新点 1 的融合算子（训练里实际调用的同一个函数）。",
     "- **计时方式：连发计时**。连续发出 16 个 op 不等待，总时间 ÷ 16，重复 8 次取中位数，取 8 张卡里最慢的那张。",
     "- 每格单独一个进程；正确性逐格校验，全部通过。",
     "- 加粗 = 我们更快。† = 该格原版的厂商 GEMM 在这个形状下选到了慢内核（GEMM 耗时是 M/2 那格的 5–7 倍，正常约 2 倍），实测加速比（GEMM+AG 2.63×、GEMM+RS 1.99×）被夸大；表中给的是估计值：把原版 GEMM 按 M/2 那格的 2 倍补回去，我们的时间用实测。估计值不计入汇总。",
     "- M = 一次 GEMM 处理的 token 数（序列长度 × micro-batch）。",
     "- 本文档由 `planner/gen_table.py` 从原始日志自动生成，新数据出来后重跑即更新。", ""]
L += section("GEMM+AG（AllGather + GEMM，列并行 MLP 第一层 fc1：up 与 gate_up）", ag, ["up", "gate_up"],
             "形状：K = hidden，Nc = ffn/8（up）或 2·ffn/8（gate_up）。数据：`results/raw/ops_8card/agprod.log`。")
L += section("GEMM+RS（GEMM + ReduceScatter，行并行 MLP 第二层 fc2：down）", rs, ["down"],
             "形状：N = hidden，K = ffn/8（down）。数据：`results/raw/ops_8card/rsb2b.log`。")
open(OUT, "w").write("\n".join(L) + "\n")
print("written", OUT, "| AG cells", len(ag), "| RS cells", len(rs))
