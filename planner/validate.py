"""Leave-one-model-out validation of the cost model + planner (CPU only).

For each of the 11 models: fit on the other 10, predict every cell of the held-out model,
let the planner choose, then score the choice against the MEASURED times of that cell.
Reported per timing mode (single, b2b):
  MAPE        prediction error of the dx time model
  decision    chosen candidate == fastest measured candidate (among those measured)
  regret      chosen measured time / fastest measured time - 1   (mean, worst)
  slower      cells where the chosen candidate is measured slower than stock (must be 0)
  speedup     geomean of stock / chosen over cells, vs always-dx and vs oracle
Baseline selector: a single global M threshold for dx (the TokenWeave / MORI style static
rule), threshold picked as the best one on the training models.
Writes planner_validation.md.
"""
import collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # cost-model cells, GEMM probes, measured step points
sys.path.insert(0, HERE)            # python -I drops the script dir from sys.path
from costmodel import CostModel, measured  # noqa: E402
cells = [c for c in json.load(open(os.path.join(DATA, "cells.json"))) if not c.get("anomaly")]
models = sorted({c["model"] for c in cells})
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
lines = ["# 代价模型 + 规划器：留一模型验证（CPU，仅用已测的算子级数据）", ""]


def static_rule(train, mode):
    """best single M threshold for 'use dx when M >= T' on the training cells."""
    best = None
    for T in (4096, 8192, 16384, 32768, 10 ** 9):
        tot = []
        for c in train:
            m = measured(c, mode)
            pick = "dx" if (c["M"] >= T and "dx" in m) else "stock"
            tot.append(m["stock"] / m[pick])
        s = gm(tot)
        if best is None or s > best[1]:
            best = (T, s)
    return best[0]


for mode in ("single", "b2b"):
    use = [c for c in cells if ("dx_" + mode) in c]
    if len(use) < 10:
        lines.append(f"## {mode}: 数据不足（{len(use)} 格），跳过"); continue
    per = collections.defaultdict(list)
    ape, rows = [], []
    for hold in models:
        train = [c for c in use if c["model"] != hold]
        test = [c for c in use if c["model"] == hold]
        if not test:
            continue
        cm = CostModel(mode=mode).fit(train)
        T = static_rule(train, mode)
        for c in test:
            m = measured(c, mode)
            pick, pred = cm.choose(c["op"], c["G"], c["B"], allowed=set(m))
            oracle = min(m, key=m.get)
            if "dx" in pred:
                ape.append(abs(pred["dx"] - m["dx"]) / m["dx"])
            stat = "dx" if (c["M"] >= T and "dx" in m) else "stock"
            rows.append(dict(op=c["op"], model=c["model"], M=c["M"], pick=pick, oracle=oracle,
                             t_pick=m[pick], t_oracle=m[oracle], t_stock=m["stock"],
                             t_dx=m.get("dx", m["stock"]), t_static=m[stat]))
    n = len(rows)
    dec = sum(r["pick"] == r["oracle"] for r in rows)
    reg = [r["t_pick"] / r["t_oracle"] - 1 for r in rows]
    slower = [r for r in rows if r["t_pick"] > r["t_stock"] * 1.005]
    sp = lambda key: gm([r["t_stock"] / r[key] for r in rows])
    lines += [f"## 计时方式：{'单次' if mode == 'single' else '连发'}（{n} 个调用点，{len(models)} 个模型）", "",
              "| 指标 | 值 |", "|---|---|",
              f"| dx 时间预测误差 MAPE | {100 * np.mean(ape):.1f}%（最大 {100 * max(ape):.1f}%） |",
              f"| 选对最快方案 | {dec}/{n} |",
              f"| 相对最优的平均损失 | {100 * np.mean(reg):.2f}%（最差 {100 * max(reg):.2f}%） |",
              f"| 选中方案比原版慢的格子 | {len(slower)} |",
              f"| 几何均值加速：规划器 | {sp('t_pick'):.3f}× |",
              f"| 几何均值加速：总用 dx | {sp('t_dx'):.3f}× |",
              f"| 几何均值加速：静态 M 阈值 | {sp('t_static'):.3f}× |",
              f"| 几何均值加速：事后最优 | {sp('t_oracle'):.3f}× |", ""]
    wrong = [r for r in rows if r["pick"] != r["oracle"]]
    if wrong:
        lines += ["选错的格子：", ""] + [
            f"- {r['op']} {r['model']} M{r['M']}：选 {r['pick']}（{r['t_pick']:.3f} ms），最优 {r['oracle']}（{r['t_oracle']:.3f} ms），"
            f"损失 {100 * (r['t_pick'] / r['t_oracle'] - 1):.1f}%" for r in wrong] + [""]
open(os.path.join(DATA, "planner_validation.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
