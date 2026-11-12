"""Predictor stage 2, v3: step-level control-plane term refit on every measured E2E point (CPU only).

What changed since v2 (step_planner.py, F48):
  - 11-01 A: the fused AG host time is not a platform constant (70B/27B = 2.15x / 1.85x).
  - 11-01 B: 27B MBS4 loses +3.26% vs MBS8 +3.05%; v2's linear-slack formula predicted +11% / +5%.
  - 11-01 C-F + 2-card probes (F50): every engine copy into another process's memory pulls the host
    back toward the GPU's progress (block ~25% of the GPU lag per copy, independent of count, size,
    fork style and API). A fused AG call issues 7 such pushes BEFORE its GEMMs, so it nearly resets
    the host's lead over the GPU; the GPU then idles while the host re-issues the next ops.
So the term to model is "GPU idle per fused AG call", not "host time beyond a fixed slack".

Data (fbc or fcth = all four sites overlap method, relative to the same-batch fbag), %:
  27B MBS8 +3.05 (F17), 27B MBS4 +3.26 (11-01 B), 70B MBS8 -1.68 (mean of 4, F27/F39/F41),
  70B MBS4 and MBS2: 11-01 G (filled in below when measured).
Per layer cycle: measured step change = dD (operator-level model, 2 fused AG sites x 2 calls) + 4 x e,
  e = GPU idle per fused AG call (ms). We fit e from the points and compare candidate laws:
  L0  e = 0                                    (operator-level only)
  L1  e = k * D_cycle                          (idle proportional to the layer's GPU time; 1 param)
  L2  e = k * D_cycle * [D_cycle < D*]         (L1 with a cutoff; 2 params)
  L3  e = max(0, c - s * D_cycle)              (fixed host re-issue cost minus a share of GPU time; 2 params)
Each law is scored by leave-one-point-out: fit on the others, predict the left-out point, and check
whether the decision (fbc if predicted at least STEP_MARGIN faster, else fbag) matches the measured best.
"""
import itertools, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # cost-model cells, GEMM probes, measured step points
sys.path.insert(0, HERE)
from costmodel import CostModel  # noqa: E402

STEP_MARGIN = 1.0
SITES = ("qkv", "fc1_gate_up")
# (model, MBS): original step, fbag step, layer cycles, probe file, measured fbc-vs-fbag %
A800_POINTS = os.path.join(DATA, 'a800_step_points.json')
if not os.path.exists(A800_POINTS):
    raise SystemExit('A800 完整 fbag/fbc 整步数据尚未采集，不能使用 BW1000 的第二级参数')
_a800 = json.load(open(A800_POINTS))
if _a800.get('platform') != 'A800-TP8' or _a800.get('classification') != 'measured-full-fbag-fbc':
    raise SystemExit('第二级输入不是 A800 完整 fbag/fbc 实测数据')
POINTS = {}
for key, value in _a800['points'].items():
    model, batch = key.split('_mbs')
    POINTS[(model, int(batch))] = value
if len(POINTS) < 5:
    raise SystemExit('A800 第二级需要原表的 5 个整步点')

cm = CostModel(mode="b2b").fit(json.load(open(os.path.join(DATA, "cells.json"))))
_pr = {}


def probe(key, site):
    f = os.path.join(DATA, POINTS[key]["probe"])
    if f not in _pr:
        _pr[f] = {(p["model"], p["site"]): p for p in json.load(open(f))} if os.path.exists(f) else None
    if _pr[f] is None:
        return None
    return _pr[f][(key[0], site)]


def dD(key):
    tot = 0.0
    for s in SITES:
        p = probe(key, s)
        if p is None:
            return None
        t = cm.predict(p["op"], p["G_b2b"], p["B"])
        tot += 2 * (t["dx"] - t["stock"])
    return tot


def d_cycle(key):
    return POINTS[key]["fbag"] / POINTS[key]["cycles"]


def to_pct(key, ms_per_cycle):
    return 100 * ms_per_cycle * POINTS[key]["cycles"] / POINTS[key]["fbag"]


def e_meas(key):
    """measured GPU idle per fused AG call (ms)"""
    ms = np.mean(POINTS[key]["meas"]) / 100 * POINTS[key]["fbag"] / POINTS[key]["cycles"]
    return (ms - dD(key)) / 4


LAWS = {
    "L0": dict(n=0, f=lambda D, th: 0.0),
    "L1": dict(n=1, f=lambda D, th: th[0] * D),
    "L2": dict(n=2, f=lambda D, th: th[0] * D if D < th[1] else 0.0),
    "L3": dict(n=2, f=lambda D, th: max(0.0, th[0] - th[1] * D)),
}


def fit(law, keys):
    xs = np.array([d_cycle(k) for k in keys]); ys = np.array([e_meas(k) for k in keys])
    L = LAWS[law]
    if L["n"] == 0:
        return ()
    if law == "L1":
        return (float(np.sum(xs * ys) / np.sum(xs * xs)),)
    best = None
    if law == "L2":
        for cut in np.linspace(80, 400, 321):
            m = xs < cut
            k = float(np.sum(xs[m] * ys[m]) / np.sum(xs[m] ** 2)) if m.any() else 0.0
            r = float(np.sum((np.where(m, k * xs, 0.0) - ys) ** 2))
            if best is None or r < best[0] - 1e-12:
                best = (r, (k, cut))
        return best[1]
    if law == "L3":
        for c in np.linspace(0, 6, 241):
            for s in np.linspace(0, 0.06, 241):
                r = float(np.sum((np.maximum(0.0, c - s * xs) - ys) ** 2))
                if best is None or r < best[0]:
                    best = (r, (c, s))
        return best[1]


def predict_pct(law, th, key):
    return to_pct(key, dD(key) + 4 * LAWS[law]["f"](d_cycle(key), th))


def choose(pct):
    return "fbc" if -pct >= STEP_MARGIN else "fbag"


def best_meas(key):
    m = float(np.mean(POINTS[key]["meas"]))
    return "fbc" if m < 0 else "fbag"


if __name__ == "__main__":
    have = [k for k, v in POINTS.items() if v["meas"] is not None and dD(k) is not None]
    out = ["# 预测器第二级 v3：每次融合 AG 调用的 GPU 空等（CPU，只用实测点）", "",
           "| 点 | 每层周期 GPU 时间 (ms) | 算子级 ΔD (ms/周期) | 实测 fbc 相对 fbag | 反推每次调用 GPU 空等 e (ms) |",
           "|---|---|---|---|---|"]
    for k in have:
        out.append(f"| {k[0].upper()} MBS{k[1]} | {d_cycle(k):.1f} | {dD(k):+.2f} | "
                   f"{np.mean(POINTS[k]['meas']):+.2f}% | {e_meas(k):.2f} |")
    out += ["", f"留一点检验（拿掉一个点拟合，预测这个点；选择规则：预测快 ≥{STEP_MARGIN}% 选 fbc，否则 fbag）：", "",
            "| 定律 | 全体拟合参数 | " + " | ".join(f"{k[0].upper()} MBS{k[1]}" for k in have) + " | 选择正确 | 平均绝对误差 |",
            "|---|---|" + "---|" * len(have) + "---|---|"]
    for law in LAWS:
        th_all = fit(law, have)
        cells, ok, err = [], 0, []
        for k in have:
            rest = [r for r in have if r != k]
            th = fit(law, rest) if len(rest) >= LAWS[law]["n"] else th_all
            p = predict_pct(law, th, k)
            m = float(np.mean(POINTS[k]["meas"]))
            hit = choose(p) == best_meas(k)
            ok += hit; err.append(abs(p - m))
            cells.append(f"{p:+.2f}{'' if hit else ' ✗'}")
        out.append(f"| {law} | {', '.join(f'{x:.4g}' for x in th_all) or '—'} | " + " | ".join(cells)
                   + f" | {ok}/{len(have)} | {np.mean(err):.2f} |")
    # final rule = L2 with the cutoff at the midpoint of the bracket (between the largest D with e > 0
    # and the smallest D with e ~ 0); plan-time input D_cycle comes from one short run (<= 3 iterations)
    # of the original configuration, read as step / (layers x micro-batches).
    pos = [k for k in have if e_meas(k) > 0.3]; zer = [k for k in have if e_meas(k) <= 0.3]
    xs = np.array([d_cycle(k) for k in pos]); ys = np.array([e_meas(k) for k in pos])
    k_ = float(np.sum(xs * ys) / np.sum(xs * xs))
    lo, hi = max(xs), min([d_cycle(k) for k in zer], default=float("inf"))
    cut = (lo + hi) / 2 if hi < float("inf") else float("inf")
    out += ["", f"最终规则：e = {k_:.4f} × 每层周期 GPU 时间（< {cut:.0f} ms 时），否则 0；截止点在 {lo:.0f}–{hi:.0f} ms 之间取中点。",
            "规划时输入：单卡 GEMM 探测（第一级）+ 原版配置短跑 ≤3 次迭代读出每层周期 GPU 时间（不建对称堆上下文）。", "",
            "| 点 | 预测 fbc 相对 fbag | 预测器选择 | 实测 | 实测最优 | 对否 |", "|---|---|---|---|---|---|"]
    ok = 0
    for k in have:
        pct = to_pct(k, dD(k) + 4 * (k_ * d_cycle(k) if d_cycle(k) < cut else 0.0))
        m = float(np.mean(POINTS[k]["meas"])); c = choose(pct); b = best_meas(k); ok += c == b
        out.append(f"| {k[0].upper()} MBS{k[1]} | {pct:+.2f}% | {c} | {m:+.2f}% | {b} | {'✓' if c == b else '✗'} |")
    out.append(f"\n全体拟合 {ok}/{len(have)} 选对；留一检验见上表 L2 行。目标：留一选对率至少 80%。")
    pend = [k for k, v in POINTS.items() if v["meas"] is None]
    if pend:
        out += ["", "待测点（用全部已测点拟合的预测，跑前登记用）：", ""]
        for law in LAWS:
            th = fit(law, have)
            out.append(f"- {law}: " + "; ".join(
                f"{k[0].upper()} MBS{k[1]} {predict_pct(law, th, k):+.2f}%" if dD(k) is not None
                else f"{k[0].upper()} MBS{k[1]} (缺 GEMM 探测)" for k in pend))
    open(os.path.join(DATA, "step_planner_v3.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
