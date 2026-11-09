"""预测器（创新点 2）统一规划脚本：输入模型形状和 MBS，输出 4 个调用点各用哪种方法。CPU 为主，可选单卡测 GEMM。

  python3 planner.py --hidden 8192 --ffn 28672 --heads 64 --kv 8 --layers 80 --mbs 4 --step-ms 9655
  python3 planner.py --model 70b --mbs 4                 # 用内置形状 + 已测的 GEMM 探测文件
  python3 planner.py --selftest                          # 在全部实测点上复现选择并打分

调用点（TP=R，SP 开）：
  QKV     注意力输入投影，先 AG 再 GEMM   op = ag_up   [M, H] x [H, (heads + 2 kv) * hd / R]
  o_proj  注意力输出投影，先 GEMM 再 RS   op = rs      [M, H / R] x [H / R, H]
  fc1     MLP 第一层（gate_up），先 AG    op = ag_gu   [M, H] x [H, 2 F / R]
  fc2     MLP 第二层（down），先 GEMM 再 RS op = rs    [M, F / R] x [F / R, H]
  M = seq * MBS（序列并行下 GEMM 的总行数），bf16。

第一级（costmodel.py，算子级代价模型，F47）：每个调用点按形状和单卡 GEMM 时间 G 预测
  原版 / 重叠方法 / 拆批 2 / 拆批 4 的时间，带安全门（悲观估计必须比原版快 2%）。
第二级（整步控制面项，F49–F51，step_planner_v3.py）：
  AG 两处：融合 AG 每次调用让 GPU 空等 e = K_E * D（D < D_CUT 时），否则 0；
           D = 每层周期 GPU 时间 = 整步 / (层数 x micro-batch 数)。每个开重叠方法的 AG 调用点
           在每个层周期里被调用 2 次（前向 + 重算），整步变化 = 第一级 dD + 2 * e * (开的 AG 点数)。
           只有预测整步至少快 STEP_MARGIN 才开，否则用原版。
  RS 两处：作为一组（= fbag 的 RS 部分：前向融合 GEMM+RS、反向融合 AG+dgrad、独立 RS）开关。
           实测（相对原版）27B MBS8 -6.37%、70B MBS8 -7.44%、70B MBS4 -5.80%、70B MBS2 +3.1%（F30）
           => D < D_RS_CUT 时 RS 组回退原版。只有 1 个输点，截止点只夹在 80–143 ms 之间（局限）。
拆批：第一级会给出拆批选择，但拆批还没接进训练（E2 待办），所以"可部署选择"里拆批暂按原版处理。
"""
import argparse, json, math, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")          # cost-model cells, GEMM probes, measured step points
sys.path.insert(0, HERE)
from costmodel import CostModel  # noqa: E402

R = 8
SEQ = 4096
MICRO = 8                 # GBS = 8 x MBS, DP = 1 -> 8 micro-batches per step
STEP_MARGIN = 1.0         # % ; AG sites: overlap only if the step is predicted this much faster
K_E = 0.0096              # ms GPU idle per fused AG call per ms of layer-cycle GPU time (v3 fit, 5 points)
D_CUT = 205.0             # ms ; bracket 137-273 (largest D with idle > 0, smallest with idle ~ 0)
D_RS_CUT = 111.0          # ms ; bracket 80-143 (70B MBS2 loses, 27B MBS8 / 70B MBS4 win)
FBAG_OVER_BASE = 0.94     # step ratio fbag / original on points where both are measured (0.926-0.955)
BUILTIN = {               # hidden, ffn, heads, kv groups, layers in the 8-card vehicle
    "27b": (5120, 17408, 40, 8, 32),
    "70b": (8192, 28672, 64, 8, 8),
}
NAME = {"stock": "原版", "dx": "重叠方法", "split2": "拆批 2", "split4": "拆批 4"}


def sites(h, f, nh, kv, M):
    hd = h // nh
    return [("QKV", "qkv", "ag_up", M, h, (nh + 2 * kv) * hd // R),
            ("o_proj", "o_proj", "rs", M, h // R, h),
            ("fc1", "fc1_gate_up", "ag_gu", M, h, 2 * f // R),
            ("fc2", "fc2", "rs", M, f // R, h)]


def payload_mb(op, M, K, N):
    return (M * N * 2 if op == "rs" else M * K * 2) / 1e6


def gemm_times(model, h, f, nh, kv, M):
    """G (back-to-back ms) per site: probe file if it has this model and M, else time on one GPU."""
    fn = {32768: "gemm_probe.json", 16384: "gemm_probe_m16384.json", 8192: "gemm_probe_m8192.json"}.get(M)
    if model and fn and os.path.exists(os.path.join(DATA, fn)):
        rows = {p["site"]: p["G_b2b"] for p in json.load(open(os.path.join(DATA, fn))) if p["model"] == model}
        if len(rows) == 4:
            return rows, f"探测文件 {fn}"
    import torch
    out = {}
    for _, key, _, m, k, n in sites(h, f, nh, kv, M):
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
        w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda") * 0.05
        c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            torch.matmul(a, w, out=c)
        torch.cuda.synchronize()
        bb = []
        for _ in range(8):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(16):
                torch.matmul(a, w, out=c)
            e.record(); torch.cuda.synchronize()
            bb.append(s.elapsed_time(e) / 16)
        out[key] = statistics.median(bb)
        del a, w, c
        torch.cuda.empty_cache()
    return out, "单卡现测（16 连发 × 8 次中位）"


def plan(h, f, nh, kv, layers, mbs, step_ms=None, model=None, d_cycle=None, cm=None, quiet=False,
         stage2=True, M=None):
    """stage2=False: operator-level choice only (inference forward: the stage-2 term is calibrated on
    training layer cycles = forward + recompute + backward and does not transfer). M overrides
    seq x MBS (vLLM prefill chunks)."""
    cm = cm or CostModel(mode="b2b").fit(json.load(open(os.path.join(DATA, "cells.json"))))
    M = M or SEQ * mbs
    G, gsrc = gemm_times(model, h, f, nh, kv, M)
    cycles = layers * MICRO
    if d_cycle is not None:
        D, dsrc = d_cycle, "给定（fbag 口径）"
    elif step_ms is not None:
        D, dsrc = step_ms * FBAG_OVER_BASE / cycles, f"原版短跑整步 {step_ms:.0f} ms × {FBAG_OVER_BASE} / {cycles} 个层周期"
    else:
        D = 15.4 * sum(G.values())
        dsrc = "未给整步时间，按 15.4 × 四处 GEMM 时间之和粗估（实测比值 12.2–20.0，结论可能不准）"
    rows = []
    for nm, key, op, m, k, n in sites(h, f, nh, kv, M):
        B = payload_mb(op, m, k, n)
        best, p = cm.choose(op, G[key], B)                       # operator level, all candidates
        dep, _ = cm.choose(op, G[key], B, allowed={"dx"})       # deployable today (split not in training)
        rows.append(dict(name=nm, key=key, op=op, M=m, K=k, N=n, G=G[key], B=B, stage1=best, stage1d=dep, pred=p))
    # ---- stage 2, RS group
    rs_on = (D >= D_RS_CUT or not stage2) and all(r["stage1d"] == "dx" for r in rows if r["op"] == "rs")
    # ---- stage 2, AG sites: best subset by predicted step change (relative to "RS group as decided, AG original")
    # AG sites are decided at step level: a site is a candidate when the operator-level model predicts
    # the overlap method faster than the original at all (raw prediction, no per-op guard); the step-level
    # margin STEP_MARGIN plays the guard's role here (this is the rule validated in step_planner_v3).
    ag = [r for r in rows if r["op"] != "rs" and r["pred"]["dx"] < r["pred"]["stock"]]
    if not stage2:
        ag = [r for r in rows if r["op"] != "rs" and r["stage1d"] == "dx"]
    e = (K_E * D if D < D_CUT else 0.0) if stage2 else 0.0
    best_sub, best_pct, table = (), 0.0, []
    for n_on in range(len(ag) + 1):
        from itertools import combinations
        for sub in combinations(ag, n_on):
            if not sub:
                continue
            dD = sum(2 * (r["pred"]["dx"] - r["pred"]["stock"]) for r in sub)
            pct = 100 * (dD + 2 * e * len(sub)) / D
            table.append((sub, dD, pct))
            if pct < best_pct and (-pct >= STEP_MARGIN or not stage2):
                best_sub, best_pct = sub, pct
    on = {r["key"] for r in best_sub}
    for r in rows:
        if r["op"] == "rs" and not stage2:     # inference: each RS site on its own operator-level choice
            r["final"] = r["stage1d"]
            r["why"] = "第一级（算子级，推理不加第二级）"
        elif r["op"] == "rs":
            r["final"] = "dx" if rs_on else "stock"
            r["why"] = ("RS 组：第一级选重叠方法且每层周期 GPU 时间 ≥ %.0f ms" % D_RS_CUT if rs_on else
                        ("每层周期 GPU 时间 %.0f ms < %.0f ms，RS 组整步实测会输（70B MBS2 +3.1%%）" % (D, D_RS_CUT)
                         if D < D_RS_CUT else "第一级选了原版或拆批，RS 组按原版"))
        else:
            if r["pred"]["dx"] >= r["pred"]["stock"]:
                r["final"] = "stock"
                r["why"] = "第一级（算子级）预测重叠方法不比原版快"
            else:
                r["final"] = "dx" if r["key"] in on else "stock"
                if r["key"] in on:
                    r["why"] = "第二级：所选组合整步预测快 %.2f%%（≥%.0f%%）" % (-best_pct, STEP_MARGIN)
                elif e > 0:
                    r["why"] = "第二级：每次融合 AG 调用让 GPU 空等 %.2f ms，抵掉了算子级收益" % e
                else:
                    r["why"] = "第二级：没有空等，但含它的组合整步收益都不足 %.0f%%" % STEP_MARGIN
        if r["final"] == "stock" and r["stage1"].startswith("split"):
            r["note"] = "算子级最优是%s（未接进训练）" % NAME[r["stage1"]]
        r["deploy"] = r["final"]
    ag_on = {r["key"] for r in rows if r["op"] != "rs" and r["deploy"] == "dx"}
    rs_dep = {r["key"] for r in rows if r["op"] == "rs" and r["deploy"] == "dx"}
    if not stage2:
        cfg = "按调用点：" + "，".join(f"{r['name']}={NAME[r['deploy']]}" for r in rows)
    elif not rs_on:
        cfg = "base（四处原版）" if not ag_on else "（RS 原版 + AG 重叠：未定义的配置）"
    else:
        cfg = {frozenset(): "fbag（RS 两处用重叠方法，AG 两处用原版）",
               frozenset({"qkv", "fc1_gate_up"}): "fbc（四处全用重叠方法）",
               frozenset({"qkv"}): "fcq（fbag + 只 QKV 用重叠方法）",
               frozenset({"fc1_gate_up"}): "fcl（fbag + 只 fc1 用重叠方法）"}[frozenset(ag_on)]
    res = dict(M=M, D=D, dsrc=dsrc, gsrc=gsrc, e=e, rows=rows, rs_on=rs_on, ag_table=table,
               ag_best_pct=best_pct, config=cfg)
    if not quiet:
        report(res, h, f, nh, kv, layers, mbs)
    return res


def report(res, h, f, nh, kv, layers, mbs):
    print(f"# 预测器规划：hidden {h} ffn {f} heads {nh} kv {kv} 层数 {layers} MBS {mbs}（M = {res['M']}，TP = {R}）")
    print(f"- GEMM 时间来源：{res['gsrc']}")
    print(f"- 每层周期 GPU 时间 D = {res['D']:.1f} ms（{res['dsrc']}）")
    print(f"- 每次融合 AG 调用的 GPU 空等 e = {res['e']:.2f} ms（D {'<' if res['D'] < D_CUT else '≥'} 截止点 {D_CUT:.0f} ms）\n")
    print("| 调用点 | 形状 M×K×N | G (ms) | 第一级预测 原版 / 重叠 / 拆批2 / 拆批4 (ms) | 第一级选择（含拆批） | 最终选择（可部署） | 理由 |")
    print("|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        p = r["pred"]
        ps = " / ".join(f"{p[k]:.2f}" if k in p else "—" for k in ("stock", "dx", "split2", "split4"))
        print(f"| {r['name']} | {r['M']}×{r['K']}×{r['N']} | {r['G']:.2f} | {ps} | {NAME[r['stage1']]} | "
              f"{NAME[r['final']]} | {r['why']}{'；' + r['note'] if r.get('note') else ''} |")
    if res["ag_table"]:
        print("\nAG 调用点组合的整步预测（相对'AG 两处用原版'）：")
        for sub, dD, pct in res["ag_table"]:
            print(f"- 开 {' + '.join(r['name'] for r in sub)}：算子级 {dD:+.2f} ms/层周期，加 GPU 空等后 {pct:+.2f}%")
    print(f"\n**可部署配置：{res['config']}**")
    if any(r["stage1"].startswith("split") for r in res["rows"]):
        print("（第一级有调用点的算子级最优是拆批；拆批未接进训练，最终选择只在重叠方法 / 原版之间做）")


# ---------------------------------------------------------------- self test on every measured E2E point
MEASURED = [  # model, MBS, fbag step (ms), measured best configuration, evidence
    ("27b", 8, 34989.4, "fbag", "fbag 比原版 −6.37%（F26/F35）；fbc 比 fbag +3.05%（F17）"),
    ("27b", 4, 19051.0, "fbag*", "fbc 比 fbag +3.26%（F49）；原版未测"),
    ("70b", 8, 17486.3, "fbc", "fbag 比原版 −7.44%（F27）；fbc 比 fbag −1.68%（4 次均值）"),
    ("70b", 4, 8583.1, "fbag", "fbag 比原版 −5.80%（F30）；fbc 比 fbag +2.21%（F51）"),
    ("70b", 2, 4921.9, "base", "fbag 比原版 +3.1%（F30）；fbc 比 fbag +4.77%（F51）"),
]


def selftest():
    cm = CostModel(mode="b2b").fit(json.load(open(os.path.join(DATA, "cells.json"))))
    print("# 预测器自检：在全部实测点上复现选择（D 用实测 fbag 整步 / 层周期数）\n")
    print("| 点 | D (ms) | 预测器配置 | 实测最优 | 依据 | 对否 |")
    print("|---|---|---|---|---|---|")
    ok = n = 0
    for model, mbs, fbag, best, ev in MEASURED:
        h, f, nh, kv, L = BUILTIN[model]
        D = fbag / (L * MICRO)
        r = plan(h, f, nh, kv, L, mbs, model=model, d_cycle=D, cm=cm, quiet=True)
        got = r["config"].split("（")[0]
        want = best.rstrip("*")
        hit = got == want
        if best.endswith("*"):          # only the AG decision is measured at this point
            hit = (got in ("fbag", "base")) == (want in ("fbag", "base"))
        ok += hit; n += 1
        print(f"| {model.upper()} MBS{mbs} | {D:.0f} | {r['config']} | {best} | {ev} | {'✓' if hit else '✗'} |")
    print(f"\n{ok}/{n} 选对（注：这些点也是第二级的拟合数据，属于样本内；第二级 AG 部分的留一检验为 4/5，见 step_planner_v3.md）。")
    print("* 27B MBS4 只测了 AG 两处的选择（fbc vs fbag），RS 组的选择（fbag vs 原版）未测，按 AG 决策计分。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(BUILTIN))
    ap.add_argument("--hidden", type=int); ap.add_argument("--ffn", type=int)
    ap.add_argument("--heads", type=int); ap.add_argument("--kv", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--step-ms", type=float, help="原版配置短跑（≤3 次迭代）读出的整步时间")
    ap.add_argument("--dcycle", type=float, help="直接给每层周期 GPU 时间（fbag 口径）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    if a.model:
        h, f, nh, kv, L = BUILTIN[a.model]
        L = a.layers or L
    else:
        h, f, nh, kv, L = a.hidden, a.ffn, a.heads, a.kv, a.layers
        assert None not in (h, f, nh, kv, L), "give --model or all of --hidden --ffn --heads --kv --layers"
    plan(h, f, nh, kv, L, a.mbs, step_ms=a.step_ms, model=a.model, d_cycle=a.dcycle)
