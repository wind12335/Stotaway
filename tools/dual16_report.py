"""16 卡结果汇总：每臂两遍的 iter3+ 中位（取 node0 的 r7），对 BASE / NULL / 预测器对照臂的比值。
usage: python3 tools/dual16_report.py <RUN_DIR>   (reports every shape subdir: 70b/, 27b/)"""
import glob, os, re, statistics, sys


def report(D):
    med = {}
    # iteration lines are printed by the LAST global rank (rank 15 = node1/out_r7); take whichever node has them
    for a in sorted(glob.glob(os.path.join(D, "*", "node*", "out_r7.log"))):
        arm = a.split(os.sep)[-3]
        if arm in med:
            continue
        v = [float(m.group(1)) for l in open(a, errors="ignore") for m in [re.search(r"per iteration \(ms\): ([\d.]+)", l)] if m]
        if len(v) >= 4:
            med[arm] = statistics.median(v[2:])
    kinds = {}
    for arm, t in med.items():
        kinds.setdefault(arm.split(".")[0], []).append(t)
    plan = {}
    for n in (0, 1):
        p = os.path.join(D, f"plan_node{n}.md")
        if os.path.exists(p):
            m = re.search(r"可部署配置：([a-z]+)", open(p).read())
            plan[n] = m.group(1) if m else "?"
    print(f"# 16 卡结果 {os.path.basename(D)}\n\n预测器选择：node0 {plan.get(0)}，node1 {plan.get(1)}\n")
    print("| 臂 | 每遍 iter3+ 中位 (ms) | 均值 | 相对 BASE | 相对 NULL |\n|---|---|---|---|---|")
    mean = {k: sum(v) / len(v) for k, v in kinds.items()}
    for k, v in kinds.items():
        rb = f"{mean['BASE'] / mean[k]:.3f}×" if "BASE" in mean else "—"
        rn = f"{mean['NULL'] / mean[k]:.3f}×" if "NULL" in mean else "—"
        print(f"| {k} | {' / '.join(f'{x:.1f}' for x in v)} | {mean[k]:.1f} | {rb} | {rn} |")
    if "PLAN" in mean:
        others = [k for k in mean if k not in ("BASE", "NULL", "PLAN")]
        for k in others:
            print(f"\nPLAN（预测器选的配置）相对 {k}（ALT = 预测器没选的那个）：{100 * (mean['PLAN'] / mean[k] - 1):+.2f}%（负 = 预测器选对）")


top = sys.argv[1]
subs = [d for d in sorted(glob.glob(os.path.join(top, "*"))) if os.path.isdir(d) and glob.glob(os.path.join(d, "*", "node1"))]
for d in subs or [top]:
    report(d)
    print()
v = sorted(glob.glob(os.path.join(top, "rdma_verdict_node*.txt")))
for p in v:
    print(open(p).read().strip())
