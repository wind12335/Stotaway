"""Summarise one e2e run directory (driver.log written by e2e/megatron/run.sh or run_moe.sh).

  python3 tools/e2e_summary.py <run dir>

Per arm: kind, iterations, median step time (iteration 3 on). Per kind: mean of its arms, speedup
vs the original (base) and vs the null control (fnull) when those arms exist in the same run.
Speedup = reference step time / this step time (> 1 = faster). Only compare numbers of one run.
"""
import collections
import os
import re
import sys

d = sys.argv[1]
arms, plan = [], {}
for line in open(os.path.join(d, "driver.log"), errors="replace"):
    m = re.search(r"\] (\S+) predictor -> (\S+)", line)
    if m:
        plan[m.group(1)] = m.group(2)
    m = re.search(r"\] (\S+) kind=(\S+) rc=(\S+) iters=(\S*) med\(it3\+\)=(\S*)", line)
    if m:
        name, kind, rc, it, med = m.groups()
        arms.append((name, kind, rc, it, float(med) if med else None))
if not arms:
    sys.exit(f"no finished arms in {d}/driver.log")
print(f"# {os.path.basename(os.path.normpath(d))}\n")
print("| arm | kind | rc | iterations | median step (ms) |")
print("|---|---|---|---|---|")
for name, kind, rc, it, med in arms:
    tag = f"{kind} (predictor)" if name in plan else kind
    print(f"| {name} | {tag} | {rc} | {it} | {med if med is not None else 'FAILED'} |")
by = collections.OrderedDict()
for name, kind, rc, it, med in arms:
    if med is not None:
        by.setdefault(kind, []).append(med)
mean = {k: sum(v) / len(v) for k, v in by.items()}
print("\n| kind | arms | mean step (ms) | vs base | vs fnull |")
print("|---|---|---|---|---|")
for k, v in mean.items():
    b = f"{mean['base'] / v:.3f}x" if "base" in mean else "-"
    n = f"{mean['fnull'] / v:.3f}x" if "fnull" in mean else "-"
    print(f"| {k} | {len(by[k])} | {v:.1f} | {b} | {n} |")
if plan:
    print("\nPredictor decisions: " + ", ".join(f"{a} -> {k}" for a, k in plan.items())
          + " (reports: plan_<arm>.md)")
