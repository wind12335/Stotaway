"""Collect operator-level measurements into one cell table for the cost model.

Sources (all already measured, nothing here touches a GPU):
  GEMM+RS  results/raw/ops_8card/{prodop,prodop2,prodop3,rsb2b}.log   (fused op Megatron calls)
  GEMM+AG  results/raw/ops_8card/agprod.log                            (fused op Megatron calls)
  split    results/raw/ops_8card/split_e1.csv                                  (batch-split + RCCL side stream)

One cell = (op, model, M, N, K). op: rs | ag_up | ag_gu.
Per timing mode t in {single, b2b} we keep, when measured:
  stock_t   vendor GEMM + RCCL collective, serial
  dx_t      our fused op (permutation direct push)
Single mode also has the components G (vendor GEMM alone), C (RCCL collective alone),
and for RS our pure RS time Cdx. Split cells keep speedup ratios vs their own stock
(different harness: only the ratio is comparable across harnesses).
Output: cells.json
"""
import collections, csv, json, os, re, statistics as st, sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "ops_8card")
D = RES
R = 8
acc = collections.defaultdict(lambda: collections.defaultdict(list))


def add(key, **kv):
    for k, v in kv.items():
        acc[key][k].append(v)


# ---- GEMM+RS ----------------------------------------------------------------
RS_RE = re.compile(r"stock ([0-9.]+) \(gemm ([0-9.]+) \+ rccl-rs ([0-9.]+)\) \| dx-rs ([0-9.]+) \| "
                   r"gemm\+dx-rs ([0-9.]+) \| dx-fused ([0-9.]+)")
RS_B2B = re.compile(r"\| b2b/op stock ([0-9.]+) fused ([0-9.]+)")
for fn in ("prodop.log", "prodop2.log", "prodop3.log", "rsb2b.log"):
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        continue
    cur = None
    for line in open(p):
        m = re.match(r"=== (\S+) M=(\d+) N=(\d+) K=(\d+)", line)
        if m:
            cur = ("rs", m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            continue
        m = RS_RE.search(line)
        if m and cur:
            s, g, c, cdx, gdx, f = map(float, m.groups())
            add(cur, stock_single=s, G=g, C=c, Cdx=cdx, dx_single=f)
            mb = RS_B2B.search(line)
            if mb:
                add(cur, stock_b2b=float(mb.group(1)), dx_b2b=float(mb.group(2)))

# ---- GEMM+AG ----------------------------------------------------------------
AG_RE = re.compile(r"\[agf\] Nc=(\d+) stock ([0-9.]+) \(ag ([0-9.]+) \+ gemm ([0-9.]+)\) \| "
                   r"dx-ag-gemm ([0-9.]+) .*b2b/op stock ([0-9.]+) dx ([0-9.]+)")
p = os.path.join(D, "agprod.log")
cur = None
for line in open(p) if os.path.exists(p) else []:
    m = re.match(r"=== (\S+) M=(\d+) K=(\d+) Nc=(\d+),(\d+)", line)
    if m:
        cur = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        continue
    m = AG_RE.search(line)
    if m and cur:
        nc = int(m.group(1))
        op = "ag_up" if nc == cur[3] else "ag_gu"
        s, c, g, f, sb, fb = map(float, m.groups()[1:])
        add((op, cur[0], cur[1], nc, cur[2]), stock_single=s, C=c, G=g, dx_single=f,
            stock_b2b=sb, dx_b2b=fb)

# ---- split (batch split + RCCL side stream): keep best ovl arm ratio per nb ------
split = collections.defaultdict(lambda: collections.defaultdict(list))
p = os.path.join(RES, "split_e1.csv")
for r in csv.DictReader(open(p)) if os.path.exists(p) else []:
    if r["arm"].startswith("ovl"):
        op = "rs" if r["op"] == "rs" else "ag_up"
        k = (op, r["key"], int(r["M"]))
        split[k]["split%s_single" % r["arm"][3:]].append(float(r["speedup_vs_stock"]))
        split[k]["split%s_b2b" % r["arm"][3:]].append(float(r["b2b_speedup"]))

cells = []
for (op, model, M, N, K), d in sorted(acc.items()):
    c = dict(op=op, model=model, M=M, N=N, K=K,
             B=(M * N * 2 if op == "rs" else M * K * 2) / 1e6,   # collective payload, MB
             n=len(d["stock_single"]))
    c.update({k: round(st.median(v), 4) for k, v in d.items()})
    sk = (op, model, M)              # split harness measured rs and ag up only; gate_up gets none
    for k, v in (split.get(sk, {}) if op in ("rs", "ag_up") else {}).items():
        c["r_" + k] = round(st.median(v), 4)                     # speedup ratio vs split-harness stock
    # split-harness sanity: the same arm's single and b2b ratios must agree within 1.5x;
    # otherwise that harness's own stock misbehaved (e.g. gemma3-27b rs M8192 b2b 2.58 vs single 1.04)
    for nb in (2, 4):
        a_, b_ = c.get(f"r_split{nb}_single"), c.get(f"r_split{nb}_b2b")
        if a_ and b_ and not (1 / 1.5 <= b_ / a_ <= 1.5):
            c.pop(f"r_split{nb}_single"); c.pop(f"r_split{nb}_b2b")
            c.setdefault("split_dropped", []).append(nb)
    cells.append(c)

# flag cells whose stock time is anomalous (stock(2M)/stock(M) far from 2; F42/F45)
by = {(c["op"], c["model"], c["M"]): c for c in cells}
for c in cells:
    lo = by.get((c["op"], c["model"], c["M"] // 2))
    c["anomaly"] = bool(lo and c["stock_single"] / lo["stock_single"] > 2.6)
json.dump(cells, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cells.json"), "w"), indent=1)
cnt = collections.Counter(c["op"] for c in cells)
print(f"cells: {dict(cnt)} | with b2b: {sum('dx_b2b' in c for c in cells)} | "
      f"with split: {sum(any(k.startswith('r_') for k in c) for c in cells)} | "
      f"anomaly: {[(c['op'], c['model'], c['M']) for c in cells if c['anomaly']]} | "
      f"split dropped: {[(c['op'], c['model'], c['M'], c['split_dropped']) for c in cells if c.get('split_dropped')]}")
