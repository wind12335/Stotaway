"""Tables and summaries of the BW1000 OPRS experiments (handoff doc section 7). CPU only.

  python3 tools/oprs_tables.py training  <run dir>              Megatron A/B of e2e/megatron/run_oprs.sh:
                                                             <run dir>/e2e_training.csv, summary on stdout
  python3 tools/oprs_tables.py inference <run dir> <M>       vLLM A/B of e2e/vllm/run_oprs.sh:
                                                             <run dir>/e2e_inference.csv, summary on stdout
  python3 tools/oprs_tables.py operator  <results root> --tp 8 [--data files...]
                                                             <results root>/tables/operator_samples.csv
  python3 tools/oprs_tables.py choices   <results root> --tp 8 --topology BW8 [--legacy F | --data F...]
                                                             tables/predictor_choices.csv, predictor_summary.csv,
                                                             predictor_family_holdout.json
  python3 tools/oprs_tables.py sites     <results root> --plans <plan-set dir | plan.json ...> [--data F...]
                                                             tables/target_sites.csv

Laws kept by every table: one number always travels with its data class (legacy table vs new b2b16
protocol; operator-level vs end-to-end); a point that could not run is a row with its status, never a
zero; tables under tables/ are never overwritten (they are rebuilt from ALL raw batches: move the old
one aside first). End-to-end ratios pair runs of the same driver batch by their run index.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

ITER = re.compile(r'elapsed time per iteration \(ms\): ([0-9.]+).*?lm loss: ([0-9.Ee+-]+).*?grad norm: (\S+).*?'
                  r'number of skipped iterations:\s+(\d+) \| number of nan iterations:\s+(\d+)')
VAL = re.compile(r'validation loss at iteration \d+ on (?:validation|test) set \| lm loss value: ([0-9.Ee+-]+)')
MAX_ALLOC = re.compile(r'max allocated: ([0-9.]+)')
READY = re.compile(r'OPRS_READY rank=(\d+) tp=(\d+) workload=(\S*) plan=(\w*) backend=(\S*) entries=(\d+) '
                   r'deo_bwd=(\S*) contexts=(\d+)')
HIT = re.compile(r'OPRS_HIT rank=(\d+) site=(\w+) op=(\w+) M=(\d+) N=(\d+) K=(\d+) choice=(\w+) count=(\d+)')
FALLBACK = re.compile(r'OPRS_FALLBACK rank=(\d+) reason=(\w+) count=(\d+)')
START = re.compile(r'\[e2e\] (\S+) kind=(\S+) plan=(\S+) model=(\S+) tp=(\d+) MBS=(\d+) NL=(\d+|\S*) start (\S+)')
END = re.compile(r'\[e2e\] (\S+) kind=(\S+) rc=(\d+)(?: py_rc=(\S+))? iters=(\d+) med\(it3\+\)=(\S*) '
                 r'hit_lines_r0=(\d+) fallback_lines_r0=(\d+) end (\S+) wall=(\d+)s')
EXIT_LINE = re.compile(r'^EXIT (\d+)\s*$', re.M)
OPRS_LOG = re.compile(r'\[oprs\] (OPRS_\w+) (.*)')      # what the runtime prints, one line per event


def write_csv(path, rows):
    """Exclusive create: a table is never overwritten. Columns = union of the rows' keys, in order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval='')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    return path


def exclusive(paths):
    """Refuse to overwrite a derived table (it is rebuilt from ALL raw batches: move the old one aside)."""
    for path in map(Path, paths):
        if path.exists():
            sys.exit(f'{path} exists: move it aside first (a table is rebuilt from all raw batches)')


# ---- Megatron training -----------------------------------------------------------------------
def training_arm(run_dir, start, end, tp):
    name, kind = start['name'], start['kind']
    out = run_dir / name
    text = (out / f'out_r{tp - 1}.log').read_text(errors='replace') if (out / f'out_r{tp - 1}.log').exists() else ''
    iters = ITER.findall(text)
    steps = [float(m[0]) for m in iters]
    allocs = [float(x) for x in MAX_ALLOC.findall(text)]
    ready, hits, fallbacks, ranks_with_hits = {}, {}, {}, 0
    for rank in range(tp):
        path = out / f'out_r{rank}.log'
        if not path.exists():
            continue
        log = path.read_text(errors='replace')
        found = READY.search(log)
        if found:
            ready[rank] = {'plan': found[4], 'backend': found[5], 'entries': int(found[6]), 'contexts': int(found[8])}
        any_hit = False
        for m in HIT.finditer(log):
            hits[(m[2], m[7])] = hits.get((m[2], m[7]), 0) + int(m[8])
            any_hit = True
        ranks_with_hits += any_hit
        for m in FALLBACK.finditer(log):
            fallbacks[m[2]] = fallbacks.get(m[2], 0) + int(m[3])
    plan_path = Path(start['plan']) if start['plan'] not in ('none', '') else None
    nonserial = None
    if plan_path is not None and plan_path.exists():
        nonserial = sum(e['choice'] != 'serial' for e in json.loads(plan_path.read_text())['entries'])
    # Exit status. mpirun's own is 1 for every arm that brought up the engine world: a rank that
    # initialised MPI and leaves without MPI_Finalize is an "abnormal termination" to prterun, and
    # that is how the engine arms of this stack leave (teardown-order law). The arm's real code is
    # the `EXIT <n>` line its rank printed after python returned.
    printed = EXIT_LINE.findall(text)
    if printed:
        rc = int(printed[-1])
    elif end and end['py_rc'] not in (None, 'none'):                   # the driver writes py_rc=none if so
        rc = int(end['py_rc'])
    else:                                       # killed before python returned: mpirun's status (124 = timeout)
        rc = int(end['rc']) if end else None
    mpirun_rc = int(end['rc']) if end else None
    hit_total = sum(hits.values())
    if end is None or rc != 0 or not steps:
        status = 'failed'
    elif kind == 'serial':
        status = 'ok'
    elif nonserial is None:
        status = 'ok'
    elif nonserial and not hit_total:
        status = 'NO-HITS'                      # the plan replaces something, nothing was replaced: broken, not slow
    elif fallbacks.get('unknown_shape') or fallbacks.get('dtype_mismatch'):
        status = 'shape-mismatch'               # a call site was not in the plan
    else:
        status = 'ok'
    digests = sorted({v['plan'] for v in ready.values()})
    tail = steps[2:]
    median = statistics.median(tail) if tail else None
    return {'arm': name, 'kind': kind, 'plan': plan_path.name if plan_path is not None else '',
            'plan_digest': digests[0] if len(digests) == 1 else ';'.join(digests),
            'plan_on_ranks': f'{len(ready)}/{tp}', 'ranks_with_hits': f'{ranks_with_hits}/{tp}',
            'backend': next((v['backend'] for v in ready.values()), ''),
            'rc': rc, 'mpirun_rc': mpirun_rc, 'steps': len(steps), 'step_ms_median': round(median, 2) if median is not None else None,
            'step_ms_all': [round(s, 1) for s in steps], 'train_loss': iters[-1][1] if iters else None,
            'grad_norm': iters[-1][2] if iters else None, 'nan_iters': int(iters[-1][4]) if iters else None,
            'skipped_iters': int(iters[-1][3]) if iters else None,
            'val_loss': (VAL.findall(text) or [None])[-1], 'max_alloc_mb': max(allocs) if allocs else None,
            'hits': {f'{site}={choice}': n for (site, choice), n in sorted(hits.items())},
            'fallbacks': dict(sorted(fallbacks.items())), 'status': status,
            'wall_s': int(end['wall']) if end else None}


def training_rows(run_dir, tp):
    starts, ends = {}, {}
    if not (run_dir / 'driver.log').exists():
        sys.exit(f'no driver.log in {run_dir}: not a run directory of e2e/megatron/run_oprs.sh')
    for line in (run_dir / 'driver.log').read_text(errors='replace').splitlines():
        m = START.search(line)
        if m:
            starts[m[1]] = {'name': m[1], 'kind': m[2], 'plan': m[3], 'start': m[8]}
            continue
        m = END.search(line)
        if m:
            ends[m[1]] = {'rc': m[3], 'py_rc': m[4], 'wall': m[10]}
    return [training_arm(run_dir, start, ends.get(name), tp) for name, start in starts.items()]


def paired(reference, arm):
    """Ratios of paired runs (by run index inside this batch): reference step time / arm step time."""
    pairs = [s / a for s, a in zip(reference, arm) if s and a]
    if not pairs:
        return None
    return {'ratios': [round(p, 4) for p in pairs], 'median': round(statistics.median(pairs), 4),
            'min': round(min(pairs), 4), 'max': round(max(pairs), 4)}


def cmd_training(args):
    run_dir = Path(args.dir)
    found = re.search(r'tp(\d+)', run_dir.name)                 # the driver names its run dirs ..._tp<N>_...
    if not args.tp and not found:
        sys.exit('the run directory name carries no tp<N>: give --tp')
    tp = args.tp or int(found[1])
    rows = training_rows(run_dir, tp)
    if not rows:
        sys.exit(f'no arm in {run_dir}/driver.log')
    csv_path = run_dir / 'e2e_training.csv'
    if csv_path.exists():
        csv_path.unlink()                       # summary of one run: rebuilt in place
    write_csv(csv_path, rows)
    print(f"# Megatron 训练端到端：{run_dir.name}（TP={tp}，每臂独立进程；步长中位数自第 3 步起）\n")
    print("| 臂 | kind | plan | 退出码 | 步数 | 步长中位 (ms) | 训练 loss | 验证 loss | 最大已分配 (MB) | "
          "命中（站点=选择:次数） | 回退（原因:次数） | 状态 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        hits = ' '.join(f'{k}:{v}' for k, v in r['hits'].items()) or '-'
        falls = ' '.join(f'{k}:{v}' for k, v in r['fallbacks'].items()) or '-'
        med = '-' if r['step_ms_median'] is None else (
            f"{r['step_ms_median']}" + ('' if r['status'] == 'ok' else ' ✗'))
        print(f"| {r['arm']} | {r['kind']} | {r['plan'] or '-'} | {r['rc']} | {r['steps']} | {med} | "
              f"{r['train_loss'] or '-'} | {r['val_loss'] or '-'} | {r['max_alloc_mb'] or '-'} | {hits} | {falls} | "
              f"{r['status']} |")
    print("\n退出码 = 每个臂最后一个 rank 日志里的 `EXIT n`（python 的真实退出码；CSV 另列 mpirun 自身的返回值）。"
          "带引擎世界的臂 mpirun 恒返回 1：进程不做 MPI_Finalize 即离开，prterun 记为异常终止（本机栈的拆解顺序定律），"
          "不表示训练失败。命中与回退为各 rank 日志中 OPRS_HIT / OPRS_FALLBACK 行的求和（训练结束后的评估若继续派发，"
          "其增量行也计入）。\n")
    print("✗ = 状态不是 ok 的臂（失败、超时被杀、计划要替换却一次没替换、站点形状不在计划里）：它的步长不进入任何比值"
          "——坏掉的臂不当作慢臂。\n")
    # ratios only among arms that completed and dispatched what their plan says
    by, order = {}, []
    for r in rows:
        if r['status'] == 'ok' and r['step_ms_median'] is not None:
            if r['kind'] not in by:
                order.append(r['kind'])
            by.setdefault(r['kind'], []).append(r['step_ms_median'])
    serial = by.get('serial', [])
    print("## 同批次配对比值（步长时间之比，> 1 = 更快；第 k 次该臂 对 第 k 次 serial）\n")
    print("| 臂 kind | 对 serial：各对 | 中位 | 最小..最大 |")
    print("|---|---|---|---|")
    for kind in order:
        if kind == 'serial':
            continue
        p = paired(serial, by[kind])
        shown = (f"{p['ratios']} | {p['median']:.4f} | {p['min']:.4f}..{p['max']:.4f}" if p
                 else '无 serial 臂可配对 | - | -')
        print(f'| {kind} | {shown} |')
    for other in ('always_deo', *[k for k in order if k.startswith('always_') and k != 'always_deo']):
        if 'oprs' in by and other in by:
            p = paired(by[other], by['oprs'])
            if p:
                print(f"\nOPRS 对 {other}（{other} 步长 / oprs 步长，> 1 = OPRS 更快）：{p['ratios']} → 中位 "
                      f"{p['median']:.4f}（{p['min']:.4f}..{p['max']:.4f}）")
    return 0


# ---- vLLM inference ---------------------------------------------------------------------------
CELLS = ('init', 'ttft', 'ttft2', 'thru')
DRIVER_END = re.compile(r'\[e2e\] (\S+) kind=(\S+) rc=(\d+) end')


def inference_runs(run_dir, M):
    """One entry per arm run of e2e/vllm/run_oprs.sh. Arm names carry their round (s1, s2, ...): a name
    without its trailing digits is the arm, the digits order its runs inside the batch."""
    path = run_dir / 'voprs.csv'
    rows = [r for r in csv.DictReader(path.open()) if int(r['M']) == M] if path.exists() else []
    seen, runs, order, current = {}, {}, [], None
    for r in rows:                                 # the rows of one arm run are contiguous (rank 0 appends)
        if r['arm'] != current:                    # a new arm run starts with its `init` row
            current = r['arm']
            base = r['arm'].rstrip('0123456789') or r['arm']
            k = seen.get(base, 0)
            seen[base] = k + 1
            runs[(base, k)] = {'base': base, 'name': r['arm'], 'kind': r['kind'], 'plan': r['plan_digest'],
                               'backend': r.get('backend', ''), 'cells': {}, 'last': r}
            order.append((base, k))
        if r['cell'].startswith('warmup'):        # an unrecorded-by-design cell: never a sample
            continue
        run = runs[(base, k)]
        run['cells'].setdefault(r['cell'], []).append(r)
        run['last'] = r
    for line in (run_dir / 'driver.log').read_text(errors='replace').splitlines() if (run_dir / 'driver.log').exists() else []:
        m = DRIVER_END.search(line)
        if m and int(m[3]) != 0 and not any(run['name'] == m[1] for run in runs.values()):
            runs[(m[1], 0)] = {'base': m[1], 'name': m[1], 'kind': m[2], 'plan': '', 'backend': '', 'cells': {},
                               'last': {'status': f'FAILED rc={m[3]} (no row written)'}}
            order.append((m[1], 0))
    for run in runs.values():
        run['med'] = {cell: statistics.median(float(r['wall_s']) for r in rep) for cell, rep in run['cells'].items()}
        # the arm script writes its status with its last rows; an arm that stopped after `init` has none
        run['status'] = run['last'].get('status') or (
            'incomplete: stopped after init' if set(run['cells']) == {'init'} else '')
    return [runs[key] for key in order]


def cmd_inference(args):
    run_dir, M = Path(args.dir), int(args.M)
    runs = inference_runs(run_dir, M)
    gates = sorted(run_dir.glob(f'gatetext_*_M{M}.json'))
    if not runs and not gates:
        sys.exit(f'no row for M={M} in {run_dir}')
    timed = [r for r in runs if any(c in r['cells'] for c in CELLS[1:])]
    if any(run['cells'] for run in runs):
        csv_path = run_dir / 'e2e_inference.csv'
        if csv_path.exists():
            csv_path.unlink()                   # summary of one run: rebuilt in place
        write_csv(csv_path, [r for run in runs for rep in run['cells'].values() for r in rep])
    if not timed and any('gate_text' in r['cells'] for r in runs):
        print(f"# vLLM 自然文本检查：{run_dir.name}（M = {M}；本目录只跑了 VS_GATE=1，没有计时格）\n")
        print("| 臂 | kind | plan | 自然文本提示耗时 (s) | 实际预填 token | SP 前向 / 原版前向 | 命中 | 回退 | "
              "载入 / 准备 (ms) | 状态 |")
        print('|---|---|---|---|---|---|---|---|---|---|')
        for run in runs:
            last = run['last']
            hits = f"{last.get('hits', '')} {last.get('hit_detail', '')}".strip() or '-'
            walls = run['med'].get('gate_text')
            pre = run['cells'].get('gate_text', [{}])[0].get('prefilled_tokens')
            print(f"| {run['name']} | {run['kind']} | {run['plan'] or '-'} | "
                  f"{f'{walls:.3f}' if walls is not None else '—'} | {pre or '-'} | "
                  f"{last.get('sp_forwards', '-')} / {last.get('stock_forwards', '-')} | {hits} | "
                  f"{last.get('fallbacks', '-') or '-'} | {last.get('load_ms', '-') or '-'} / "
                  f"{last.get('prepare_ms', '-') or '-'} | {run['status']} |")
        gates = sorted(run_dir.glob(f'gatetext_*_M{M}.json'))
        if gates:
            gate_table(gates)
        print("\n提示各位置 logprob 差 = stock 与该臂各自模型对同一个提示 token 给出的 logprob 之差，"
              "与贪心 token 是否打平无关。带随机权重的小模型只能说明算术与通路，不代表任何真实模型。")
        return 0
    if timed:
        kinds = sorted({r['kind'] for r in runs})
        print(f"# vLLM 预填充端到端：{run_dir.name}（M = {M}；每格 = 该臂各遍的中位时间，单位 s）\n")
        print("| 臂 | kind | plan | " + ' | '.join(c for c in CELLS if any(c in r['cells'] for r in timed)) +
              ' | 实际预填 token（ttft / ttft2 / thru） | SP 前向 / 原版前向 | 命中 | 回退 | 载入 / 准备 (ms) | 状态 |')
        print('|---|---|---|' + '---|' * sum(any(c in r['cells'] for r in timed) for c in CELLS) + '---|---|---|---|---|---|')
        for run in runs:
            shown = [f"{run['med'][c]:.3f}" if c in run['med'] else '—' for c in CELLS if any(c in r['cells'] for r in timed)]
            pre = [f"{statistics.median(float(r['prefilled_tokens']) for r in run['cells'][c]):.0f}"
                   if c in run['cells'] and run['cells'][c][0]['prefilled_tokens'] else '-' for c in CELLS[1:]]
            last = run['last']
            hits = f"{last.get('hits', '')} {last.get('hit_detail', '')}".strip() or '-'
            print(f"| {run['name']} | {run['kind']} | {run['plan'] or '-'} | " + ' | '.join(shown) +
                  f" | {' / '.join(pre)} | {last.get('sp_forwards', '-')} / {last.get('stock_forwards', '-')} | "
                  f"{hits} | {last.get('fallbacks', '-') or '-'} | {last.get('load_ms', '-') or '-'} / "
                  f"{last.get('prepare_ms', '-') or '-'} | {run['status']} |")
        print("\n状态不是 ok 的臂（NO-SP-FORWARD：从未跑到 M-token 的 SP 前向；NO-HITS：计划替换了站点却一次没替换；"
              "FAILED：进程以非零码退出；incomplete：init 之后没有留下记录）不进入任何比值。比值一律用同批次配对。\n")
        ratio_tables(runs, kinds)
    if gates:
        gate_table(gates)
    elif timed:
        print('\n（本目录没有自然文本检查文件 gatetext_*_M<M>.json：数值一致性未测；用 VS_GATE=1 另跑一批）')
    return 0


def _pair_ratio(reference, arm, cell):
    """reference median / arm median for one cell, or None."""
    if cell not in reference['med'] or cell not in arm['med'] or not arm['med'][cell]:
        return None
    return reference['med'][cell] / arm['med'][cell]


def _by_kind(runs, kind):
    return [r for r in runs if r['kind'] == kind]


def _fmt(ratios):
    if not ratios:
        return '-'
    if len(ratios) == 1:
        return f'{ratios[0]:.3f}'
    return (f"{'，'.join(f'{v:.3f}' for v in ratios)} → {statistics.median(ratios):.3f} "
            f"[{min(ratios):.3f}..{max(ratios):.3f}]")


def ratio_tables(runs, kinds):
    """Every arm against the two references of the handoff (original vLLM = stock, and the SP form on
    RCCL = sp_rccl), then each constant-choice arm against the predicted plan. Only runs whose status
    is ok take part: a broken arm is never a slow arm."""
    runs = [r for r in runs if r['status'] == 'ok']
    refs = [k for k in ('stock', 'sp_rccl') if k in kinds]
    print('## 比值（中位时间之比，> 1 = 更快；第 k 次该臂 对 第 k 次参照臂；多次时给出 中位 [最小..最大]）\n')
    print('| 臂 | 格 | ' + ' | '.join(f'对 {k}' for k in refs) + ' |')
    print('|---|---|' + '---|' * len(refs))
    bases = list(dict.fromkeys(r['base'] for r in runs))
    for cell in ('ttft', 'ttft2', 'thru'):
        if not any(cell in r['med'] for r in runs):
            continue
        for base in bases:
            arms = [r for r in runs if r['base'] == base and cell in r['med']]
            if not arms:
                continue
            cols = []
            for ref_kind in refs:
                if arms[0]['kind'] == ref_kind:
                    cols.append('—')
                    continue
                of_kind = _by_kind(runs, ref_kind)
                if not of_kind:                       # no completed run of that reference: no ratio, not a crash
                    cols.append('-')
                    continue
                values = [_pair_ratio(of_kind[min(k, len(of_kind) - 1)], arm, cell) for k, arm in enumerate(arms)]
                cols.append(_fmt([v for v in values if v is not None]))
            print(f"| {base}（{arms[0]['kind']}） | {cell} | " + ' | '.join(cols) + ' |')
    predicted = _by_kind(runs, 'oprs')
    notes = []
    for kind in [k for k in kinds if k.startswith('always_') or k == 'oprs_deo_only']:
        for cell in ('ttft', 'ttft2', 'thru'):
            values = [_pair_ratio(arm, predicted[min(k, len(predicted) - 1)], cell)
                      for k, arm in enumerate(r for r in runs if r['kind'] == kind)]
            if any(v is not None for v in values):
                notes.append(f"| {kind} / oprs | {cell} | {_fmt([v for v in values if v is not None])} |")
    if notes:
        print('\n| 恒定选择臂 / OPRS | 格 | 比值（> 1 = OPRS 更快） |')
        print('|---|---|---|')
        print('\n'.join(notes))


def gate_table(paths):
    """Natural-text gate of every arm against the stock arm (same numbers as tools/vllm_gate_report.py, plus
    the prompt-logprob statistics when the gate recorded them)."""
    arms = {}
    for path in paths:
        j = json.loads(path.read_text())
        arms[j['arm']] = j
    stock = next((a for a, j in arms.items() if j.get('kind') == 'stock' or a == 'stock'), None)
    if stock is None:
        print('\n自然文本检查：没有 stock 臂的文件，无法比较')
        return
    ref = arms[stock]['prompts']
    print(f"\n## 自然文本输出检查（M = {arms[stock]['M']}，每臂 {len(ref)} 个提示 × {arms[stock]['osl']} 个贪心 token，"
          f"对 {stock}）\n")
    print("| 臂 | 与 stock 完全相同的 token | 各提示第一个不同位置 | 首 token logprob 差 | 相同前缀上 max logprob 差 "
          "| 提示各位置 logprob 差 max / 平均 |")
    print('|---|---|---|---|---|---|')
    for arm, j in arms.items():
        if arm == stock:
            continue
        same = tot = 0
        firsts, d0, dmax, pdiff = [], 0.0, 0.0, []
        for r, x in zip(ref, j['prompts']):
            n = min(len(r['ids']), len(x['ids']))
            tot += n
            first = next((i for i in range(n) if r['ids'][i] != x['ids'][i]), None)
            firsts.append('无' if first is None else str(first))
            same += sum(r['ids'][i] == x['ids'][i] for i in range(n))
            for i in range(n if first is None else first):
                token = str(r['ids'][i])
                if token in x['top5'][i]:
                    diff = abs(r['top5'][i][token] - x['top5'][i][token])
                    dmax = max(dmax, diff)
                    d0 = max(d0, diff) if i == 0 else d0
            pdiff += [abs(p - q) for p, q in zip(r.get('prompt_lp') or [], x.get('prompt_lp') or [])
                      if p is not None and q is not None]
        lp = f"{max(pdiff):.4f} / {statistics.fmean(pdiff):.5f}" if pdiff else '-'
        print(f"| {arm} | {same}/{tot} | {' / '.join(firsts)} | {d0:.4f} | {dmax:.4f} | {lp} |")


# ---- operator samples ------------------------------------------------------------------------
SAMPLE_COLUMNS = ['batch', 'file', 'line', 'batch_id', 'kind', 'point_id', 'model', 'family', 'site', 'origin',
                  'table_op', 'op', 'M', 'N', 'K', 'tp', 'dtype', 'topology', 'candidate', 'status', 'protocol',
                  'data_class', 'anomaly', 'correctness_pass', 'rel_vs_serial', 'serial_reference_median_ms',
                  'candidate_median_ms', 'speedup', 'lower_observed', 'upper_observed', 'serial_gemm_only_ms',
                  'serial_comm_only_ms', 'commit', 'src_dirty', 'support_status', 'support_evidence', 'error',
                  'warning']


def operator_rows(files, tp, topology=None):
    rows = []
    for file in files:
        file = Path(file)
        with file.open() as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f'WARNING: unreadable line {file}:{number} (crashed writer?) -- not tabulated', file=sys.stderr)
                    continue
                if item.get('tp') != tp or (topology and (item.get('route') or {}).get('topology') not in (None, topology)):
                    continue
                med = item.get('median_ms') or {}
                correctness = item.get('correctness') or {}
                support = item.get('support') if isinstance(item.get('support'), dict) else {}
                rows.append({
                    'batch': file.parent.name, 'file': file.name, 'line': number, 'batch_id': item.get('batch_id'),
                    'kind': item.get('kind'), 'point_id': item.get('point_id'), 'model': item.get('model'),
                    'family': item.get('family'), 'site': item.get('site'), 'origin': item.get('origin'),
                    'table_op': item.get('table_op'), 'op': item.get('op'), 'M': item.get('M'), 'N': item.get('N'),
                    'K': item.get('K'), 'tp': item.get('tp'), 'dtype': item.get('dtype'),
                    'topology': (item.get('route') or {}).get('topology'), 'candidate': item.get('candidate'),
                    'status': item.get('status', 'ok'), 'protocol': item.get('protocol'),
                    'data_class': item.get('data_class'), 'anomaly': item.get('anomaly'),
                    'correctness_pass': correctness.get('pass') if correctness.get('checked') else None,
                    'rel_vs_serial': correctness.get('rel_vs_serial'),
                    'serial_reference_median_ms': item.get('serial_reference_median_ms'),
                    'candidate_median_ms': item.get('candidate_median_ms'), 'speedup': item.get('speedup'),
                    'lower_observed': item.get('lower_observed'), 'upper_observed': item.get('upper_observed'),
                    'serial_gemm_only_ms': med.get('gemm_only'), 'serial_comm_only_ms': med.get('comm_only'),
                    'commit': item.get('commit'), 'src_dirty': item.get('src_dirty'),
                    'support_status': support.get('status'), 'support_evidence': support.get('evidence'),
                    'error': item.get('error'), 'warning': item.get('warning')})
    return rows


def cmd_operator(args):
    root = Path(args.root)
    files = args.data or sorted(root.glob('raw/operators/*/samples_*.jsonl'))
    if not files:
        sys.exit('no sample file: give --data or run under a results root with raw/operators/*/samples_*.jsonl')
    rows = operator_rows(files, args.tp, args.topology)
    path = root / 'tables' / 'operator_samples.csv'
    exclusive([path])
    write_csv(path, rows)
    count = {}
    for r in rows:
        key = (r['kind'], r['candidate'], r['status'])
        count[key] = count.get(key, 0) + 1
    print(f'[oprs_tables] {len(rows)} rows from {len(files)} file(s) at tp={args.tp} -> {path}')
    for key, n in sorted(count.items()):
        print(f'  {key[0]:<9} {key[1]:<12} {key[2]:<12} {n}')
    return 0


# ---- predictor choices (family holdout) -------------------------------------------------------
def cmd_choices(args):
    """One evaluation per data class and per candidate set: the serial/DEO two-way, and the three-way
    with each backend B that has samples (one deployment revolves around one B). Classes are never
    pooled: a number always travels with its data class."""
    from stowaway.oprs import evaluate as E
    from stowaway.oprs.data import load_legacy_cells, load_samples, usable
    from stowaway.oprs.export_plan import LEGACY_LIMITATION, NEW_LIMITATION, sha256_file
    from stowaway.oprs.platform import BACKENDS, DEO, detect_versions

    root, tp = Path(args.root), args.tp
    topology = args.topology or f'BW{tp}'
    classes = []                                   # (data_class, records, sources, limitation)
    if args.legacy:
        records = load_legacy_cells(args.legacy, tp=tp, topology=topology, versions=detect_versions(),
                                    dtype=args.dtype)
        classes.append(('legacy', records, [sha256_file(args.legacy)], LEGACY_LIMITATION))
    data = args.data
    if data is None and not args.legacy:           # every new-protocol batch of this results root
        data = sorted(root.glob('raw/operators/*/samples_*.jsonl'))
    if data:
        records, failures = load_samples(data)
        for failure in failures:
            print(f"WARNING: failed point kept in the raw file, out of the evaluation: "
                  f"{failure.get('model')}/{failure.get('table_op') or failure.get('site')}/M{failure.get('M')} "
                  f"[{failure.get('candidate')}] {failure.get('status')}: "
                  f"{failure.get('error') or failure.get('support')}", file=sys.stderr)
        classes.append(('b2b16', records, [sha256_file(p) for p in data], NEW_LIMITATION))
    if not classes:
        sys.exit('nothing to evaluate: give --legacy or --data')
    flat, summary, results = [], [], {}
    for data_class, records, sources, limitation in classes:
        present = sorted({r.method for r in usable(records) if r.kind == 'operator'})
        wanted = [m for m in (args.methods.split(',') if args.methods else present) if m]
        missing = sorted(set(wanted) - set(present))
        if missing:
            print(f'WARNING: no usable operator sample of {missing} in the {data_class} data: not evaluated',
                  file=sys.stderr)
        if DEO not in wanted:
            continue
        for methods in [(DEO,)] + [(DEO, b) for b in BACKENDS if b in wanted]:
            name = f'{data_class}_{"+".join(methods)}'
            try:
                result = E.evaluate(records, methods, exclude_anomaly=args.exclude_anomaly, sources=sources,
                                    limitation=limitation)
            except ValueError as exc:              # e.g. too few families to hold out: reported, never a number
                print(f'WARNING: {name} not evaluated: {exc}', file=sys.stderr)
                continue
            results[name] = result
            flat += [{'candidates': '+'.join(methods), **row} for row in E.choice_rows(result)]
            for policy, scopes in result['summary'].items():
                summary += [{'data_class': data_class, 'classification': result['classification'],
                             'candidates': '+'.join(methods), 'exclude_anomaly': args.exclude_anomaly,
                             'policy': policy, 'scope': scope, **v} for scope, v in scopes.items() if v]
    if not summary:
        sys.exit('nothing could be evaluated (see the warnings)')
    out = root / 'tables'
    exclusive([out / name for name in ('predictor_choices.csv', 'predictor_summary.csv',
                                       'predictor_family_holdout.json')])
    write_csv(out / 'predictor_choices.csv', flat)
    write_csv(out / 'predictor_summary.csv', summary)
    with (out / 'predictor_family_holdout.json').open('x') as handle:
        json.dump(results, handle, indent=1)
    print(f'[oprs_tables] {len(flat)} choice rows, {len(summary)} summary rows -> {out}')
    print("\n| 数据类别 | 候选 | 策略 | 范围 | 点数 | 选对率 | 选中方案几何平均加速 | 低于 serial 的点数 | "
          "对 oracle 的 regret 比 | oracle 几何平均加速 |")
    print('|---|---|---|---|---|---|---|---|---|---|')
    for s in summary:
        if s['scope'] == 'all' or args.verbose:
            print(f"| {s['classification']} | {s['candidates']} | {s['policy']} | {s['scope']} | {s['count']} | "
                  f"{s['accuracy']:.3f} | {s['selected_gmean_speedup']:.3f} | {s['selected_below_stock']} | "
                  f"{s['regret_gmean_ratio']:.3f} | {s['oracle_gmean_speedup']:.3f} |")
    return 0


# ---- target call sites -----------------------------------------------------------------------
def cmd_sites(args):
    """Every call site of the plans with its predicted scores, static support and whether its exact
    (table_op, M, N, K) was measured as an operator-table point (a site that was not is off-table)."""
    from stowaway.oprs.data import load_samples
    from stowaway.oprs.platform import DEO, static_support
    from stowaway.oprs.workloads import SITE_TABLE_OP

    files = []
    for target in args.plans:
        path = Path(target)
        if path.is_dir():
            data = json.loads((path / 'manifest.json').read_text())
            files += [path / item['path'] for item in data['plans'].values()]
        else:
            files.append(path)
    measured = set()
    if args.data:
        records, _ = load_samples(args.data)
        measured = {(r.table_op, *r.shape) for r in records if r.kind == 'operator'}
    rows = []
    for file in files:
        plan = json.loads(file.read_text())
        backend = plan['external_backend']
        methods = [m for m in dict.fromkeys((DEO, backend)) if m != 'none']
        provenance = plan.get('provenance') or {}
        named = {(o['site'], o['op'], o['M'], o['N'], o['K']): o['origin'] for o in provenance.get('origins', [])}
        outside = {}
        for o in provenance.get('out_of_range', []):
            outside.setdefault((o['site'], o['op'], o['M'], o['N'], o['K']), []).append(
                f"{o['method']}:{''.join(o['outside'])}")
        for e in plan['entries']:
            shape = (e['M'], e['N'], e['K'])
            rows.append({
                'plan': file.name, 'workload': plan['workload'], 'tp': plan['tp'], 'site': e['site'],
                'origin': named.get((e['site'], e['op'], *shape), ''), 'op': e['op'],
                'table_op': SITE_TABLE_OP[e['site']],
                'M': e['M'], 'N': e['N'], 'K': e['K'], 'choice': e['choice'],
                'predicted_log_speedup': e['predicted_log_speedup'],
                **{f'predicted_log_speedup_{m}': e['scores'].get(m) for m in methods},
                **{f'support_{m}': static_support(m, e['op'], *shape, plan['tp'])[0] for m in methods},
                'out_of_range': ';'.join(outside.get((e['site'], e['op'], *shape), [])),
                'on_operator_table': ((SITE_TABLE_OP[e['site']], *shape) in measured) if measured else None})
    path = Path(args.root) / 'tables' / 'target_sites.csv'
    exclusive([path])
    write_csv(path, rows)
    print(f'[oprs_tables] {len(rows)} site rows -> {path}')
    return 0


def parse(argv=None):
    """argparse.Namespace of one command line."""
    parser = argparse.ArgumentParser(description='tables and summaries of the BW1000 OPRS experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('training', help='Megatron A/B of e2e/megatron/run_oprs.sh')
    p.add_argument('dir')
    p.add_argument('--tp', type=int, help='default: from the run directory name (…_tp<N>_…)')
    p.set_defaults(fn=cmd_training)
    p = sub.add_parser('inference', help='vLLM A/B of e2e/vllm/run_oprs.sh')
    p.add_argument('dir')
    p.add_argument('M', type=int)
    p.set_defaults(fn=cmd_inference)
    p = sub.add_parser('operator', help='tables/operator_samples.csv from the raw operator batches')
    p.add_argument('root')
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--topology')
    p.add_argument('--data', nargs='+')
    p.set_defaults(fn=cmd_operator)
    p = sub.add_parser('choices', help='tables/predictor_choices.csv (family holdout)')
    p.add_argument('root')
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--topology')
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--legacy')
    p.add_argument('--data', nargs='+')
    p.add_argument('--methods', help='comma list; default: every method with usable samples')
    p.add_argument('--exclude-anomaly', action='store_true')
    p.add_argument('--verbose', action='store_true', help='also the per-op scopes')
    p.set_defaults(fn=cmd_choices)
    p = sub.add_parser('sites', help='tables/target_sites.csv: call sites of plans, on- or off-table')
    p.add_argument('root')
    p.add_argument('--plans', nargs='+', required=True, help='plan-set directory or plan files')
    p.add_argument('--data', nargs='+', help='operator samples, to tell on- from off-table sites')
    p.set_defaults(fn=cmd_sites)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    sys.exit(args.fn(args) or 0)


if __name__ == '__main__':
    main()
