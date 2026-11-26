"""Operator-level sample matrix on BW1000 under the b2b16 protocol (handoff doc, section 3.2).

ONE process measures, per shape, the serial path and every overlap candidate against it:
  serial      AG -> GEMM (op ag) or GEMM -> RS (op rs) with RCCL collectives: the reference of the bar
  candidates  deo (Hygon's own overlap method) / async_tp / triton_dist (the external backends B),
              each through the same exchange adapters the runtime dispatcher uses

Per point (model, table_op, M):
  inputs        random, generated in fp32, scale 0.05 (the convention of bench/ag_gemm.py, gemm_rs.py)
  correctness   #1 before any timing, #2 after the timing on fresh inputs: the legacy metric
                max |out - ref| / max(|ref|, 1) over all ranks < tol (2e-2); fp32 reference recorded
  timing        WARMUP calls per arm, then 8 rounds (Python cyclic GC off inside: its pauses were
                measured landing in timed groups); each round runs every arm once in a rotated order
                (no arm is always first), one arm's group = 16 back-to-back calls timed with events on
                the compute stream, group time / 16; the sample kept is the SLOW rank's (all_reduce
                MAX); every rank's own groups are stored too
  bar           median(serial_8) / median(candidate_8)   (never a mean of ratios)
  range         min / max of median(serial_8) / candidate_j   (observed range, not a confidence interval)
  parts         after the rounds: the GEMM alone and the collective alone of the serial path (which
                half of the serial reference moved: the diagnostic asked for the two legacy anomalies)

Rows: one JSON record per (point, candidate), schema stowaway-oprs-operator-sample-v1, read by
stowaway.oprs.data.load_samples. A candidate whose shape is outside its static support is an
`unsupported` row (never a 0), a failed check is an `ok` row with correctness.pass = false (the fit
excludes it), a crash writes `error` rows and the process leaves; a rerun with --resume continues
after the finished points. This is an operator-level, forward-only measurement: not an in-framework
result, and the training backward is not timed here.

  --first-batch            7 models x 3 ops x M=32768 (21 shapes)   --table   all 99 points
  --models a,b --ms ..     a subset of the table                    --points model/op/M,...
  --sites <workload>       call-site shapes of an end-to-end workload (kind 'site': a target,
                           never training material)
  --plan P.json            plan dispatch check (section 6.4): every non-serial entry of a real plan
                           timed through Dispatcher.run (kind 'plan_dispatch'), with its hit count
  --list [--tp R]          print the selected points and exit (no device needed)

Launch: bench/run_oprs_matrix.sh (mpirun -np R; SW_NP before env/env.sh).
"""
import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
import uuid
import zlib
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:                 # --list / direct calls without env/env.sh
    sys.path.insert(0, str(ROOT / 'src'))
DEFAULT_PORT = 29991
# plan site name of an operator-table point (workloads.SITE_TABLE_OP): the bench has no call site
SITE_OF_TABLE_OP = {'ag_gu': 'fc1', 'ag_up': 'qkv', 'rs': 'fc2'}
TIMING_SCOPE = ('compute-stream events around 16 back-to-back calls of the complete projection path '
                '(AG -> GEMM or GEMM -> RS, or the fused operator); output allocation and kernel '
                'launches inside, no synchronisation between the calls, no layout restore')
WARNING = ('operator-level, forward-only, measured in one process against the serial path of that '
           'process; not an in-framework result')
REPEATS, BATCH = 8, 16


def clean(value):
    """JSON-safe copy: a non-finite float (a NaN in a failed check) becomes its name, not a crash."""
    if isinstance(value, float):
        return value if math.isfinite(value) else f'{value}'
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def rotate(items, r):
    r %= len(items)
    return items[r:] + items[:r]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state(path):
    """(commit, dirty) of one checkout, by path."""
    try:
        commit = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, timeout=20).stdout.strip() or 'unknown'
        dirty = bool(subprocess.run(['git', '-C', str(path), 'status', '--porcelain'], capture_output=True,
                                    text=True, timeout=60).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return 'unknown', None
    return commit, dirty


def hy_smi():
    """Card rows of hy-smi (temperature, power, utilisation): the device state travels with the numbers."""
    try:
        text = subprocess.run(['hy-smi'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in text.splitlines() if line.strip()[:1].isdigit()]


# ---- point selection (CPU only) -----------------------------------------------------------------------------
def select_points(args, tp):
    from stowaway.oprs.workloads import FIRST_BATCH, RECHECK, build_workload, operator_inventory
    if args.points:
        wanted = [x for x in args.points.split(',') if x]
        points = [p for p in operator_inventory(tp) if p['point_id'] in wanted]
        missing = sorted(set(wanted) - {p['point_id'] for p in points})
        if missing:
            raise ValueError(f'unknown points {missing} (expected model/table_op/M of the operator table)')
    elif args.first_batch:
        points = operator_inventory(tp, FIRST_BATCH, (32768,))
    elif args.sites:
        workload = build_workload(args.sites, tp, mbs=args.mbs or 8, seq=args.seq or 4096, rows=args.rows,
                                  config=args.config)
        # proj and fc2 share table_op 'rs': a call site is named by its site, never by its operator
        points = [{'kind': 'site', 'model': workload['workload'], 'site': s['site'], 'origin': s['origin'],
                   'table_op': s['table_op'], 'op': s['op'], 'M': s['M'], 'N': s['N'], 'K': s['K'],
                   'point_id': f"{workload['workload']}/{s['site']}/M{s['M']}", 'origin_id': True}
                  for s in workload['sites']]
        for point in points:
            point.pop('origin_id')
    else:
        models = [x for x in (args.models or '').split(',') if x]
        ms = [int(x) for x in (args.ms or '').split(',') if x]
        points = operator_inventory(tp, *([models] if models else []), *([ms] if ms else []))
    for point in points:
        if 'point_id' not in point:
            point['point_id'] = f"{point['model']}/{point['table_op']}/M{point['M']}"
        point.setdefault('origin', point['point_id'])
        # a re-measurement of a legacy anomaly keeps its label (same exclusion switch as the old table)
        legacy = f'legacy-anomaly:{point["model"]}/{point["table_op"]}/M{point["M"]}'
        point['anomaly'] = legacy if legacy.removeprefix('legacy-anomaly:') in RECHECK else None
        point['recheck_of'] = point['point_id'] if point['anomaly'] else None
    return points


def point_key(point, candidate):
    return (point['kind'], point['model'], point['table_op'], point['site'], point['M'], point['N'],
            point['K'], candidate)


# ---- measurement ------------------------------------------------------------------------------------------------
def inputs(op, m, n, k, tp, dt, rank, seed):
    """(activation, weight) of one shape: op ag -> [M/TP, K] x [N, K]; op rs -> [M, K] x [N, K]."""
    import torch
    g = torch.Generator(device='cuda').manual_seed(seed + rank)
    rows = m // tp if op == 'ag' else m
    x = (torch.randn(rows, k, generator=g, dtype=torch.float32, device='cuda') * 0.05).to(dt)
    w = (torch.randn(n, k, generator=g, dtype=torch.float32, device='cuda') * 0.05).to(dt)
    return x, w


def serial(op, x, w):
    """The serial path exactly as Dispatcher.serial defines it: AG -> GEMM, or GEMM -> RS."""
    import torch
    import torch.distributed as dist
    if op == 'ag':
        full = torch.empty(x.shape[0] * dist.get_world_size(), x.shape[1], dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(full, x.contiguous())
        return torch.matmul(full, w.detach().t())
    part = torch.matmul(x, w.detach().t())
    out = torch.empty(x.shape[0] // dist.get_world_size(), w.shape[0], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, part.contiguous())
    return out


def serial_fp32(op, x, w, tp):
    import torch
    import torch.distributed as dist
    wf = w.detach().float().t()
    if op == 'ag':
        full = torch.empty(x.shape[0] * tp, x.shape[1], dtype=torch.float32, device=x.device)
        dist.all_gather_into_tensor(full, x.float())
        return torch.matmul(full, wf)
    out = torch.empty(x.shape[0] // tp, w.shape[0], dtype=torch.float32, device=x.device)
    dist.reduce_scatter_tensor(out, torch.matmul(x.float(), wf).contiguous())
    return out


def rel(value, reference):
    """Legacy metric (bench/gemm_rs.py): max |x - ref| / max(|ref|, 1), max over the ranks."""
    import torch.distributed as dist
    v = ((value.float() - reference.float()).abs() / reference.float().abs().clamp_min(1.0)).max().reshape(1)
    dist.all_reduce(v, op=dist.ReduceOp.MAX)
    return v.item()


def rel_rms(value, reference):
    """||x - ref|| / ||ref|| over all ranks (scale-free companion of the legacy metric)."""
    import torch.distributed as dist
    d2 = (value.float() - reference.float()).pow(2).sum().reshape(1)
    r2 = reference.float().pow(2).sum().reshape(1)
    dist.all_reduce(d2), dist.all_reduce(r2)
    return math.sqrt(d2.item() / r2.item())


def all_finite(tensor):
    import torch
    import torch.distributed as dist
    ok = torch.isfinite(tensor).all().reshape(1).to(torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


def check(cand, op, m, n, k, tp, dt, rank, seed):
    """Correctness of one candidate on fresh inputs against the serial output and an fp32 reference."""
    import torch
    x, w = inputs(op, m, n, k, tp, dt, rank, seed)
    with torch.no_grad():
        ref, out = serial(op, x, w), cand(x, w)
    torch.cuda.synchronize()
    if tuple(out.shape) != tuple(ref.shape) or out.dtype != ref.dtype:
        raise RuntimeError(f'output {tuple(out.shape)}/{out.dtype} != serial {tuple(ref.shape)}/{ref.dtype}')
    ref32 = serial_fp32(op, x, w, tp)
    return {'rel_vs_serial': rel(out, ref), 'rel_vs_fp32': rel(out, ref32),
            'serial_rel_vs_fp32': rel(ref, ref32), 'rel_rms_vs_serial': rel_rms(out, ref),
            'finite': all_finite(out)}


TRACE = os.environ.get('OPRS_MATRIX_TRACE', '') == '1'     # diagnostic: per-call times of every group

if TRACE:                                    # which Python GC pauses fall into the timed groups?
    import gc
    import time as _time

    def _gc_phase(phase, info):
        if phase == 'start':
            _gc_phase.t0 = _time.perf_counter()
        else:
            dt = (_time.perf_counter() - getattr(_gc_phase, 't0', _time.perf_counter())) * 1e3
            if dt > 0.5:
                print(f'[oprs-trace] python gc gen={info["generation"]} pause_ms={dt:.2f} '
                      f'(collected {info["collected"]})', flush=True)
    gc.callbacks.append(_gc_phase)


def group_time(fn, batch, name=''):
    """(own ms per call, slow-rank ms per call) of one group: `batch` back-to-back calls."""
    import torch
    import torch.distributed as dist
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(batch)] if TRACE else []
    start.record()
    for i in range(batch):
        fn()
        if TRACE:
            marks[i].record()
    end.record()
    end.synchronize()
    if TRACE:
        print(f'[oprs-trace] rank={dist.get_rank()} arm={name} per-call ms '
              f'{[round(m.elapsed_time(n), 3) for m, n in zip([start] + marks[:-1], marks)]}', flush=True)
    own = start.elapsed_time(end) / batch
    t = torch.tensor([own], device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return own, t.item()


def measure(fns, groups, warmup):
    """`groups` rounds; each round runs every arm once, rotated so that no arm is always first.
    Returns (own samples, slow-rank samples, run order).

    Python's cyclic garbage collector is off inside the timed region (OPRS_MATRIX_TRACE showed its
    pauses landing in timed groups: generation-2 collections of this process take 10-50 ms, generation-1
    ones 0.5-1.6 ms). What is left is reference counting, as in any Python host loop."""
    import gc
    import torch
    import torch.distributed as dist
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    gc.collect(), gc.freeze()                # long-lived objects leave the collector's work for good
    was_enabled = gc.isenabled()
    gc.disable()
    torch.cuda.synchronize()
    dist.barrier()
    names = list(fns)
    own = {name: [] for name in names}
    slow = {name: [] for name in names}
    order = [rotate(names, r) for r in range(groups)]
    for round_order in order:
        for name in round_order:
            o, s = group_time(fns[name], BATCH, name)
            own[name].append(o)
            slow[name].append(s)
    if was_enabled:
        gc.enable()
    return own, slow, order


def summary(slow):
    serial_ms, candidate_ms = statistics.median(slow['serial']), statistics.median(slow['candidate'])
    return {'serial_ms': serial_ms, 'candidate_ms': candidate_ms, 'speedup': serial_ms / candidate_ms,
            'lower_observed': serial_ms / max(slow['candidate']),
            'upper_observed': serial_ms / min(slow['candidate'])}


def candidate_call(exchange, op):
    """What Dispatcher.run calls on a hit."""
    if op == 'ag':
        return lambda x, w: exchange.all_gather_gemm(x.contiguous(), w.detach())
    return lambda x, w: exchange.gemm_reduce_scatter(x.contiguous(), w.detach())


# ---- one point ---------------------------------------------------------------------------------------------------
def run_point(world, point, methods, args, info, emit, log):
    import torch
    import torch.distributed as dist
    from stowaway.oprs.backends import create_exchange
    from stowaway.oprs.data import SAMPLE_SCHEMA
    from stowaway.oprs.platform import DEO, WARMUP, bw_route, heap_bytes, static_support

    tp = world.size
    m, n, k, op = point['M'], point['N'], point['K'], point['op']
    dt = getattr(torch, args.dtype)
    base = {key: point[key] for key in ('kind', 'point_id', 'model', 'site', 'origin', 'table_op', 'op', 'anomaly',
                                    'recheck_of')}
    base.update(family=point['model'], M=m, N=n, K=k, tp=tp, dtype=args.dtype, schema=SAMPLE_SCHEMA,
                protocol='b2b16', data_class='b2b16', batch=BATCH, repeats=REPEATS, warmup=WARMUP,
                interleave='rotating: every round runs each arm once, arm i leads round i',
    python_gc='cyclic GC disabled inside the timed region (gc.collect + gc.freeze before it)', **info['common'])
    support = {method: static_support(method, op, m, n, k, tp) for method in methods}
    runnable = [method for method in methods if support[method][0] == 'supported']
    rows = [dict(base, candidate=method, status='unsupported', model=point['model'],
                 support=dict(zip(('status', 'evidence'), support[method])),
                 route=asdict(bw_route(method, op, tp=tp, topology=info['topology'], versions=info['versions'],
                                       dtype=args.dtype)))
            for method in methods if method not in runnable]
    seed = zlib.crc32(point['point_id'].encode()) % 1000000000
    # heap of this point's contexts, the same formula the dispatcher checks a plan with
    need = sum(heap_bytes(method, op, m, n, k, tp, deo_bwd=False) for method in runnable)
    if need > info['heap_bytes']:
        rows += [dict(base, candidate=method, status='error', model=point['model'],
                      error=f'contexts need {need} bytes of symmetric heap, DUSHMEM_SYMMETRIC_SIZE is '
                            f'{info["heap_bytes"]}') for method in runnable]
        emit(rows)
        return rows
    if not runnable:
        emit(rows)
        return rows
    log(f"{point['point_id']} candidates={runnable} heap_mb={need / 2**20:.0f} "
        f"{'(recheck of a legacy anomaly)' if point['anomaly'] else ''}")
    exchanges, deo_contexts = {}, None
    try:
        x, w = inputs(op, m, n, k, tp, dt, world.rank, seed + 2)
        if DEO in runnable:
            from stowaway.oprs.backends.deo import DeoContexts
            deo_contexts = DeoContexts(world, bwd='stock')            # forward only: no backward context
        for method in sorted(runnable):                               # the dispatcher's canonical order
            exchanges[method] = create_exchange(method, world, op, m, n, k, dt, group=world.group,
                                                deo_contexts=deo_contexts)
        cands = {method: candidate_call(exchange, op) for method, exchange in exchanges.items()}
        first = {method: check(cands[method], op, m, n, k, tp, dt, world.rank, seed + 1) for method in runnable}
        fns = {'serial': lambda: serial(op, x, w), **{method: (lambda c=cands[method]: c(x, w)) for method in runnable}}
        own, slow, order = measure(fns, REPEATS, WARMUP)
        second = {method: check(cands[method], op, m, n, k, tp, dt, world.rank, seed + 3) for method in runnable}
        # parts of the serial path: GEMM alone and collective alone (diagnostic, after the rounds)
        if op == 'ag':
            full = torch.empty(m, k, dtype=dt, device='cuda')
            dist.all_gather_into_tensor(full, x)
            parts = {'gemm_only': lambda: torch.matmul(full, w.detach().t()),
                     'comm_only': lambda: dist.all_gather_into_tensor(full, x)}
        else:
            part = torch.matmul(x, w.detach().t())
            out = torch.empty(m // tp, n, dtype=dt, device='cuda')
            parts = {'gemm_only': lambda: torch.matmul(x, w.detach().t()),
                     'comm_only': lambda: dist.reduce_scatter_tensor(out, part)}
        own_parts, slow_parts, _ = measure(parts, REPEATS, WARMUP)
        gathered = [None] * tp
        dist.all_gather_object(gathered, {'own': own, 'own_parts': own_parts})
        torch.cuda.synchronize()
        dist.barrier()
    except BaseException:
        trace = traceback.format_exc()
        log(f"{point['point_id']} FAILED on rank {world.rank}:\n{trace}")
        sys.stdout.flush()
        # the ranks without a row for this point get an error row (rank 0 writes); the process leaves
        have = set()
        for row in rows:
            have.add((row.get('candidate'), row.get('status')))
        missing = [(method, 'error') for method in methods if (method, 'unsupported') not in have
                   and (method, 'error') not in have]
        emit([dict(base, candidate=method, status='error', model=point['model'], error=trace.strip().splitlines()[-1],
                   failed_on_rank=world.rank, traceback=trace) for method, _ in missing], every_rank=False)
        os._exit(3)
    # collective release, reverse order of creation (Dispatcher.close)
    for method in sorted(exchanges, reverse=True):
        exchanges[method].close()
    if deo_contexts is not None:
        deo_contexts.close()
    exchanges, deo_contexts = {}, None
    torch.cuda.empty_cache()
    dist.barrier()
    per_rank = {'serial': [g['own']['serial'] for g in gathered]}
    for method in runnable:
        per_rank[method] = [g['own'][method] for g in gathered]
    per_rank.update({name: [g['own_parts'][name] for g in gathered] for name in ('gemm_only', 'comm_only')})
    state = hy_smi() if world.rank == 0 else None
    for method in runnable:
        passed = first[method]['finite'] and second[method]['finite'] and \
            max(first[method]['rel_vs_serial'], second[method]['rel_vs_serial']) < args.tol
        stats = summary({'serial': slow['serial'], 'candidate': slow[method]})
        parts_ms = {name: statistics.median(slow_parts[name]) for name in ('gemm_only', 'comm_only')}
        rows.append(dict(
            base, candidate=method, status='ok', model=point['model'],
            support=dict(zip(('status', 'evidence'), support[method])),
            route=asdict(bw_route(method, op, tp=tp, topology=info['topology'], versions=info['versions'],
                                  dtype=args.dtype)),
            run_order=order, batch_id=info['batch_id'], point_in_process=info['point_index'],
            serial_samples_ms=slow['serial'], candidate_samples_ms=slow[method],
            serial_reference_median_ms=stats['serial_ms'], candidate_median_ms=stats['candidate_ms'],
            median_ms={'serial': stats['serial_ms'], 'candidate': stats['candidate_ms'], **parts_ms},
            speedup=stats['speedup'], lower_observed=stats['lower_observed'],
            upper_observed=stats['upper_observed'], serial_parts_ms=parts_ms,
            serial_part_samples_ms={name: slow_parts[name] for name in ('gemm_only', 'comm_only')},
            per_rank=per_rank, correctness={'checked': True, 'pass': bool(passed), 'tol': args.tol,
                                            'metric': 'max |out - ref| / max(|ref|, 1), max over ranks',
                                            'before_timing': first[method], 'after_timing': second[method]},
            timing_scope=TIMING_SCOPE, warning=WARNING, gpu_state_at_point=state))
        log(f"{point['point_id']} [{method}] serial={stats['serial_ms']:.3f} ms candidate={stats['candidate_ms']:.3f} ms "
            f"speedup={stats['speedup']:.3f} [{stats['lower_observed']:.3f}, {stats['upper_observed']:.3f}] "
            f"gemm_only={parts_ms['gemm_only']:.3f} comm_only={parts_ms['comm_only']:.3f} ms "
            f"correctness={'PASS' if passed else 'FAIL'} rel={max(first[method]['rel_vs_serial'], second[method]['rel_vs_serial']):.2e}")
    emit(rows)
    return rows


# ---- plan dispatch check (section 6.4) ---------------------------------------------------------------------------
def run_plan(world, args, methods, info, emit, log):
    """Every non-serial entry of a real plan, timed through Dispatcher.run next to the serial path."""
    import torch
    import torch.distributed as dist
    from stowaway.oprs.auto_plan import SCHEMA as PLAN_SCHEMA
    from stowaway.oprs.data import SAMPLE_SCHEMA
    from stowaway.oprs.dispatcher import Dispatcher
    from stowaway.oprs.platform import WARMUP, bw_route

    data = json.loads(Path(args.plan).read_text())
    if data.get('schema') != PLAN_SCHEMA:
        raise ValueError(f'{args.plan}: not a {PLAN_SCHEMA} plan')
    tp = world.size
    dt = getattr(torch, args.dtype)
    if data.get('dtype') != args.dtype:
        raise ValueError(f'plan dtype {data.get("dtype")} != bench dtype {args.dtype}')
    common = {'kind': 'plan_dispatch', 'model': data['workload'], 'family': data['workload'], 'table_op': None,
              'anomaly': None, 'tp': tp, 'dtype': args.dtype, 'schema': SAMPLE_SCHEMA, 'protocol': 'b2b16',
              'data_class': 'plan_dispatch', 'batch': BATCH, 'repeats': REPEATS, 'warmup': WARMUP,
              'interleave': 'rotating: every round runs each arm once, arm i leads round i',
              'python_gc': 'cyclic GC disabled inside the timed region (gc.collect + gc.freeze before it)',
              **info['common'], 'timing_scope': TIMING_SCOPE, 'warning': WARNING + '; plan dispatch check',
              'plan': {'path': str(args.plan), 'digest': '', 'workload': data['workload'],
                       'arm': (data.get('provenance') or {}).get('arm')}}
    # the real runtime entry: plan digest agreed over the TP ranks, contexts in canonical order, heap checked
    disp = Dispatcher(args.plan, data['workload'], group=world.group, methods=methods, deo_bwd='stock',
                      engine=world)
    common['plan'].update(digest=disp.plan.digest, backend=disp.plan.backend)
    dispatch = {'load_ms': disp.load_ms, 'prepare_ms': disp.prepare_ms, 'contexts': len(disp.exchanges),
                'heap_need_mb': disp.heap_bytes / 2**20, 'deo_bwd': 'stock'}
    log(f'plan {args.plan} digest={disp.plan.digest[:12]} workload={data["workload"]} backend={disp.plan.backend} '
        f'load_ms={disp.load_ms:.1f} prepare_ms={disp.prepare_ms:.1f} contexts={len(disp.exchanges)}')
    try:
        for index, entry in enumerate(data['entries']):
            if entry['choice'] == 'serial':
                continue
            site, op = entry['site'], entry['op']
            m, n, k = entry['M'], entry['N'], entry['K']
            point_id = f"{data['workload']}/{site}/M{m}"
            x, w = inputs(op, m, n, k, tp, dt, world.rank, zlib.crc32(point_id.encode()) % 1000000000)
            route = asdict(bw_route(entry['choice'], op, tp=tp, topology=info['topology'],
                                    versions=info['versions'], dtype=args.dtype))
            correctness = check(lambda a, b: disp.run(site, op, a, b), op, m, n, k, tp, dt, world.rank, 11)
            disp.runs.clear()
            hit = (site, op, m, n, k, entry['choice'], 'plan')
            own, slow, order = measure({'serial': lambda: serial(op, x, w),
                                        'candidate': lambda: disp.run(site, op, x, w)}, REPEATS, WARMUP)
            runs = sum(disp.runs.values())
            if disp.runs.get(hit, 0) != runs:                 # every timed call took the plan path
                raise RuntimeError(f'plan dispatch did not take the plan path: {dict(disp.runs)}')
            gathered = [None] * tp
            dist.all_gather_object(gathered, {'own': own})
            stats = summary(slow)
            row = dict(common, candidate=entry['choice'], status='ok', site=site, op=op,
                       origin=entry.get('origin', site), point_id=point_id, M=m, N=n, K=k, route=route,
                       run_order=order,
                       batch_id=info['batch_id'], point_in_process=index, serial_samples_ms=slow['serial'],
                       candidate_samples_ms=slow['candidate'], serial_reference_median_ms=stats['serial_ms'],
                       candidate_median_ms=stats['candidate_ms'],
                       median_ms={'serial': stats['serial_ms'], 'candidate': stats['candidate_ms']},
                       speedup=stats['speedup'], lower_observed=stats['lower_observed'],
                       upper_observed=stats['upper_observed'],
                       per_rank={'serial': [g['own']['serial'] for g in gathered],
                                 'candidate': [g['own']['candidate'] for g in gathered]},
                       correctness={'checked': True, 'pass': correctness['rel_vs_serial'] < args.tol,
                                    'tol': args.tol, 'metric': 'max |out - ref| / max(|ref|, 1), max over ranks',
                                    'before_timing': correctness}, gpu_state_at_point=hy_smi() if world.rank == 0 else None,
                       plan_entry=entry, plan_speedup=math.exp(entry['predicted_log_speedup']),
                       dispatch=dict(dispatch, runs=runs, hits=dict(disp.hits), fallbacks=dict(disp.fallbacks)))
            emit([row])
            log(f"{point_id} [{entry['choice']}] serial={stats['serial_ms']:.3f} ms dispatched={stats['candidate_ms']:.3f} ms "
                f"speedup={stats['speedup']:.3f} plan_predicted={row['plan_speedup']:.3f} runs={runs} "
                f"correctness_rel={correctness['rel_vs_serial']:.2e}")
            x = w = None
            torch.cuda.empty_cache()
    finally:
        disp.close()
        if world.rank == 0:
            disp.print_report()


# ---- driver ------------------------------------------------------------------------------------------------------
def done_set(path):
    """(point, candidate) keys that already have a final row (ok / unsupported)."""
    done = set()
    if Path(path).exists():
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get('status') in ('ok', 'unsupported'):
                    done.add((item['kind'], item['model'], item.get('table_op'), item.get('site'), item['M'],
                              item['N'], item['K'], item['candidate']))
    return done


def parse():
    parser = argparse.ArgumentParser(description='BW operator-level sample matrix under the b2b16 protocol')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--first-batch', action='store_true', help='7 models x 3 ops x M=32768')
    choice.add_argument('--table', action='store_true', help='all 99 operator-table points')
    choice.add_argument('--models', help='comma list of operator-table models (default: all)')
    choice.add_argument('--sites', help='workload name: its call-site shapes (kind site)')
    choice.add_argument('--points', help='comma list model/table_op/M')
    choice.add_argument('--plan', help='plan dispatch check on this plan file')
    parser.add_argument('--ms', help='comma list of M (default 8192,16384,32768)')
    for name in ('seq', 'mbs'):
        parser.add_argument(f'--{name}', type=int)
    for name in ('rows', 'config'):
        parser.add_argument(f'--{name}')
    parser.add_argument('--methods', default='deo,async_tp,triton_dist')
    parser.add_argument('--dtype', default='bfloat16', choices=('bfloat16', 'float16'))
    parser.add_argument('--tol', type=float, default=2e-2)
    parser.add_argument('--topology', help='default BW<tp>')
    parser.add_argument('--tp', type=int, help='card count of --list (a run takes it from mpirun)')
    parser.add_argument('--out', help='JSONL output file (append; one record per point and candidate)')
    parser.add_argument('--resume', action='store_true', help='continue an existing file after its finished rows')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--list', action='store_true', help='print the selected points and exit (no device)')
    return parser.parse_args()


def main():
    args = parse()
    if args.list:
        for point in select_points(args, args.tp or 8):
            print(f"{point['point_id']:<28} op={point['op']} M={point['M']} N={point['N']} K={point['K']}")
        return
    if not args.out:
        raise SystemExit('--out is required')
    from stowaway.oprs.engine import EngineWorld
    world = EngineWorld.standalone(port=args.port)      # MPI + symmetric heap before anything else
    import torch
    import torch.distributed as dist
    from stowaway.oprs.export_plan import git_commit
    from stowaway.oprs.platform import METHODS, detect_versions, runtime_platform

    tp, rank = world.size, world.rank
    if args.tp is not None and args.tp != tp:
        raise SystemExit(f'--tp {args.tp} but mpirun started {tp} ranks')
    methods = [x for x in args.methods.split(',') if x]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown or not methods:
        raise ValueError(f'--methods: choose from {METHODS}, got {unknown or methods}')
    topology = args.topology or f'BW{tp}'
    out = Path(args.out)
    if out.exists() and not args.resume:
        raise SystemExit(f'{out} exists: history is kept -- choose a new file, or pass --resume to continue it')
    versions = detect_versions()
    sources = [Path(__file__).resolve()] + sorted((ROOT / 'src/stowaway/oprs').rglob('*.py')) + \
        [ROOT / 'src/stowaway/kernels/gemm_rs_dx.py', ROOT / 'src/stowaway/kernels/common_ops.py']
    port_root = os.path.realpath(os.environ.get('OPRS_TRITON_DIST_ROOT') or '/root/private_data/tri-dist-hygon')
    for name in ('all_gather_gemm', 'gemm_reduce_scatter', 'common_ops'):
        sources.append(Path(port_root) / 'triton_dist/kernels/dcu' / f'{name}.py')
    sources.append(Path(torch.__file__).parent / 'ops/symm_mem.py')
    port_commit, port_dirty = git_state(port_root)
    git = git_commit()                                  # {'commit', 'src_dirty'} of this checkout
    if rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    info = {
        'batch_id': f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}Z-{socket.gethostname()}-{uuid.uuid4().hex[:8]}",
        'topology': topology, 'versions': versions,
        'heap_bytes': int(os.environ.get('DUSHMEM_SYMMETRIC_SIZE') or 1 << 33),
        'common': {'platform': runtime_platform(tp=tp, topology=topology), 'software_versions': [list(v) for v in versions],
                   **git,
                   'gpu': {'name': torch.cuda.get_device_name(), 'memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                           'hy_smi_at_start': hy_smi() if rank == 0 else None},
                   'source_files': [{'path': str(p), 'sha256': sha256(p)} for p in sources if p.exists()],
                   'triton_dist_source': {'root': port_root, 'commit': port_commit, 'dirty': port_dirty},
                   'fp32_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                   'env': {key: os.environ.get(key) for key in ('DUSHMEM_SYMMETRIC_SIZE', 'HSA_FORCE_FINE_GRAIN_PCIE',
                                                                'OPRS_TRITON_DIST_ROOT', 'HIP_VISIBLE_DEVICES',
                                                                'ROCR_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS')},
                   'source_files_note': 'sha256 of every file on the measured path (bench, OPRS package, kernels, triton_dist port, torch symm_mem ops)',
                   'software': {'comm': 'rccl', 'symmetric': 'dushmem'}}}

    def emit(rows, every_rank=True):
        """Rank 0 appends the rows (fsync); with every_rank=False the other ranks only leave."""
        if rank == 0:
            with out.open('a') as handle:
                for row in rows:
                    handle.write(json.dumps(clean(row), allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        elif not every_rank:
            os._exit(3)

    def log(message):
        if rank == 0:
            print(f'[oprs-matrix] {message}', flush=True)

    log(f"tp={tp} dtype={args.dtype} methods={methods} heap_mb={info['heap_bytes'] / 2**20:.0f} "
        f"topology={topology} batch_id={info['batch_id']} commit={git['commit'][:10]} src_dirty={git['src_dirty']} out={out}")
    if args.plan:
        run_plan(world, args, methods, info, emit, log)
    else:
        points = select_points(args, tp)
        done = done_set(out)
        todo = [p for p in points if any(point_key(p, method) not in done for method in methods)]
        log(f'points={len(points)} todo={len(todo)}')
        for index, point in enumerate(todo):
            if all(point_key(point, method) in done for method in methods):
                continue
            world.host.barrier_all()
            run_point(world, point, methods, args, {**info, 'point_index': index}, emit, log)
            done |= {point_key(point, method) for method in methods}
            torch.cuda.empty_cache()
        if not todo:
            log('nothing to do: every selected point has a final row')
    dist.barrier()
    world.host.barrier_all()
    log(f'done -> {out}')
    sys.stdout.flush()
    try:
        world.mpi.MPI_Finalize()
    except Exception:                                                  # noqa: BLE001 - as vllm_sp.py
        pass
    sys.exit(0)


if __name__ == '__main__':
    main()
