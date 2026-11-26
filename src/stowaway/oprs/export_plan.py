"""Fit BW route models on BW-local operator samples and export an executable workload plan.

Hygon port of nvidia/planner/original_a800/export_tp8_auto_plan.py with the hard-coded A800 paths,
device, TP and example shapes replaced by parameters. Every fitted route is built from samples
measured on THIS machine under ONE data class (the legacy table or new-protocol records); the two
classes are never mixed in one fit because their Routes differ. No A800 parameter is loaded.

A plan fixes one external backend B (or none) and gives every call site one of serial / DEO / B.
Constant-choice plans for the end-to-end arms are exported with --force:
    --force deo | <backend>            Always DEO / Always B (B only where measured and shape-supported)
    --force serial --allocate-from P   Null arm: same contexts as plan P, nothing replaced
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .auto_plan import SCHEMA
from .data import LEGACY, NEW, grouped, load_legacy_cells, load_samples, usable
from .platform import BACKENDS, DEO, DTYPES, detect_versions, static_support
from .predictor import SERIAL
from .route_local_residual import RouteLocalResidual
from .workloads import build_workload

LEGACY_LIMITATION = ('historical BW1000 8-card operator table (planner/data/cells.json): per-op event timing '
                     'inside bursts of 16, warm-up and environment knobs unrecorded, forward-only fused DEO '
                     'operators, software versions assumed to be the current stack. A plan fitted on it is an '
                     'interim artifact for getting the pipeline through; re-measure under the b2b16 protocol '
                     '(bench/oprs_matrix.py) and regenerate before any publication claim')
NEW_LIMITATION = ('operator-level estimates under the b2b16 protocol; the plan needs in-framework (end-to-end) '
                  'validation on the same workload')
ARMS = {SERIAL: 'serial', DEO: 'always_deo', **{b: f'always_{b}' for b in BACKENDS}}


def sha256_file(path) -> dict:
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json_exclusive(path, payload) -> Path:
    """Never overwrite: history is kept, a new export goes to a new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return path


def git_commit() -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=20).stdout.strip() or 'unknown'
        dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--', 'src'],
                                    capture_output=True, text=True, timeout=20).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return {'commit': 'unknown', 'src_dirty': None}
    return {'commit': head, 'src_dirty': dirty}


def fit_routes(records, *, exclude_anomaly=False) -> dict:
    """{(method, op): RouteLocalResidual} fitted on all usable records of each BW route."""
    fitted = {}
    protocols = sorted({r.protocol for r in records})
    if len(protocols) > 1:                        # the two data classes have different Routes: never one fit
        raise ValueError(f'one data class per fit, got {protocols}')
    for method, op in sorted({(r.method, r.op) for r in records}):
        rows = grouped(records, method, op, exclude_anomaly=exclude_anomaly)
        if len({r.family for r in rows}) < 4:
            raise ValueError(f'{method}/{op}: four model families with a measured sample are required')
        fitted[(method, op)] = RouteLocalResidual(rows[0].sample.route).fit(rows)
    return fitted


def load_records(args):
    """(records, sources, data_class, limitation) of exactly one data class."""
    versions = detect_versions()
    if args.legacy:
        records = load_legacy_cells(args.legacy, tp=args.tp, topology=args.topology, versions=versions,
                                    dtype=args.dtype)
        return records, [sha256_file(args.legacy)], LEGACY, LEGACY_LIMITATION
    records, failures = load_samples(args.data)
    if failures:
        print(f'WARNING: {len(failures)} failed points are kept in the raw files and excluded from the fit')
    if not records:
        raise ValueError('no usable sample records')
    return records, [sha256_file(p) for p in args.data], NEW, NEW_LIMITATION


def build_plan(workload, tp, *, backend, platform, dtype, records=(), fitted=None, sources=(), data_class=None,
               limitation=None, exclude_anomaly=False, force=None, allocate_from=None, note='', commit=None) -> dict:
    methods = (DEO,) if backend == 'none' else (DEO, backend)
    fitted, entries, allocate, demoted, unmeasured = fitted or {}, [], set(), [], set()
    used = [r for r in usable(records, exclude_anomaly=exclude_anomaly) if (r.method, r.op) in fitted]
    span = {}                 # (method, op) -> [min, max] of M, N and K over the samples the route was fitted on
    for r in used:
        bounds = span.setdefault((r.method, r.op), [[math.inf, 0], [math.inf, 0], [math.inf, 0]])
        for d in range(3):
            bounds[d][0], bounds[d][1] = min(bounds[d][0], r.shape[d]), max(bounds[d][1], r.shape[d])
    origins, out_of_range = [], []                  # (site, origin, shape) of each entry: provenance, the entry keys are the schema
    for site in workload['sites']:
        op, shape = site['op'], (site['M'], site['N'], site['K'])
        # exactly the keys of the plan schema (auto_plan.ENTRY_KEYS): a loader refuses anything else
        entry = {'site': site['site'], 'op': op, 'M': shape[0], 'N': shape[1], 'K': shape[2]}
        origins.append({'site': site['site'], 'origin': site.get('origin', site['site']), 'op': op,
                        'M': shape[0], 'N': shape[1], 'K': shape[2]})
        # A method is a candidate at this site only if it was MEASURED on this route and the shape is
        # inside its static support; anything else stays serial and is recorded.
        gains = {}
        for method in methods:
            if (method, op) not in fitted:
                unmeasured.add(f'{method}/{op}')
                continue
            support, evidence = static_support(method, op, *shape, tp)
            if support != 'supported':
                demoted.append({**entry, 'method': method, 'support': support, 'evidence': evidence})
                continue
            gains[method] = fitted[(method, op)].predict(fitted[(method, op)].route, shape)['predicted_log_speedup']
            # the route model is linear in log (M, N, K): outside the range of its samples a score is an
            # extrapolation, and it is recorded (qkv / proj of the vehicles are off-table by construction)
            outside = [name for d, name in enumerate('MNK')
                       if not span[(method, op)][d][0] <= shape[d] <= span[(method, op)][d][1]]
            if outside:
                out_of_range.append({'site': site['site'], 'origin': site.get('origin', site['site']), 'op': op,
                                     'M': shape[0], 'N': shape[1], 'K': shape[2], 'method': method,
                                     'outside': outside, 'predicted_log_speedup': gains[method]})
        scores = {SERIAL: 0.0, **gains}
        if force is None:
            choice = max(scores, key=scores.get)        # serial is first: a method needs a positive gain
        elif force in gains:
            choice = force
        else:
            choice = SERIAL
        entry.update(choice=choice, predicted_log_speedup=scores.get(choice), scores=scores)
        entries.append(entry)
        if choice != SERIAL:
            allocate.add((choice, op, *shape))
    derived, arm = None, ARMS.get(force, 'oprs') if force else 'oprs'
    if allocate_from is not None:
        source = json.loads(Path(allocate_from).read_text())
        allocate = {tuple(item) for item in source['allocate']}
        derived = {**sha256_file(allocate_from), 'external_backend': source['external_backend'],
                   'arm_of_source': (source.get('provenance') or {}).get('arm')}
        # the Null arm has no fit of its own: it borrows the contexts of another plan and replaces nothing
        arm, data_class = 'null', 'derived'
        limitation = ('Null arm derived from another plan: the same contexts are allocated, no call site is '
                      'replaced; measures the hook + heap cost of the OPRS arm it accompanies')
    samples = [{'method': r.method, 'op': r.op, 'family': r.family, 'model': r.model, 'table_op': r.table_op,
                'M': r.shape[0], 'N': r.shape[1], 'K': r.shape[2], 'stock_ms': r.sample.stock_ms,
                'candidate_ms': r.sample.candidate_ms, 'origin': r.origin} for r in used]
    return {'schema': SCHEMA, 'platform': platform, 'workload': workload['workload'], 'tp': tp, 'dtype': dtype,
            'external_backend': backend, 'default': SERIAL, 'candidates': [SERIAL, *methods],
            'entries': entries, 'allocate': [list(item) for item in sorted(allocate)],
            'workload_source': workload.get('source'), 'workload_note': workload.get('note'),
            'provenance': {'sources': list(sources), 'data_class': data_class, 'limitation': limitation,
                           'exclude_anomaly': exclude_anomaly, 'arm': arm, 'origins': origins,
                           'out_of_range': out_of_range,
                           'out_of_range_note': 'a call site outside [min, max] of the fitted samples of a '
                                                'route in the listed dimensions: its score for that method is '
                                                'an extrapolation of a log-linear model, not an interpolation',
                           'force': force, 'allocate_from': derived, 'demoted': demoted,
                           'unmeasured': sorted(unmeasured), 'note': note,
                           'method': 'independent per-route models: ridge on (log M, log N, log K) plus '
                                     'family-holdout local residual (RouteLocalResidual), each with its own '
                                     'measured serial reference',
                           'routes': [asdict(m.route) for m in fitted.values()],
                           'fit': {f'{method}/{op}': m.to_dict() for (method, op), m in fitted.items()},
                           'families': sorted({s['family'] for s in samples}), 'n_samples': len(samples),
                           'samples': samples, **(commit if commit is not None else git_commit()),
                           'generator': 'stowaway.oprs.export_plan'}}


def main(argv=None):
    import time
    parser = argparse.ArgumentParser(description='Export a BW automatic route plan (serial / DEO / one backend B)')
    parser.add_argument('--workload', required=True, choices=('bw27b_train', 'bw70b_train', 'custom_train',
                                                        'llama70_infer', 'hf_infer', 'qwen36_train'))
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--topology', help='default BW<tp>')
    parser.add_argument('--dtype', default='bfloat16', choices=DTYPES)
    parser.add_argument('--seq', type=int, default=4096)
    parser.add_argument('--mbs', type=int, default=8)
    parser.add_argument('--rows', help='global rows, or a comma list for several prefill chunk sizes; '
                        'default seq x mbs (training workloads)')
    parser.add_argument('--config', help='model config.json or inline JSON (qwen36_train, custom_train, '
                        'hf_infer; llama70_infer optional)')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--legacy', help='planner/data/cells.json (DEO only, historical protocol)')
    source.add_argument('--data', nargs='+', help='raw/operators/*.jsonl of bench/oprs_matrix.py')
    parser.add_argument('--backend', default='none', choices=('none',) + BACKENDS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--force', choices=(SERIAL, DEO,) + BACKENDS,
                        help='constant-choice end-to-end arm plan instead of the predicted plan')
    parser.add_argument('--allocate-from', type=Path, help='with --force serial: Null arm sharing this plan\'s contexts')
    parser.add_argument('--exclude-anomaly', action='store_true')
    parser.add_argument('--note', default='')
    args = parser.parse_args(argv)
    from .platform import runtime_platform
    args.topology = args.topology or f'BW{args.tp}'
    if args.allocate_from is not None and args.force != SERIAL:
        parser.error('--allocate-from is the Null arm: it requires --force serial')
    if args.allocate_from is not None and args.backend == 'none':
        # the Null arm allocates the contexts of its source plan, so it must offer that plan's backend
        # (a context of a method the plan does not offer cannot be allocated)
        args.backend = json.loads(args.allocate_from.read_text()).get('external_backend') or 'none'
        if args.backend != 'none':
            print(f'[export_plan] Null arm: external backend {args.backend} taken from {args.allocate_from}')
    if args.legacy is None and not args.data and args.force != SERIAL:
        parser.error('a predicted or always-on plan needs --legacy or --data: a method without a measured '
                     'sample cannot be a candidate (only the Null arm, --force serial, needs no data)')
    workload = build_workload(args.workload, args.tp, mbs=args.mbs, seq=args.seq, rows=args.rows, config=args.config)
    commit = git_commit()                       # read before the clock: git latency is not plan generation
    import importlib
    importlib.import_module('torch')             # warmed before the clock; the version guard needs no device
    t0 = time.perf_counter()
    if args.legacy or args.data:
        records, sources, data_class, limitation = load_records(args)
        fitted = fit_routes(records, exclude_anomaly=args.exclude_anomaly)
    else:                                   # constant arm without a fit, or the Null arm derived from a plan
        records, sources, data_class, limitation, fitted = [], [], None, None, {}
    t1 = time.perf_counter()                # offline fit time (runtime overhead record, handoff section 3.4)
    # the plan says where its samples were measured; this offline tool never opens a device itself
    measured = next((r.raw.get('platform') for r in records if (r.raw or {}).get('platform')), {})
    if args.backend != 'none' and args.force in (None, args.backend) \
            and not any(m == args.backend for m, _ in fitted):
        parser.error(f'--backend {args.backend}: no measured {args.backend} sample in the data; a path that is '
                     f'not measured cannot be a candidate')
    plan = build_plan(workload, args.tp, backend=args.backend, dtype=args.dtype,
                      platform=runtime_platform(tp=args.tp, topology=args.topology, probe_device=False,
                                                device_name=measured.get('device_name'), arch=measured.get('arch')),
                      records=records, fitted=fitted,
                      sources=sources,
                      data_class=data_class, limitation=limitation, exclude_anomaly=args.exclude_anomaly,
                      force=args.force, allocate_from=args.allocate_from, note=args.note, commit=commit)
    plan['provenance']['topology'] = args.topology
    plan['provenance']['overheads'] = {'fit_s': round(t1 - t0, 4), 'export_s': round(time.perf_counter() - t1, 4)}
    write_json_exclusive(args.output, plan)
    print(f"[export_plan] fit_s={plan['provenance']['overheads']['fit_s']} "
          f"export_s={plan['provenance']['overheads']['export_s']}")
    named = {(o['site'], o['op'], o['M'], o['N'], o['K']): o['origin'] for o in plan['provenance']['origins']}
    for entry in plan['entries']:
        print(f"{entry['site']:>4} {named.get((entry['site'], entry['op'], entry['M'], entry['N'], entry['K']), ''):<9} "
              f"{entry['op']} M={entry['M']} N={entry['N']} K={entry['K']} "
              f"-> {entry['choice']:<12} {json.dumps(entry['scores'])}")
    for item in plan['provenance']['demoted']:
        print(f"demoted {item['method']} at {item['site']} M={item['M']} N={item['N']} K={item['K']}: {item['support']}")
    for item in plan['provenance']['out_of_range']:
        print(f"out-of-range {item['site']} {item['method']}: {''.join(item['outside'])} "
              f"(predicted {item['predicted_log_speedup']:+.3f} is an extrapolation)")
    print(f"allocate={plan['allocate']} arm={plan['provenance']['arm']} data_class={data_class} -> {args.output}")


if __name__ == '__main__':
    main()
