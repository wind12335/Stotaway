"""Plan set of ONE deployment (one external backend B) for an end-to-end workload.

  python3 tools/oprs_plans.py <new dir> --tp 8 --backend async_tp --workload bw27b_train \\
         --data results/<date>/raw/operators/<batch>/samples_tp8_bfloat16.jsonl [--seq 4096 --mbs 8] [--note TEXT]

A plan gives every call site of the workload one choice: serial / deo / <B>. Every plan of the set is
fitted on the same operator samples -- one data class only: --data = records of bench/oprs_matrix.py,
or --legacy = planner/data/cells.json (the historical table; such plans are labelled legacy) -- and
exported by stowaway.oprs.export_plan (never overwritten: a new set goes into a new directory):

  oprs_<B>.json        the plan: at each call site whichever of serial / deo / <B> the fitted route models predict
  oprs_deo_only.json   two-way plan (serial / deo): the deployment for when no external backend is runnable
  always_deo.json      constant-choice arms (DEO where it is measured and shape-supported)
  always_<B>.json      constant-choice arm of the external backend
  null.json            Null arm: the contexts of oprs_<B>.json are allocated, nothing is replaced
  manifest.json        files, digests, data class, choices, fit / export time
  export.log           what export_plan printed for each plan
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from stowaway.oprs import export_plan  # noqa: E402
from stowaway.oprs.platform import BACKENDS, DEO  # noqa: E402
from stowaway.oprs.predictor import SERIAL  # noqa: E402

SCHEMA = 'stowaway-oprs-plan-set-v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description='plan set of one deployment (one external backend B)')
    parser.add_argument('out', type=Path, help='NEW directory')
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--backend', required=True, choices=BACKENDS)
    parser.add_argument('--workload', required=True, choices=('bw27b_train', 'bw70b_train', 'custom_train',
                                                              'llama70_infer', 'hf_infer'))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--legacy', help='planner/data/cells.json (DEO only, historical protocol)')
    source.add_argument('--data', nargs='+', help='raw/operators/*.jsonl of bench/oprs_matrix.py')
    parser.add_argument('--topology', help='default BW<tp>')
    parser.add_argument('--dtype', default='bfloat16', choices=('bfloat16', 'float16'))
    parser.add_argument('--seq', type=int, default=4096)
    parser.add_argument('--mbs', type=int, default=8)
    parser.add_argument('--rows', help='prefill chunk size(s) of an inference workload (default 16384)')
    parser.add_argument('--config', help='model config.json or inline JSON (custom_train / hf_infer)')
    parser.add_argument('--exclude-anomaly', action='store_true')
    parser.add_argument('--note', default='')
    args = parser.parse_args(argv)
    backend, out = args.backend, args.out
    if out.exists():
        sys.exit(f'refusing to overwrite {out}: history is kept, choose a new directory')
    out.mkdir(parents=True)
    common = ['--workload', args.workload, '--tp', str(args.tp), '--dtype', args.dtype, '--seq', str(args.seq),
              '--mbs', str(args.mbs)]
    common += ['--topology', args.topology] if args.topology else []
    common += ['--rows', args.rows] if args.rows else []
    common += ['--config', args.config] if args.config else []
    common += ['--legacy', args.legacy] if args.legacy else ['--data', *args.data]
    common += ['--exclude-anomaly'] if args.exclude_anomaly else []
    plans = {
        'oprs': dict(file=f'oprs_{backend}.json', argv=['--backend', backend, '--note', args.note]),
        'oprs_deo_only': dict(file='oprs_deo_only.json', argv=['--backend', 'none', '--note', args.note]),
        'always_deo': dict(file='always_deo.json', argv=['--backend', backend, '--force', DEO]),
        f'always_{backend}': dict(file=f'always_{backend}.json', argv=['--backend', backend, '--force', backend]),
        'null': dict(file='null.json', argv=['--backend', backend, '--force', SERIAL,
                                             '--allocate-from', str(out / f'oprs_{backend}.json')]),
    }
    manifest = {'schema': SCHEMA, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'tp': args.tp, 'backend': backend, 'workload': args.workload, 'dtype': args.dtype,
                'workload_args': {'seq': args.seq, 'mbs': args.mbs, 'rows': args.rows, 'config': args.config},
                'topology': args.topology or f'BW{args.tp}', 'exclude_anomaly': args.exclude_anomaly,
                'note': args.note, 'generator': 'tools/oprs_plans.py', 'plans': {}}
    with (out / 'export.log').open('w') as log:
        for arm, spec in plans.items():
            print(f'== {arm} -> {spec["file"]}', file=log, flush=True)
            saved, sys.stdout = sys.stdout, log
            try:
                export_plan.main([*common, *spec['argv'], '--output', str(out / spec['file'])])
            finally:
                sys.stdout = saved
            plan = json.loads((out / spec['file']).read_text())
            provenance = plan['provenance']
            manifest['plans'][arm] = {
                'path': spec['file'], 'sha256': sha256(out / spec['file']), 'arm': provenance['arm'],
                'external_backend': plan['external_backend'], 'data_class': provenance['data_class'],
                'commit': provenance.get('commit'), 'src_dirty': provenance.get('src_dirty'),
                'overheads': provenance.get('overheads'), 'allocate': len(plan['allocate']),
                'choices': {e['site']: e['choice'] for e in plan['entries']},
                'limitation': provenance['limitation']}
            manifest.setdefault('data_class', provenance['data_class'])
    with (out / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(f'[oprs_plans] tp={args.tp} backend={backend} workload={args.workload} data_class={manifest["data_class"]} '
          f'-> {out}')
    for arm, item in manifest['plans'].items():
        print(f'  {arm:<20} {item["path"]:<24} ' + ' '.join(f'{s}={c}' for s, c in item['choices'].items())
              + f'  contexts={item["allocate"]}')


if __name__ == '__main__':
    main()
