"""Create reviewable coverage and measured TP2 plans from existing data."""
import hashlib
import json
from pathlib import Path


def main():
    results = Path('/root/data/stowaway-nvidia/results')
    source = results / 'tp2_async_comparison_20261113.json'
    raw = source.read_bytes()
    report = json.loads(raw)
    for group in report['groups']:
        for block in group['blocks']:
            for arm in block['arms'].values():
                if hashlib.sha256(Path(arm['path']).read_bytes()).hexdigest() != arm['sha256']:
                    raise ValueError('raw timing digest mismatch')
    example = json.loads(Path(report['groups'][0]['blocks'][0]['arms']['stock']['path']).read_bytes())
    runtime = {k: example[k] for k in ('tp', 'dtype', 'gpu', 'torch', 'triton', 'cuda_build',
                                       'cuda_device_max_connections')}
    measured_entries = []
    for group in report['groups']:
        target = group['target']
        totals = {arm: sum(block['arms'][arm]['median_ms'] for block in group['blocks'])
                  for arm in ('stock', 'async_tp', 'dx')}
        choice = min(totals, key=totals.get)
        measured_entries.append({'site': 'measured_' + target['op'], **target, 'choice': choice})
    template = {'schema': 'stowaway-tp2-replacement-v1', 'external_backend': 'async_tp',
                'runtime': runtime, 'selection_source': 'measured_existing',
                'evidence': {'summary_path': str(source), 'summary_sha256': hashlib.sha256(raw).hexdigest(),
                             'method': 'minimum sum of 3 observed process medians; no estimated overhead',
                             'warning': 'previously observed shapes, not predictor or blind test'},
                'entries': measured_entries}
    coverage = dict(template)
    coverage['selection_source'] = 'manual_coverage'
    coverage['evidence'] = {'method': 'manually forced choices for execution coverage; not a predictor'}
    coverage['entries'] = [
        {'site': 'qkv', 'op': 'ag', 'M': 1024, 'N': 4096, 'K': 4096, 'choice': 'async_tp'},
        {'site': 'o_proj', 'op': 'rs', 'M': 1024, 'N': 4096, 'K': 4096, 'choice': 'dx'},
        {'site': 'fc1', 'op': 'ag', 'M': 1024, 'N': 4096, 'K': 4096, 'choice': 'stock'},
        {'site': 'fc2', 'op': 'rs', 'M': 1024, 'N': 4096, 'K': 4096, 'choice': 'async_tp'},
    ]
    for name, data in (('tp2_replacement_measured_20261113.json', template),
                       ('tp2_replacement_coverage_20261113.json', coverage)):
        path = results / name
        with path.open('x') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        print(name, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
