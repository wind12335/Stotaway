"""Audit three-arm Syncopate TP2 measurements and preserve actual order."""
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    directory = Path('/root/data/stowaway-nvidia/results')
    groups = []
    for op in ('ag', 'rs'):
        raw = (root / f'experiments/tp2_syncopate_{op}_20261113.json').read_bytes()
        design = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        signature = None
        blocks = []
        for block, order in enumerate(design['orders'], 1):
            arms = {}
            for arm in order:
                path = directory / f'tp2_syncopate_{op}_{arm}_b{block}_20261113.json'
                data = path.read_bytes()
                row = json.loads(data)
                if row['arm'] != arm or row['target'] != design['target'] or row['manifest_sha256'] != digest:
                    raise ValueError(f'provenance mismatch: {path}')
                for key in ('tp', 'dtype', 'batch', 'warmup', 'repeats'):
                    if row[key] != design[key]:
                        raise ValueError(f'protocol mismatch {key}: {path}')
                if row['checks'] != 2 or row['cuda_device_max_connections'] != 'unset':
                    raise ValueError(f'checks/environment mismatch: {path}')
                sig = tuple(row[k] for k in ('gpu', 'torch', 'triton', 'cuda_build'))
                if signature is not None and sig != signature:
                    raise ValueError(f'software or GPU mismatch: {path}')
                signature = sig
                samples = row['samples_ms']
                if len(samples) != design['repeats'] or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError(f'invalid samples: {path}')
                if not math.isclose(statistics.median(samples), row['median_ms'], rel_tol=1e-9):
                    raise ValueError(f'bad median: {path}')
                arms[arm] = {'median_ms': row['median_ms'], 'path': str(path),
                             'sha256': hashlib.sha256(data).hexdigest()}
            blocks.append({'block': block, 'order': order, 'arms': arms,
                           'dx_vs_syncopate': arms['syncopate']['median_ms'] / arms['dx']['median_ms'],
                           'dx_vs_stock': arms['stock']['median_ms'] / arms['dx']['median_ms'],
                           'syncopate_vs_stock': arms['stock']['median_ms'] / arms['syncopate']['median_ms']})
        gm = lambda field: math.exp(statistics.mean(math.log(b[field]) for b in blocks))
        groups.append({'target': design['target'], 'manifest_sha256': digest, 'blocks': blocks,
                       'dx_vs_syncopate_gmean': gm('dx_vs_syncopate'),
                       'dx_vs_stock_gmean': gm('dx_vs_stock'),
                       'syncopate_vs_stock_gmean': gm('syncopate_vs_stock')})
    result = {'groups': groups, 'warning': 'Exploratory already-smoke-tested shapes; independent processes. '
              'AG block 2 Syncopate retried after rank temp-file race; earlier block-2 stock and DX retained. '
              'RS uses adapted upstream transform/copy-engine path, not paper hand-written RS. '
              'Excludes setup, planning and E2E.'}
    out = directory / 'tp2_syncopate_comparison_20261113.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    for g in groups:
        print(g['target']['op'], 'DX/Syncopate', g['dx_vs_syncopate_gmean'],
              'DX/stock', g['dx_vs_stock_gmean'], 'Syncopate/stock', g['syncopate_vs_stock_gmean'])


if __name__ == '__main__':
    main()
