"""Audit exploratory Async-TP vs DX and stock TP2 timings."""
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    raw = (root / 'experiments/tp2_async_compare_20261113.json').read_bytes()
    design = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path('/root/data/stowaway-nvidia/results')
    groups = []
    for index, target in enumerate(design['targets']):
        blocks = []
        signature = None
        for block, order in enumerate(design['orders'], 1):
            arms = {}
            for arm in order:
                path = directory / f'tp2_async_compare_t{index}_{arm}_b{block}_20261113.json'
                data = path.read_bytes()
                row = json.loads(data)
                if row['arm'] != arm or row['target'] != target or row['manifest_sha256'] != digest:
                    raise ValueError(f'provenance mismatch: {path}')
                for key in ('tp', 'dtype', 'batch', 'warmup', 'repeats'):
                    if row[key] != design[key]:
                        raise ValueError(f'protocol mismatch {key}: {path}')
                if row['checks'] != 2 or row['cuda_device_max_connections'] != 'unset':
                    raise ValueError(f'checks/environment mismatch: {path}')
                sig = tuple(row[k] for k in ('gpu', 'torch', 'triton', 'cuda_build', 'weight_layout'))
                if signature is not None and sig != signature:
                    raise ValueError(f'environment mismatch: {path}')
                signature = sig
                samples = row['samples_ms']
                if len(samples) != design['repeats'] or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError(f'invalid samples: {path}')
                if not math.isclose(statistics.median(samples), row['median_ms'], rel_tol=1e-9):
                    raise ValueError(f'bad median: {path}')
                arms[arm] = {'median_ms': row['median_ms'], 'path': str(path),
                             'sha256': hashlib.sha256(data).hexdigest()}
            blocks.append({'block': block, 'order': order, 'arms': arms,
                           'dx_vs_async_tp': arms['async_tp']['median_ms'] / arms['dx']['median_ms'],
                           'dx_vs_stock': arms['stock']['median_ms'] / arms['dx']['median_ms'],
                           'async_tp_vs_stock': arms['stock']['median_ms'] / arms['async_tp']['median_ms']})
        gm = lambda field: math.exp(statistics.mean(math.log(b[field]) for b in blocks))
        groups.append({'target': target, 'blocks': blocks, 'dx_vs_async_tp_gmean': gm('dx_vs_async_tp'),
                       'dx_vs_stock_gmean': gm('dx_vs_stock'), 'async_tp_vs_stock_gmean': gm('async_tp_vs_stock')})
    result = {'manifest_sha256': digest, 'groups': groups,
              'warning': 'Direct PyTorch fused symm_mem ops with test mode disabled; '
                         'not proof of Inductor graph rewrite or model E2E. Independent processes; '
                         'setup, planning and weight layout conversion excluded.'}
    with (directory / 'tp2_async_comparison_20261113.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    for group in groups:
        print(group['target']['op'], 'DX/Async', group['dx_vs_async_tp_gmean'],
              'DX/stock', group['dx_vs_stock_gmean'], 'Async/stock', group['async_tp_vs_stock_gmean'])


if __name__ == '__main__':
    main()
