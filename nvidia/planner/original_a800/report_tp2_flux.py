"""Audit and summarize the exploratory TP2 Flux comparison."""
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = root / 'experiments/tp2_flux_compare_20261113.json'
    raw = manifest.read_bytes()
    design = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path('/root/data/stowaway-nvidia/results')
    groups = []
    for i, target in enumerate(design['targets']):
        blocks = []
        signature = None
        for block, planned in enumerate(design['orders'], 1):
            actual = ['flux', 'stock', 'dx'] if i == 0 and block == 1 else planned
            arms = {}
            for arm in actual:
                path = directory / f'tp2_flux_compare_t{i}_{arm}_b{block}_20261113.json'
                data = path.read_bytes()
                row = json.loads(data)
                if row['manifest_sha256'] != digest or row['target'] != target or row['arm'] != arm:
                    raise ValueError(f'provenance mismatch: {path}')
                for key in ('tp', 'dtype', 'batch', 'warmup', 'repeats'):
                    if row[key] != design[key]:
                        raise ValueError(f'{key} mismatch: {path}')
                if row['checks'] != 2 or row['cuda_device_max_connections'] != '1':
                    raise ValueError(f'protocol mismatch: {path}')
                sig = tuple(row[k] for k in ('gpu', 'torch', 'triton', 'cuda_build'))
                if signature is not None and sig != signature:
                    raise ValueError(f'software or GPU mismatch: {path}')
                signature = sig
                samples = row['samples_ms']
                if len(samples) != 8 or any(not math.isfinite(t) or t <= 0 for t in samples):
                    raise ValueError(f'invalid samples: {path}')
                if not math.isclose(statistics.median(samples), row['median_ms'], rel_tol=1e-9):
                    raise ValueError(f'invalid median: {path}')
                arms[arm] = {'median_ms': row['median_ms'], 'path': str(path),
                             'sha256': hashlib.sha256(data).hexdigest()}
            blocks.append({'block': block, 'planned_order': planned, 'actual_order': actual, 'arms': arms,
                           'dx_vs_flux': arms['flux']['median_ms'] / arms['dx']['median_ms'],
                           'dx_vs_stock': arms['stock']['median_ms'] / arms['dx']['median_ms'],
                           'flux_vs_stock': arms['stock']['median_ms'] / arms['flux']['median_ms']})
        gm = lambda field: math.exp(statistics.mean(math.log(b[field]) for b in blocks))
        groups.append({'target': target, 'blocks': blocks,
                       'dx_vs_flux_gmean': gm('dx_vs_flux'), 'dx_vs_stock_gmean': gm('dx_vs_stock'),
                       'flux_vs_stock_gmean': gm('flux_vs_stock')})
    result = {'manifest_sha256': digest, 'groups': groups,
              'warning': 'First AG block ran flux-stock-dx, differing from manifest planned order. '
                         'Same shape was first used as a smoke test. Shared NCCL, Flux CUDA IPC build, '
                         'DX RS scratch, independent processes. Excludes plan and E2E overhead.'}
    out = directory / 'tp2_flux_compare_summary_20261113.json'
    with out.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    for g in groups:
        print(g['target']['op'], 'DX/Flux', g['dx_vs_flux_gmean'], 'DX/stock', g['dx_vs_stock_gmean'],
              'Flux/stock', g['flux_vs_stock_gmean'])


if __name__ == '__main__':
    main()
