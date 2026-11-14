"""Validate provenance and summarize FlashOverlap vs DX plus RMSNorm."""
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    data_dir = Path('/root/data/stowaway-nvidia/results')
    config_dir = Path('/root/data/stowaway-nvidia/flashoverlap_tp2_work_20261112/configs')
    groups = []
    for m in (1024, 2048):
        config = (config_dir / f'm{m}n4096k4096_a800.json').read_bytes()
        paths = [data_dir / f'tp2_flash_equivalent_norm_m{m}_20261113.json']
        paths += [data_dir / f'tp2_flash_equivalent_norm_m{m}_b{b}_20261113.json' for b in (2, 3)]
        blocks = []
        signature = None
        for b, path in enumerate(paths, 1):
            raw = path.read_bytes()
            record = json.loads(raw)
            if record['target'] != {'M': m, 'N': 4096, 'K': 4096}:
                raise ValueError(f'wrong shape: {path}')
            if record['config_sha256'] != hashlib.sha256(config).hexdigest():
                raise ValueError(f'wrong config: {path}')
            sig = tuple(record[x] for x in ('tp', 'dtype', 'warmup', 'batch', 'repeats',
                                            'gpu', 'torch', 'triton', 'cuda_build', 'checks'))
            if signature is not None and sig != signature:
                raise ValueError(f'mismatched protocol: {path}')
            signature = sig
            arms = record['results']
            for name in ('flash_plus_reorder_norm', 'dx_interleaved_plus_norm'):
                samples = arms[name]['samples_ms']
                if len(samples) != record['repeats'] or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError(f'invalid samples: {path}')
                if not math.isclose(statistics.median(samples), arms[name]['median_ms'], rel_tol=1e-9):
                    raise ValueError(f'bad median: {path}')
            flash = arms['flash_plus_reorder_norm']['median_ms']
            dx = arms['dx_interleaved_plus_norm']['median_ms']
            order = record.get('order', ['flash_plus_reorder_norm', 'dx_interleaved_plus_norm'])
            blocks.append({'block': b, 'order': order, 'flash_ms': flash, 'dx_ms': dx,
                           'dx_speedup': flash / dx, 'path': str(path),
                           'sha256': hashlib.sha256(raw).hexdigest()})
        groups.append({'target': {'M': m, 'N': 4096, 'K': 4096}, 'blocks': blocks,
                       'dx_speedup_gmean': math.exp(statistics.mean(math.log(b['dx_speedup']) for b in blocks))})
    result = {'groups': groups, 'warning': 'Synthetic row-parallel RS plus unit-weight RMSNorm. '
              'FlashOverlap uses its own reorder+RMSNorm; DX directly computes interleaved rows and uses '
              'a separate Triton RMSNorm. Both compare against the same rank-interleaved reference. '
              'No proof of packed-tile compatibility, model integration, TP8 or E2E.'}
    with (data_dir / 'tp2_flash_equivalent_norm_summary_20261113.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    for group in groups:
        print(group['target'], group['dx_speedup_gmean'])


if __name__ == '__main__':
    main()
