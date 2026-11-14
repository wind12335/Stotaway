"""Fit independent TP8 route models and export an executable workload plan.

Each model predicts log(stock / candidate), using its own measured stock.
The old DX and Async data were collected in different batches; therefore the
relative estimates are hypotheses to be tested in the model runtime.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .orthogonal_predictor import Route, Sample
from .route_local_residual import RouteLocalResidual
from .route_policy import GroupedSample
from .qwen36_shapes import build_manifest

ROOT = Path('/root/data/stowaway-nvidia/results')
DX = Path(__file__).parent / 'data/cells.json'


def route(method, op):
    dx = method == 'dx'
    td = method == 'triton_dist'
    return Route(method, 'triton' if dx or td else 'pytorch_gemm',
                 'nvshmem' if dx or td else 'pytorch_symm_mem',
                 'symmetric_memory', 'stream', op,
                 'direct_exchange8' if dx else 'triton_dist_tp8' if td else 'torch.ops.symm_mem',
                 'A800-SXM4-80GB', 'NV8', 8, 'bfloat16',
                 (('cuda', '12.6'), ('torch', '2.7.0+cu126'), ('triton', '3.4.0')),
                 'b2b16', None if dx else 4, 'dx_existing' if dx else 'default_no_autotune' if td else 'pytorch_direct')


def generate(workload, backend='async_tp'):
    paths = sorted(ROOT.glob('tp8_async_matrix_m*20261113.jsonl' if backend == 'async_tp' else 'tp8_td_*20261113.jsonl'))
    data = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if line:
                r = json.loads(line)
                data[tuple(r['identity'])] = r
    external_rows = list(data.values())
    dx_rows = json.loads(DX.read_text())
    predictors = {}
    for method, rows in [(backend, external_rows), ('dx', dx_rows)]:
        for op in ('ag', 'rs'):
            rt = route(method, op)
            group = [r for r in rows if ('ag' if r['op'].startswith('ag') else 'rs') == op]
            samples = []
            for r in group:
                if method == 'dx':
                    stock, candidate = r['stock_b2b'], r['dx_b2b']
                else:
                    stock = r['measurements']['stock']['median_ms']
                    candidate = r['measurements'][backend]['median_ms']
                samples.append(GroupedSample(r['model'], Sample(rt, (r['M'], r['N'], r['K']), stock, candidate)))
            predictors[(method, op)] = RouteLocalResidual(rt).fit(samples)
    if workload == 'qwen36_train':
        config = json.loads(Path('/root/eb-public/huggingface-models/Qwen/Qwen3.6-27B/config.json').read_text())
        sites = build_manifest(config, 8192)['sites']
    elif workload == 'llama70_infer':
        sites = [dict(adapter_name=site, op=op, M=16384, N=n, K=k)
                 for site, op, n, k in [('qkv', 'ag', 1280, 8192), ('proj', 'rs', 8192, 1024),
                                        ('fc1', 'ag', 7168, 8192), ('fc2', 'rs', 8192, 3584)]]
    else:
        raise ValueError('unknown workload')
    entries = []
    for site in sites:
        op = 'ag' if site['op'].startswith('ag') else 'rs'
        shape = (site['M'], site['N'], site['K'])
        gains = {method: predictors[(method, op)].predict(route(method, op), shape)['predicted_log_speedup']
                 for method in (backend, 'dx')}
        scores = {'stock': 0., **gains}
        entries.append({'site': site['adapter_name'], 'op': op, 'M': shape[0], 'N': shape[1],
                        'K': shape[2], 'choice': max(scores, key=scores.get), 'predicted_log_speedup': scores})
    return {'schema': 'stowaway-tp8-auto-plan-v1', 'workload': workload, 'tp': 8, 'dtype': 'bfloat16',
            'external_backend': backend, 'default': 'async_tp' if backend == 'async_tp' else 'stock', 'entries': entries,
            'provenance': {'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                       for p in [*paths, DX]],
                           'method': 'independent route-local residual models with per-route stock normalization',
                           'limitation': 'DX and Async measurements are from separate batches; operator estimates require E2E validation',
                           'routes': [asdict(p.route) for p in predictors.values()]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workload', required=True, choices=('qwen36_train', 'llama70_infer'))
    p.add_argument('--backend', default='async_tp', choices=('async_tp', 'triton_dist'))
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = generate(args.workload, args.backend)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result['entries'], indent=2))


if __name__ == '__main__':
    main()
