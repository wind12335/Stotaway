"""Export fixed external-backend FP16 plans from measured TP8 route ratios.

DX calibration remains BF16 from the existing inventory; plan provenance
labels it a cross-dtype estimate until paired FP16 measurements are available.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .export_tp8_auto_plan import DX, ROOT, route
from .orthogonal_predictor import Route, Sample
from .qwen36_shapes import build_manifest
from .route_local_residual import RouteLocalResidual
from .route_policy import GroupedSample


def read_sync():
    rows = []
    for f in (ROOT / 'tp8_sync_matrix_20261113').glob('*_stock.json'):
        g = f.with_name(f.name.replace('_stock.json', '_syncopate.json'))
        if not g.exists():
            continue
        a, b = json.loads(f.read_text()), json.loads(g.read_text())
        r = a['target']
        rows.append({'model': f.name.split('_')[0], 'op': r['op'], 'M': r['M'], 'N': r['N'], 'K': r['K'],
                     'stock': a['median_ms'], 'candidate': b['median_ms']})
    return rows


def read_flash():
    rows = []
    files = list((ROOT / 'tp8_flash_matrix_20261113').glob('*.json')) + [ROOT / 'tp8_flash_b16_20261113.json']
    for f in files:
        if not f.exists() or f.name.endswith('_check.json'):
            continue
        try:
            r = json.loads(f.read_text())
            rows.append({'model': f.name.split('_')[0], 'op': 'rs', 'M': r['M'], 'N': r['N'], 'K': r['K'],
                         'stock': r['measurements']['stock']['median_ms'], 'candidate': r['measurements']['overlap']['median_ms']})
        except (KeyError, ValueError):
            continue
    return rows


def make_route(method, op):
    return Route(method, 'cutlass' if method == 'flashoverlap' else 'triton',
                 'nccl' if method == 'flashoverlap' else 'nvshmem', 'symmetric_memory', 'stream', op,
                 'flashoverlap_rs' if method == 'flashoverlap' else 'syncopate_generated',
                 'A800-SXM4-80GB', 'NV8', 8, 'float16',
                 (('cuda', '12.6'), ('torch', '2.7.0+cu126'), ('triton', '3.4.0')),
                 'b2b16', 4, 'measured_tp8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('syncopate', 'flashoverlap'), required=True)
    ap.add_argument('--workload', choices=('qwen36_train', 'llama70_infer'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    ext = read_sync() if args.backend == 'syncopate' else read_flash()
    dx = json.loads(DX.read_text())
    models = {}
    for op in ('ag', 'rs'):
        for method in (args.backend, 'dx'):
            rt = make_route(method, op) if method != 'dx' else route('dx', op)
            rows = []
            data = ext if method != 'dx' else dx
            for r in data:
                if ('ag' if r['op'].startswith('ag') else 'rs') != op:
                    continue
                stock, cand = (r['stock'], r['candidate']) if method != 'dx' else (r['stock_b2b'], r['dx_b2b'])
                rows.append(GroupedSample(r['model'], Sample(rt, (r['M'], r['N'], r['K']), stock, cand)))
            if len({r.family for r in rows}) >= 4:
                models[(method, op)] = (rt, RouteLocalResidual(rt).fit(rows))
    if args.workload == 'qwen36_train':
        path = Path('/root/eb-public/huggingface-models/Qwen/Qwen3.6-27B/config.json')
        sites = build_manifest(json.loads(path.read_text()), 8192)['sites']
    else:
        sites = [dict(adapter_name=site, op=op, M=16384, N=n, K=k)
                 for site, op, n, k in [('qkv', 'ag', 1280, 8192), ('proj', 'rs', 8192, 1024),
                                        ('fc1', 'ag', 7168, 8192), ('fc2', 'rs', 8192, 3584)]]
    entries, deferred = [], []
    for s in sites:
        op = 'ag' if s['op'].startswith('ag') else 'rs'
        shape = (s['M'], s['N'], s['K'])
        scores = {'stock': 0.0}
        for method in (args.backend,):
            if (method, op) in models:
                rt, model = models[(method, op)]
                scores[method] = model.predict(rt, shape)['predicted_log_speedup']
        choice = max(scores, key=scores.get)
        if args.backend == 'flashoverlap' and choice == 'flashoverlap':
            config = Path('/root/data/stowaway-nvidia/flashoverlap_tp8_work_20261113/configs') / f'm{shape[0]}n{shape[1]}k{shape[2]}_a800.json'
            valid = False
            if config.exists():
                try:
                    cfg = json.loads(config.read_text())
                    valid = (isinstance(cfg.get('BM'), int) and isinstance(cfg.get('BN'), int)
                             and isinstance(cfg.get('hint'), list) and isinstance(cfg.get('cSeg'), list))
                except (OSError, ValueError):
                    pass
            if not valid:
                deferred.append({'site': s['adapter_name'], 'shape': shape, 'reason': 'missing completed tuning configuration'})
                choice = 'stock'
        entries.append({'site': s['adapter_name'], 'op': op, 'M': shape[0], 'N': shape[1], 'K': shape[2],
                        'choice': choice, 'predicted_log_speedup': scores})
    payload = {'schema': 'stowaway-tp8-auto-plan-v1', 'workload': args.workload, 'tp': 8, 'dtype': 'float16',
               'external_backend': args.backend, 'default': 'stock', 'entries': entries,
               'provenance': {'method': 'route-local residual', 'external_points': len(ext), 'deferred': deferred,
                 'dx_source_sha256': hashlib.sha256(DX.read_bytes()).hexdigest(),
                 'limitation': 'DX selection disabled: available DX data are BF16, external data FP16; output remains provisional until same-dtype measurements'}}
    with args.output.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(entries, indent=2))


if __name__ == '__main__':
    main()
