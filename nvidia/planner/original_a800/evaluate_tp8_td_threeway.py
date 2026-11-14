"""LOMO selection for stock / Triton-distributed / DX on TP8 BF16 data."""
import hashlib
import json
import math
import statistics

from .export_tp8_auto_plan import DX, ROOT, route
from .orthogonal_predictor import Route, Sample
from .route_local_residual import RouteLocalResidual
from .route_policy import GroupedSample


def td_route(op):
    return Route('triton_dist', 'triton', 'nvshmem', 'symmetric_memory', 'stream', op,
                 'triton_dist_tp8', 'A800-SXM4-80GB', 'NV8', 8, 'bfloat16',
                 (('cuda', '12.6'), ('torch', '2.7.0+cu126'), ('triton', '3.4.0')),
                 'b2b16', 4, 'default_no_autotune')


def key(r):
    return (r['model'], r['op'], r['M'], r['N'], r['K'])


def main():
    paths = sorted(ROOT.glob('tp8_td_*20261113.jsonl'))
    td = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                td[key(r)] = r
    dx = {key(r): r for r in json.loads(DX.read_text())}
    keys = sorted(set(td) & set(dx))
    if len(keys) != 99:
        raise ValueError(f'expected 99 aligned shapes; got {len(keys)}')
    records = []
    for op in ('ag', 'rs'):
        selected = [k for k in keys if ('ag' if k[1].startswith('ag') else 'rs') == op]
        for held in sorted({k[0] for k in selected}):
            predictors = {}
            for method in ('triton_dist', 'dx'):
                rt = td_route(op) if method == 'triton_dist' else route('dx', op)
                rows = []
                for k in selected:
                    if k[0] == held:
                        continue
                    r = td[k] if method == 'triton_dist' else dx[k]
                    stock, cand = ((r['measurements']['stock']['median_ms'], r['measurements']['triton_dist']['median_ms'])
                                   if method == 'triton_dist' else (r['stock_b2b'], r['dx_b2b']))
                    rows.append(GroupedSample(k[0], Sample(rt, k[2:], stock, cand)))
                predictors[method] = RouteLocalResidual(rt).fit(rows)
            for k in selected:
                if k[0] != held:
                    continue
                predicted = {'stock': 0.0}
                actual = {'stock': 1.0}
                for method in ('triton_dist', 'dx'):
                    rt = td_route(op) if method == 'triton_dist' else route('dx', op)
                    predicted[method] = predictors[method].predict(rt, k[2:])['predicted_log_speedup']
                    r = td[k] if method == 'triton_dist' else dx[k]
                    actual[method] = (r['measurements']['stock']['median_ms'] / r['measurements']['triton_dist']['median_ms']
                                      if method == 'triton_dist' else r['stock_b2b'] / r['dx_b2b'])
                chosen = max(predicted, key=predicted.get)
                records.append({'identity': k, 'predicted_log_speedup': predicted, 'actual_normalized_speedup': actual,
                                'choice': chosen, 'oracle_choice': max(actual, key=actual.get),
                                'selected_speedup': actual[chosen]})
    summary = {}
    for op in ('ag', 'rs', 'all'):
        data = [r for r in records if op == 'all' or ('ag' if r['identity'][1].startswith('ag') else 'rs') == op]
        correct = sum(r['choice'] == r['oracle_choice'] for r in data)
        summary[op] = {'count': len(data), 'correct': correct, 'accuracy': correct / len(data),
                       'selected_gmean_speedup': math.exp(statistics.mean(math.log(r['selected_speedup']) for r in data)),
                       'selected_below_stock': sum(r['selected_speedup'] < 1 for r in data)}
    payload = {'classification': 'development leave-one-model-out, independently normalized backends',
               'limitation': 'different batches and some CUDA_DEVICE_MAX_CONNECTIONS settings; approximate oracle',
               'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [*paths, DX]],
               'summary': summary, 'rows': records}
    with (ROOT / 'tp8_td_dx_threeway_lomo_20261113.json').open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
