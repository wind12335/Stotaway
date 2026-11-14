"""Leave one model out for stock/Async-TP/DX, using per-batch stock ratios."""
import hashlib
import json
import math
import statistics
from pathlib import Path

from .export_tp8_auto_plan import DX, ROOT, route
from .orthogonal_predictor import Sample
from .route_local_residual import RouteLocalResidual
from .route_policy import GroupedSample


def identity(r):
    return (r['model'], r['op'], r['M'], r['N'], r['K'])


def main():
    paths = sorted(ROOT.glob('tp8_async_matrix_m*20261113.jsonl'))
    async_rows = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]
    dx_rows = json.loads(DX.read_text())
    by_method = {'async_tp': {identity(r): r for r in async_rows}, 'dx': {identity(r): r for r in dx_rows}}
    keys = sorted(set(by_method['async_tp']) & set(by_method['dx']))
    if len(keys) != 99:
        raise ValueError(f'expected 99 aligned shapes, got {len(keys)}')
    records = []
    for op in ('ag', 'rs'):
        selected = [key for key in keys if ('ag' if key[1].startswith('ag') else 'rs') == op]
        for held in sorted({key[0] for key in selected}):
            predictors = {}
            for method in ('async_tp', 'dx'):
                rt = route(method, op)
                samples = []
                for key in selected:
                    if key[0] == held:
                        continue
                    r = by_method[method][key]
                    if method == 'dx':
                        stock, candidate = r['stock_b2b'], r['dx_b2b']
                    else:
                        stock = r['measurements']['stock']['median_ms']
                        candidate = r['measurements']['async_tp']['median_ms']
                    samples.append(GroupedSample(key[0], Sample(rt, (key[2], key[3], key[4]), stock, candidate)))
                predictors[method] = RouteLocalResidual(rt).fit(samples)
            for key in selected:
                if key[0] != held:
                    continue
                predicted = {'stock': 0.0}
                actual = {'stock': 1.0}
                for method in ('async_tp', 'dx'):
                    predicted[method] = predictors[method].predict(route(method, op), key[2:])['predicted_log_speedup']
                    r = by_method[method][key]
                    actual[method] = (r['stock_b2b'] / r['dx_b2b'] if method == 'dx' else
                                      r['measurements']['stock']['median_ms'] / r['measurements']['async_tp']['median_ms'])
                choice = max(predicted, key=predicted.get)
                best = max(actual, key=actual.get)
                records.append({'identity': key, 'predicted_log_speedup': predicted, 'actual_normalized_speedup': actual,
                                'choice': choice, 'oracle_choice': best, 'selected_speedup': actual[choice]})
    summaries = {}
    for op in ('ag', 'rs', 'all'):
        items = [r for r in records if op == 'all' or ('ag' if r['identity'][1].startswith('ag') else 'rs') == op]
        summaries[op] = {'count': len(items), 'correct': sum(r['choice'] == r['oracle_choice'] for r in items),
                         'selected_gmean_speedup': math.exp(statistics.mean(math.log(r['selected_speedup']) for r in items)),
                         'selected_below_stock': sum(r['selected_speedup'] < 1 for r in items)}
        summaries[op]['accuracy'] = summaries[op]['correct'] / len(items)
    out = {'classification': 'development leave-one-model-out, normalized independently per backend',
           'limitation': 'DX and Async measured in different batches; normalized oracle is approximate, not paired latency ground truth',
           'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [*paths, DX]],
           'summary': summaries, 'rows': records}
    dest = ROOT / 'tp8_auto_threeway_lomo_20261113.json'
    with dest.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
