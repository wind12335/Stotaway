"""CPU-only grouped-shape validation of one small relative-cost baseline.

Uses only already collected four-GPU operator records. Independently fits
each method/direction; repeated rows are aggregated before splitting. Stock
latency is an input, not a free measurement. The relative model/guard are
exploratory and must not be exported to another GPU, TP or dtype.
"""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def features(shape):
    m, n, k = shape
    # Independent coordinates: work and traffic ratios are linear combinations
    # of these logarithms and are not additional evidence in this small sample.
    return np.log(np.array([m, n, k], dtype=float))


def fit(train):
    x = np.stack([features(row['shape']) for row in train])
    y = np.array([math.log(row['overlap_ms'] / row['stock_ms']) for row in train])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (x - mean) / scale
    # Unpenalized intercept, ridge regularization on the centered dimensions.
    coef = np.linalg.solve(z.T @ z + np.eye(3) * 2.0, z.T @ (y - y.mean()))
    return lambda shape: float(y.mean() + ((features(shape) - mean) / scale) @ coef)


def rows_for(data, method, op):
    by_shape = defaultdict(list)
    for row in data['rows']:
        if row['method'] != method or row['op'] != op:
            continue
        shape = (row['M'], row['N'], row['K'])
        if method == 'flux':
            a, b = row['stock_max_rank_ms'], row['overlap_max_rank_ms']
        else:
            a, b = row['stock_ms'], row['overlap_ms']
        by_shape[shape].append((a, b))
    return [{'shape': shape, 'stock_ms': statistics.median(a for a, _ in vals),
             'overlap_ms': statistics.median(b for _, b in vals), 'records': len(vals)}
            for shape, vals in sorted(by_shape.items())]


def evaluate(rows):
    # Nested grouped leave-one-out guard needs >= 3 training shapes.
    if len(rows) < 4:
        return {'status': 'insufficient distinct shapes for nested validation', 'shapes': len(rows)}
    out = []
    for i, held in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        residuals = []
        for j, val in enumerate(train):
            pred = fit([r for k, r in enumerate(train) if k != j])(val['shape'])
            residuals.append(math.log(val['overlap_ms'] / val['stock_ms']) - pred)
        # Max observed positive residual: conservative empirical heuristic,
        # not a confidence bound or a guarantee for an unseen workload.
        guard = max(0.0, max(residuals))
        prediction = fit(train)(held['shape'])
        pick = 'overlap' if prediction + guard < 0 else 'stock'
        oracle = 'overlap' if held['overlap_ms'] < held['stock_ms'] else 'stock'
        chosen = held['overlap_ms'] if pick == 'overlap' else held['stock_ms']
        out.append({**held, 'pred_log_ratio': prediction, 'guard_log_ratio': guard,
                    'pick': pick, 'oracle': oracle, 'speedup': held['stock_ms'] / chosen})
    baseline = {}
    for choice in ('stock', 'overlap', 'oracle'):
        chosen = [r['stock_ms'] if choice == 'stock' else r['overlap_ms'] if choice == 'overlap'
                  else min(r['stock_ms'], r['overlap_ms']) for r in out]
        baseline[choice] = {'correct': len(out) if choice == 'oracle' else
                            sum(r['oracle'] == choice for r in out),
                            'speedup_gmean': math.exp(sum(math.log(r['stock_ms']/t)
                                                           for r, t in zip(out, chosen)) / len(out)),
                            'slower_than_stock': sum(t > r['stock_ms'] for r, t in zip(out, chosen))}
    return {'status': 'exploratory; tiny non-independent development dataset', 'shapes': len(rows),
            'correct': sum(r['pick'] == r['oracle'] for r in out),
            'speedup_gmean': math.exp(sum(math.log(r['speedup']) for r in out) / len(out)),
            'slower_than_stock': sum(r['speedup'] < 1 for r in out), 'baselines': baseline, 'rows': out}


def main():
    source = HERE / 'data/reference_pdf_records.json'
    raw = source.read_bytes()
    data = json.loads(raw)
    groups = [(method, op) for method in ('flux', 'flashoverlap') for op in ('ag', 'rs', 'ar')
              if any(r['method'] == method and r['op'] == op for r in data['rows'])]
    result = {'source_sha256': hashlib.sha256(raw).hexdigest(),
              'warning': 'four GPUs; GPU/topology unknown; FP16; Flux AG iters=1; stock latency is measured input',
              'split': 'leave one entire shape out; nested training-only residual guard; method/direction separate',
              'results': {f'{method}_{op}': evaluate(rows_for(data, method, op)) for method, op in groups}}
    dest = HERE / 'data/family_records_exploratory.json'
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    for group, res in result['results'].items():
        print(group, {k: v for k, v in res.items() if k != 'rows'})
    print('saved', dest)


if __name__ == '__main__':
    main()
