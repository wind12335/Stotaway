"""Exploratory nested grouped policy selection for a fixed overlap method.

The outer evaluation never sees its held-out shape family's labels. Inside the
training fold, compare fixed stock, fixed DX, and relative-model thresholds
using grouped OOF predictions. Utility is log speedup less a specified penalty
for negative log speedup. No claim of safety or statistical confidence.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from .relative_model import RelativeModel
    from .validate_tp2_new import load
except ImportError:
    from relative_model import RelativeModel
    from validate_tp2_new import load


THRESHOLDS = (-0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12)


def select_policy(training, penalty):
    families = sorted({r['model'] for r in training})
    if len(families) < 3:
        return 'stock', None, {'reason': 'insufficient training families'}
    predictions = []
    for family in families:
        fit_rows = [r for r in training if r['model'] != family]
        params = RelativeModel(mode='b2b')._fit_op(fit_rows)
        for row in training:
            if row['model'] == family:
                predictions.append((RelativeModel._predict_op(params, row),
                                    math.log(row['stock_b2b']/row['dx_b2b'])))
    choices = [('stock', None), ('dx', None)] + [('threshold', t) for t in THRESHOLDS]
    def score(policy):
        kind, threshold = policy
        gains = [truth if kind == 'dx' or (kind == 'threshold' and pred > threshold) else 0.0
                 for pred, truth in predictions]
        return statistics.mean(g - penalty * max(0.0, -g) for g in gains)
    best = max(choices, key=lambda p: (score(p), p[0] == 'stock'))
    return *best, {'training_families': len(families), 'training_oof_utility': score(best),
                   'penalty': penalty}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--penalty', type=float, default=2.0)
    args = ap.parse_args()
    if not math.isfinite(args.penalty) or args.penalty < 0:
        raise ValueError('penalty must be nonnegative')
    cells, signature, sources = load(args.inputs)
    result = []
    for op in sorted({r['op'] for r in cells}):
        rows = [r for r in cells if r['op'] == op]
        for family in sorted({r['model'] for r in rows}):
            train = [r for r in rows if r['model'] != family]
            policy, threshold, audit = select_policy(train, args.penalty)
            params = RelativeModel(mode='b2b')._fit_op(train) if policy == 'threshold' else None
            for row in (r for r in rows if r['model'] == family):
                pred = RelativeModel._predict_op(params, row) if params is not None else None
                pick = 'dx' if policy == 'dx' or (policy == 'threshold' and pred > threshold) else 'stock'
                actual = row['stock_b2b']/row['dx_b2b']
                result.append({**row, 'policy': policy, 'threshold': threshold, 'pred_log_speedup': pred,
                               'audit': audit, 'choice': pick,
                               'oracle': 'dx' if actual > 1 else 'stock',
                               'speedup': actual if pick == 'dx' else 1.0})
    def summ(rows):
        return {'count': len(rows), 'correct': sum(r['choice'] == r['oracle'] for r in rows),
                'speedup_gmean': math.exp(statistics.mean(math.log(r['speedup']) for r in rows)),
                'slower': sum(r['speedup'] < 1 for r in rows)}
    out = {'method': 'nested leave-(direction,K,N)-out within each op; exploratory',
           'signature': signature, 'sources': sources, 'penalty': args.penalty,
           'summary': summ(result), 'by_op': {op: summ([r for r in result if r['op'] == op])
                                             for op in sorted({r['op'] for r in result})}, 'rows': result}
    with args.output.open('x') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps({k: out[k] for k in ('method','penalty','summary','by_op')}, indent=2))


if __name__ == '__main__':
    main()
