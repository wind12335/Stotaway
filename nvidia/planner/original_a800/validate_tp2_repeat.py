"""Evaluate fixed TP2 policies on a separately collected repeat batch.

The training batch excludes all records of the test (direction,K,N) family,
including other M and aliases. The repeated measurement supplies its own
single-GPU GEMM input, never its stock/DX labels to the policy. Source files
are not combined or overwritten. Exploratory: measurement protocol and data
have already informed development, so this is not a final blind test.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from relative_model import RelativeModel
from validate_tp2_new import load
from tp2_nested_policy import select_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='+', type=Path, required=True)
    parser.add_argument('--test', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train, train_signature, train_sources = load(args.train)
    test, test_signature, test_sources = load(args.test)
    if train_signature != test_signature:
        raise ValueError('train/test runtime signature mismatch')
    results = []
    for cell in test:
        available = [r for r in train if r['model'] != cell['model']]
        per_op = [r for r in available if r['op'] == cell['op']]
        old = RelativeModel(mode='b2b').fit(available)
        try:
            old_pick, prediction = old.choose(cell)
        except ValueError:
            old_pick, prediction = 'stock', None
        policy, threshold, audit = select_policy(per_op, 2.0)
        pred = (RelativeModel._predict_op(RelativeModel(mode='b2b')._fit_op(per_op), cell)
                if policy == 'threshold' else None)
        new_pick = 'dx' if policy == 'dx' or (policy == 'threshold' and pred > threshold) else 'stock'
        results.append({**cell, 'oracle': 'dx' if cell['dx_b2b'] < cell['stock_b2b'] else 'stock',
                        'relative_pick': old_pick, 'relative_prediction': prediction,
                        'nested_pick': new_pick, 'nested_policy': policy,
                        'nested_threshold': threshold, 'nested_prediction': pred, 'training_audit': audit})
    def summary(pick_name):
        speeds = [r['stock_b2b'] / (r['dx_b2b'] if r[pick_name] == 'dx' else r['stock_b2b']) for r in results]
        return {'count': len(results), 'correct': sum(r[pick_name] == r['oracle'] for r in results),
                'speedup_gmean': math.exp(statistics.mean(map(math.log, speeds))),
                'slower': sum(x < 1 for x in speeds)}
    for r in results:
        r['always_dx'] = 'dx'
        r['always_stock'] = 'stock'
    report = {'classification': 'exploratory independent repeat; leave full direction/K/N family out',
              'signature': train_signature, 'train_sources': train_sources, 'test_sources': test_sources,
              'summary': {name: summary(name) for name in ('relative_pick', 'nested_pick', 'always_dx', 'always_stock')},
              'rows': results}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
