"""Select a gain margin inside grouped folds; evaluate on held-out physical families."""
import argparse
import json
import math
import statistics
from pathlib import Path

from .evaluate_fragmentation_mixture import join
from .reduced_probe_policy import ReducedProbePolicy


def prediction(model, row):
    return model.predict({k: v for k, v in row.items() if k not in ('comm_ms', 'stock_ms', 'overlap_ms')})['predicted_log_speedup']


def inner_scores(rows, op):
    out = []
    for family in sorted({r['family'] for r in rows}):
        fit = [r for r in rows if r['family'] != family]
        model = ReducedProbePolicy().fit(fit, op=op)
        for row in rows:
            if row['family'] == family:
                out.append((prediction(model, row), math.log(row['stock_ms'] / row['overlap_ms'])))
    return out


def select_margin(pairs, margins=(0., .02, .05, .08)):
    # Explicitly trade selection regret for risk of choosing a slower method.
    # Objective and candidate margins are fixed before the outer fold.
    def loss(margin):
        regrets = [max(0., actual) - (actual if predicted > margin else 0.) for predicted, actual in pairs]
        slower = sum(predicted > margin and actual < 0 for predicted, actual in pairs)
        return statistics.mean(regrets) + .02 * slower / len(pairs)
    return min(margins, key=lambda margin: (loss(margin), margin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operators', nargs='+', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, sig, operators, stages = join(args.operators, args.stages)
    out = []
    for op in sorted({r['op'] for r in data}):
        group = [r for r in data if r['op'] == op]
        for held in sorted({r['family'] for r in group}):
            train = [r for r in group if r['family'] != held]
            margin = select_margin(inner_scores(train, op))
            model = ReducedProbePolicy().fit(train, op=op)
            for row in group:
                if row['family'] != held:
                    continue
                pred = prediction(model, row)
                gain = math.log(row['stock_ms'] / row['overlap_ms'])
                out.append({'identity': row['identity'], 'op': op, 'family': held, 'margin': margin,
                            'predicted_log_speedup': pred, 'actual_log_speedup': gain})
    def summarize(mode):
        choices = [r['predicted_log_speedup'] > (r['margin'] if mode == 'nested' else 0.) for r in out]
        gains = [r['actual_log_speedup'] if c else 0. for r, c in zip(out, choices)]
        return {'count': len(out), 'correct': sum(c == (r['actual_log_speedup'] > 0) for r, c in zip(out, choices)),
                'slower': sum(g < 0 for g in gains), 'speedup_gmean': math.exp(statistics.mean(gains))}
    result = {'classification': 'development data nested leave-physical-family-out margin evaluation',
              'warning': 'objective fixed before outer folds; data already inspected; no probe costs or E2E',
              'signature': sig, 'operator_sources': operators, 'stage_sources': stages,
              'summary': {name: summarize(name) for name in ('nested', 'zero_margin')}, 'rows': out}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
