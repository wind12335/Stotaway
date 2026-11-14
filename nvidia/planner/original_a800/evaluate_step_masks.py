"""Leave-one-block-out diagnostic on the existing Qwen paircal report."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .step_mask_calibration import StepBlock, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    context = 'qwen36-local4-random-mock-training-tp2-m2048-conn1-dx-sites'
    blocks = [StepBlock(context, str(i), b['ms']) for i, b in enumerate(data['blocks'])]
    masks = {'stock', 'proj', 'fc2', 'pair'}
    if len(blocks) < 3:
        raise ValueError('at least three blocks required for leave-one-out')
    # Fixed before evaluating this dataset; neither setting is a tuned threshold.
    policies = {'median_delta': 0.0, 'median_plus_mad': 1.0}
    results = {}
    for name, penalty in policies.items():
        folds = []
        for test in blocks:
            decision = select((b for b in blocks if b.block_id != test.block_id), context=context,
                              allowed_masks=masks, mad_penalty=penalty, min_blocks=2)
            choice = decision['choice']
            oracle = min(test.milliseconds, key=lambda m: (test.milliseconds[m], m != 'stock', m))
            folds.append({'test_block': test.block_id, 'training': decision, 'choice': choice,
                          'oracle': oracle, 'speedup': test.milliseconds['stock']/test.milliseconds[choice]})
        results[name] = {'correct': sum(f['choice'] == f['oracle'] for f in folds),
                         'slower': sum(f['speedup'] < 1 for f in folds),
                         'geomean_speedup': math.exp(statistics.mean(math.log(f['speedup']) for f in folds)),
                         'folds': folds}
    baselines = {mask: {'geomean_speedup': math.exp(statistics.mean(math.log(
        b.milliseconds['stock']/b.milliseconds[mask]) for b in blocks)),
        'slower': sum(b.milliseconds[mask] > b.milliseconds['stock'] for b in blocks)} for mask in sorted(masks)}
    out = {'classification': 'exploratory leave-one-block-out; only three blocks',
           'warning': 'same data inspected during development, no independent confirmation; no guarantee',
           'source': str(args.input), 'source_sha256': hashlib.sha256(raw).hexdigest(),
           'context': context, 'policies': results, 'baselines': baselines}
    with args.output.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'policies': {k: {x: v[x] for x in ('correct', 'slower', 'geomean_speedup')}
                                    for k, v in results.items()}, 'baselines': baselines}, indent=2))


if __name__ == '__main__':
    main()
