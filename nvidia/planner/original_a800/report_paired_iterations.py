"""Reanalyze paircal logs using matched iteration numbers within each block.

Equal reported losses are a diagnostic prerequisite, not proof that inputs,
GPU clocks or execution state are identical. Preserve all paired deltas.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .step_mask_calibration import StepBlock, select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    raw = args.input.read_bytes()
    source = json.loads(raw)
    runs = source['runs']
    if len(runs) % 4:
        raise ValueError('incomplete blocks')
    context = 'qwen36-local4-random-mock-training-tp2-m2048-conn1-dx-sites'
    blocks, records = [], []
    masks = {'stock', 'proj', 'fc2', 'pair'}
    for index in range(len(runs)//4):
        group = runs[index*4:index*4+4]
        by_mask = {r['arm']: {int(i): (t, loss) for i, t, loss in r['iterations'] if i > 3} for r in group}
        if set(by_mask) != masks or len(group) != len(by_mask):
            raise ValueError('missing or duplicate arm')
        iterations = set(by_mask['stock'])
        if not iterations or any(set(v) != iterations for v in by_mask.values()):
            raise ValueError('iteration numbers differ between arms')
        if any(not math.isclose(by_mask[m][i][1], by_mask['stock'][i][1], rel_tol=1e-6, abs_tol=1e-6)
               for m in masks for i in iterations):
            raise ValueError('reported losses differ across arms')
        ids = sorted(iterations)
        deltas = {m: [by_mask[m][i][0]-by_mask['stock'][i][0] for i in ids] for m in masks}
        interaction = [by_mask['pair'][i][0]-by_mask['proj'][i][0]-by_mask['fc2'][i][0]+
                       by_mask['stock'][i][0] for i in ids]
        # Select() needs positive times; the same stock reference is added to
        # each median paired delta. Only differences are meaningful here.
        ref = statistics.median(t for t, _ in by_mask['stock'].values())
        pseudo = {m: ref + statistics.median(deltas[m]) for m in masks}
        blocks.append(StepBlock(context, str(index), pseudo))
        records.append({'block': index, 'iterations': ids, 'paired_delta_ms': deltas,
                        'median_delta_ms': {m: statistics.median(d) for m, d in deltas.items()},
                        'paired_interaction_ms': interaction,
                        'median_paired_interaction_ms': statistics.median(interaction),
                        'paired_speedup': {m: statistics.median(by_mask['stock'][i][0]/by_mask[m][i][0]
                                                               for i in ids) for m in masks}})
    if len(blocks) < 3:
        raise ValueError('three or more blocks required for leave-one-block-out')
    evaluations = {}
    for name, penalty in [('median_delta', 0.), ('median_plus_mad', 1.)]:
        folds = []
        for test, record in zip(blocks, records):
            selected = select((b for b in blocks if b.block_id != test.block_id), context=context,
                              allowed_masks=masks, mad_penalty=penalty)
            choice = selected['choice']
            oracle = min(masks, key=lambda m: (record['median_delta_ms'][m], m != 'stock', m))
            folds.append({'test_block': test.block_id, 'choice': choice, 'oracle': oracle,
                          'paired_speedup': record['paired_speedup'][choice], 'selection': selected})
        evaluations[name] = {'folds': folds, 'correct': sum(f['choice'] == f['oracle'] for f in folds),
                             'slower': sum(f['paired_speedup'] < 1 for f in folds),
                             'geomean_speedup': math.exp(statistics.mean(math.log(f['paired_speedup']) for f in folds))}
    baselines = {m: {'geomean_speedup': math.exp(statistics.mean(math.log(r['paired_speedup'][m]) for r in records)),
                     'slower': sum(r['paired_speedup'][m] < 1 for r in records)} for m in sorted(masks)}
    out = {'classification': 'exploratory matched-iteration within-block analysis, three blocks',
           'warning': 'equal rounded loss does not prove identical inputs or runtime conditions; not a blind test',
           'input': str(args.input), 'input_sha256': hashlib.sha256(raw).hexdigest(), 'context': context,
           'blocks': records, 'evaluations': evaluations, 'baselines': baselines}
    with args.output.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'blocks': [{k: r[k] for k in ('block', 'median_delta_ms', 'median_paired_interaction_ms')}
                                for r in records],
                      'evaluations': {k: {x: v[x] for x in ('correct', 'slower', 'geomean_speedup')}
                                      for k, v in evaluations.items()}, 'baselines': baselines}, indent=2))


if __name__ == '__main__':
    main()
