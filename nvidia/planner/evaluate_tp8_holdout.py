"""Evaluate a conservative TP8 forward latency selector on held-out models.

Training and validation read only A800 probe data. Models are grouped by model
name, and missing cells are never imputed. This is an offline evaluation, not
a deployed E2E planner or an implementation of BW1000's feature set.
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path('/root/Stowaway-NVIDIA')
PROBES = Path('/root/data/stowaway-nvidia/results/tp8_20261111')
cells = json.loads(Path('/root/Stotaway/planner/data/cells.json').read_text())

def features(c):
    m, k, n = (math.log(float(c[v])) for v in ('M','K','N'))
    return np.array([1, m, k, n], dtype=float)

def gmean(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None

def fit(rows):
    x = np.stack([features(c) for c in rows])
    y = np.array([math.log(c['dx']) for c in rows])
    # Ridge penalty on centered features avoids a singular design matrix.
    mean, scale = x[:,1:].mean(axis=0), x[:,1:].std(axis=0)
    scale[scale < 1e-9] = 1
    z = np.c_[np.ones(len(x)), (x[:,1:] - mean)/scale]
    penalty = np.diag([0] + [0.1]*(z.shape[1]-1))
    coef = np.linalg.solve(z.T@z + penalty, z.T@y)
    residual = y - z@coef
    return mean, scale, coef, float(np.quantile(residual, .8))

def predict(model, c):
    mean, scale, coef, guard = model
    x = features(c)
    return math.exp(float(np.r_[1, (x[1:]-mean)/scale]@coef) + guard)

def read(mode):
    rows = []
    for cell in cells:
        p = PROBES / f"bwmatch_probe_{cell['model']}_m{cell['M']}_k{cell['K']}_n{cell['N']}_b{mode}.json"
        if not p.exists(): continue
        d = json.loads(p.read_text())
        if d.get('world') != 8 or d.get('checks',0) < 4 or d.get('batch') != mode:
            continue
        op = 'rs' if cell['op'] == 'rs' else 'ag'
        v = d['result'][op]
        rows.append({**{k:cell[k] for k in ('model','M','K','N','op')},
                     'stock':v['stock_median_ms'], 'dx':v['dx_median_ms']})
    return rows

def evaluate(rows):
    models = sorted({c['model'] for c in rows})
    output = []
    for hold in models:
        train = [c for c in rows if c['model'] != hold]
        test = [c for c in rows if c['model'] == hold]
        trained = {}
        for op in ('rs','ag_up','ag_gu'):
            s = [c for c in train if c['op'] == op]
            if len(s) >= 8 and len({c['model'] for c in s}) >= 3:
                trained[op] = fit(s)
        # Threshold is fitted only on training models, using the same
        # stock-vs-DX candidate set as this evaluation.
        thresholds = (4096,8192,16384,32768,10**9)
        threshold = max(thresholds, key=lambda t:gmean([c['stock']/(c['dx'] if c['M']>=t else c['stock']) for c in train]) or 0)
        for c in test:
            estimate = predict(trained[c['op']],c) if c['op'] in trained else None
            # The measured stock time is a probe input, not an E2E time.
            choice = 'dx' if estimate is not None and estimate <= .98*c['stock'] else 'stock'
            oracle = 'dx' if c['dx'] < c['stock'] else 'stock'
            static = 'dx' if c['M'] >= threshold else 'stock'
            output.append(dict(**c, choice=choice, oracle=oracle, static=static,
                               pessimistic_dx_prediction_ms=estimate, threshold=threshold))
    return output

def main():
    report = {'classification':'A800 TP8 forward-only leave-one-model-out',
              'warning':'Not an E2E predictor; no BW1000 anomalies automatically excluded.', 'modes':{}}
    for mode in (1,16):
        rows = read(mode)
        models = {r['model'] for r in rows}
        if len(models) < 4:
            report['modes'][str(mode)] = {'status':'insufficient_data','cells':len(rows),'models':len(models)}
            continue
        decisions = evaluate(rows)
        errors = [abs(r['pessimistic_dx_prediction_ms']/r['dx']-1) for r in decisions if r['pessimistic_dx_prediction_ms'] is not None]
        report['modes'][str(mode)] = {'status':'evaluated_partial' if len(rows)<99 else 'evaluated',
            'cells':len(rows),'models':len(models),'correct':sum(r['choice']==r['oracle'] for r in decisions),
            'slower_than_stock':sum(r[r['choice']]>r['stock'] for r in decisions),
            'geomean_selector':gmean([r['stock']/r[r['choice']] for r in decisions]),
            'geomean_always_dx':gmean([r['stock']/r['dx'] for r in decisions]),
            'geomean_static':gmean([r['stock']/r[r['static']] for r in decisions]),
            'geomean_oracle':gmean([r['stock']/min(r['stock'],r['dx']) for r in decisions]),
            'guarded_prediction_mape':statistics.mean(errors) if errors else None, 'decisions':decisions,
            'limitation':'Uses held-out measured stock timing as probe input; this is an oracle-assisted offline evaluation, not deployable unseen-shape inference.'}
    path = ROOT/'planner/data_a800/tp8_holdout_20261111.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x!='decisions'} for k,v in report['modes'].items()},indent=2))

if __name__=='__main__': main()
