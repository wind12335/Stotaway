"""Nested leave-one-model selection of relative-predictor guard/ridge values.

This is exploratory on previously used data; each outer model is never used
to tune its settings. Inner objectives and grids are fixed here, not derived
from each outer test. No guarantee applies to unseen environments.
"""
import hashlib
import json
import math
from pathlib import Path

from relative_model import RelativeModel
from costmodel import CostModel

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / 'data/cells.json'
    raw = path.read_bytes()
    cells = [c for c in json.loads(raw) if not c.get('anomaly')]
    families = sorted({c['model'] for c in cells})
    # Conservative inner validation objective: geometric log gain penalized
    # fivefold for any selected regression beyond 0.5%. This is an explicit
    # research trade-off, not a risk guarantee.
    settings = [(ridge, quantile, margin) for ridge in (1.0, 10.0, 100.0)
                for quantile in (0.7, 0.8, 0.9) for margin in (0.01, 0.02)]
    rows = []
    for held in families:
        outer_train = [c for c in cells if c['model'] != held]
        inner_families = sorted({c['model'] for c in outer_train})
        scores = []
        for ridge, quantile, margin in settings:
            gains = []
            for inner in inner_families:
                fit = [c for c in outer_train if c['model'] != inner]
                test = [c for c in outer_train if c['model'] == inner]
                model = RelativeModel(ridge=ridge, quantile=quantile, margin=margin).fit(fit)
                for c in test:
                    choice, _ = model.choose(c)
                    ratio = c['stock_b2b'] / (c['dx_b2b'] if choice == 'dx' else c['stock_b2b'])
                    gains.append(math.log(ratio) - 5 * max(0.0, -math.log(ratio) - math.log(1.005)))
            scores.append(sum(gains) / len(gains))
        best_index = max(range(len(scores)), key=lambda i: (scores[i], -i))
        ridge, quantile, margin = settings[best_index]
        model = RelativeModel(ridge=ridge, quantile=quantile, margin=margin).fit(outer_train)
        original = CostModel(mode='b2b').fit(outer_train)
        for c in (c for c in cells if c['model'] == held):
            pick, prediction = model.choose(c)
            stock, dx = c['stock_b2b'], c['dx_b2b']
            rows.append({'model': held, 'op': c['op'], 'M': c['M'], 'K': c['K'], 'N': c['N'],
                         'pick': pick, 'original_pick': original.choose(c['op'], c['G'], c['B'], allowed={'dx'})[0],
                         'stock': stock, 'dx': dx, 'prediction': prediction,
                         'settings': {'ridge': ridge, 'quantile': quantile, 'margin': margin}})
        print('nested held', held, 'selected', settings[best_index], flush=True)
    def stats(method):
        selected = [r['dx'] if r[method] == 'dx' else r['stock'] for r in rows]
        return {'correct': sum(r[method] == ('dx' if r['dx'] < r['stock'] else 'stock') for r in rows),
                'n': len(rows), 'speedup_gmean': math.exp(sum(math.log(r['stock']/t) for r,t in zip(rows,selected))/len(rows)),
                'slow_gt_0_5pct': sum(t > r['stock'] * 1.005 for r,t in zip(rows,selected)),
                'max_regret_pct': max(100*(t/min(r['stock'],r['dx'])-1) for r,t in zip(rows,selected))}
    report = {'data_sha256': hashlib.sha256(raw).hexdigest(),
              'classification': 'exploratory nested leave-one-model-out; no independent test',
              'inner_objective': 'log(speedup) - 5*max(0, -log(speedup)-log(1.005))',
              'settings_grid': settings, 'nested': stats('pick'), 'original': stats('original_pick'), 'rows': rows}
    target = HERE / 'data/nested_exploratory.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'nested': report['nested'], 'original': report['original']}, indent=2))


if __name__ == '__main__':
    main()
