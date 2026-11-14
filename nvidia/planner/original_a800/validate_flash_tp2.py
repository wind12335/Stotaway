"""Shape-family leaveout for FlashOverlap TP2 data, separately from DX."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

try:
    from .orthogonal_predictor import Sample, RouteRegressor
    from .predictor_registry import bind
except ImportError:
    from orthogonal_predictor import Sample, RouteRegressor
    from predictor_registry import bind


def fit(rows):
    route = rows[0]['route']
    model = RouteRegressor(route).fit([Sample(r['route'], (r['M'], r['N'], r['K']), r['stock'], r['overlap'])
                                       for r in rows])
    return lambda r: model.predict(r['route'], (r['M'], r['N'], r['K']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows, signatures, sources = [], set(), []
    for p in args.inputs:
        raw = p.read_bytes(); d = json.loads(raw)
        signatures.add(tuple(d[k] for k in ('method','direction','tp','dtype','batch','gpu','torch')) +
                       (d.get('warmup', 4), d.get('repeats'),))
        if d['method'] != 'flashoverlap' or d['direction'] != 'rs' or d['tp'] != 2 or d['batch'] != 16:
            raise ValueError('unsupported signature')
        route = bind({'method':'flashoverlap', 'actual_path':'cutlass_nccl_wave', 'direction':'rs', 'fallback':False},
                     gpu=d['gpu'], topology='nv8', tp=d['tp'], dtype=d['dtype'],
                     versions=(('torch', str(d['torch'])),), timing_mode='b2b16',
                     warmup=d.get('warmup', 4), config_family='predictive_search')
        rows.append({k:d[k] for k in ('M','N','K')} | {'stock':d['measurements']['stock']['median_ms'],
                    'overlap':d['measurements']['overlap']['median_ms'], 'source':str(p), 'route':route})
        sources.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
    if len(signatures)!=1 or len({(r['M'],r['N'],r['K']) for r in rows})!=len(rows):
        raise ValueError('signature mismatch or repeated exact shape')
    # Hold all M values of the same N/K together; this is more conservative
    # than treating different batch sizes as independent model families.
    families = {(r['N'],r['K']) for r in rows}
    out = []
    for family in sorted(families):
        train = [r for r in rows if (r['N'],r['K']) != family]
        if len(train)<2:
            continue
        model = fit(train)
        for r in rows:
            if (r['N'],r['K']) != family:
                continue
            pred = model(r)
            choice = 'overlap' if pred > 0 else 'stock'
            speed = r['stock']/r['overlap'] if choice=='overlap' else 1.0
            out.append({**{k:v for k,v in r.items() if k != 'route'}, 'pred_log_speedup':pred, 'choice':choice,
                        'oracle':'overlap' if r['overlap'] < r['stock'] else 'stock', 'speedup':speed})
    if not out:
        raise ValueError('too few distinct shape families')
    summary = {'count':len(out), 'families':len(families),
               'correct':sum(r['choice']==r['oracle'] for r in out),
               'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in out)),
               'slower':sum(r['speedup']<1 for r in out),
               'always_stock_correct':sum(r['oracle']=='stock' for r in out),
               'always_overlap_correct':sum(r['oracle']=='overlap' for r in out),
               'always_overlap_gmean':math.exp(statistics.mean(math.log(r['stock']/r['overlap']) for r in out)),
               'oracle_gmean':math.exp(statistics.mean(math.log(max(1,r['stock']/r['overlap'])) for r in out))}
    report = {'method':'FlashOverlap RS; leave-(N,K)-out; small exploratory data',
              'sources':sources, 'signature':list(signatures)[0], 'summary':summary, 'rows':out}
    with args.output.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
