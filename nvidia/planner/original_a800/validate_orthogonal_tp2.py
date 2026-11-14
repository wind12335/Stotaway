"""Compare shape and measured-component features within each exact TP2 route."""
import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from .orthogonal_predictor import Sample, RouteRegressor
    from .predictor_registry import bind
    from .validate_tp2_new import load
except ImportError:
    from orthogonal_predictor import Sample, RouteRegressor
    from predictor_registry import bind
    from validate_tp2_new import load


def route_for(signature, op):
    method,tp,dtype,gpu,torch,triton,cuda,connection,batch,dist_path,autotune,warmup = signature
    direction = 'rs' if op == 'rs' else 'ag'
    versions = (('cuda', str(cuda)), ('torch', str(torch)), ('triton', str(triton)))
    if method == 'dx':
        path = 'direct_exchange'
        config = f'{op};connection={connection}'
    if method == 'triton_dist' and direction == 'ag':
        path = 'nonpersistent_ag_all2all_pull'
        config = f'{op};autotune={autotune};path={dist_path};connection={connection}'
    elif method != 'dx':
        raise ValueError('unknown or unverified route')
    return bind({'method':method, 'actual_path':path, 'direction':direction, 'fallback':False},
                gpu=gpu, topology='nv8', tp=tp, dtype=dtype, versions=versions,
                timing_mode='b2b16', warmup=warmup, config_family=config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows, signature, sources = load(args.inputs)
    predictions = []
    for op in sorted({r['op'] for r in rows}):
        group = [r for r in rows if r['op'] == op]
        route = route_for(signature, op)
        for family in sorted({r['model'] for r in group}):
            train = [r for r in group if r['model'] != family]
            if len(train) < 2:
                continue
            samples = [Sample(route, (r['M'],r['N'],r['K']), r['stock_b2b'],r['dx_b2b'],r['G'],r['C']) for r in train]
            for use_components in (False, True):
                model = RouteRegressor(route, use_components=use_components).fit(samples)
                for r in (r for r in group if r['model'] == family):
                    p = model.predict(route, (r['M'],r['N'],r['K']), r['G'],r['C'])
                    pick = 'overlap' if p > 0 else 'stock'
                    speed = r['stock_b2b']/r['dx_b2b'] if pick == 'overlap' else 1.0
                    predictions.append({**r, 'route':route.__dict__, 'component_features':use_components,
                                        'predicted_log_speedup':p,'choice':pick,
                                        'oracle':'overlap' if r['dx_b2b'] < r['stock_b2b'] else 'stock',
                                        'speedup':speed})
    def summ(v):
        return {'count':len(v),'correct':sum(r['choice']==r['oracle'] for r in v),
                'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in v)),
                'slower':sum(r['speedup'] < 1 for r in v),
                'always_overlap_correct':sum(r['oracle']=='overlap' for r in v),
                'always_overlap_gmean':math.exp(statistics.mean(math.log(r['stock_b2b']/r['dx_b2b']) for r in v))}
    report = {'classification':'exploratory grouped route-isolated evaluation', 'signature':signature,
              'warning':'G and C are measured on the test shape; not free at planning time; no E2E claim',
              'sources':sources,'summary':{str(mode):summ([r for r in predictions if r['component_features']==mode])
                                           for mode in (False,True)}, 'rows':predictions}
    with args.output.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report['summary'],indent=2))


if __name__ == '__main__':
    main()
