"""Retrospective process-held-out diagnostic for pre-probed state features.

One known shape and route: does measuring collective+GEMM immediately before the
stock/DX trial explain the batch-to-batch winner? Not a deployment algorithm.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
from .replay_interleaved_calibration import SIGNATURE


def evaluate(paths, *, ridge=4.):
    if not math.isfinite(ridge) or ridge<=0:raise ValueError('positive ridge required')
    records=[];signature=None;seen=set()
    for path in paths:
        raw=Path(path).read_bytes();d=json.loads(raw);sig={k:d[k] for k in SIGNATURE}
        if signature is not None and sig!=signature:raise ValueError('mixed route or measurement protocol')
        signature=sig
        if d['seed'] in seen:raise ValueError('duplicate process seed')
        seen.add(d['seed'])
        if d.get('classification')!='interleaved TP2 direct-exchange operator probe' or d['checks']<1:
            raise ValueError('checked paired probes required')
        c=d['pre_components'];features=[c['gemm']['median_ms'],c['comm']['median_ms']]
        if any(not math.isfinite(x) or x<=0 for x in features):raise ValueError('invalid feature')
        samples=d['block_results']
        if len(samples)!=d['blocks'] or len(samples)<2:raise ValueError('incomplete process block')
        speed=statistics.median(x['stock_ms']/x['dx_ms'] for x in samples)
        records.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'seed':d['seed'],
                        'features':features,'speedup_dx':speed,'pre_component_wall_ms':c['wall_ms']})
    if len(records)<4:raise ValueError('at least four independent processes required')
    out=[]
    for i,target in enumerate(records):
        train=[r for j,r in enumerate(records) if j!=i]
        x=np.log(np.array([r['features'] for r in train]));y=np.array([math.log(r['speedup_dx']) for r in train])
        center=x.mean(0);sd=np.maximum(x.std(0),.03);z=(x-center)/sd
        beta=np.linalg.solve(z.T@z+ridge*np.eye(2),z.T@(y-y.mean()))
        predicted=float(y.mean()+((np.log(target['features'])-center)/sd)@beta)
        base=float(y.mean());actual=target['speedup_dx'];oracle='dx' if actual>1 else 'stock'
        choices={'state':'dx' if predicted>0 else 'stock','constant':'dx' if base>0 else 'stock',
                 'stock':'stock','dx':'dx'}
        out.append({**target,'predicted_log_speedup':predicted,'constant_log_speedup':base,
                    'oracle':oracle,'choices':choices})
    summary={}
    for mode in ('state','constant','stock','dx'):
        speeds=[r['speedup_dx'] if r['choices'][mode]=='dx' else 1. for r in out]
        summary[mode]={'correct':sum(r['choices'][mode]==r['oracle'] for r in out),
                       'total':len(out),'slower':sum(s<1 for s in speeds),
                       'geomean_speedup':math.exp(statistics.mean(math.log(s) for s in speeds))}
    return {'classification':'exploratory leave-one-process-out state-feature diagnostic',
            'signature':signature,'ridge':ridge,'summary':summary,'rows':out,
            'warning':'same previously examined shape; labels and pre-components from same process; '
                      'LOO trains on later processes; not chronological or prospective; '
                      'component cost excluded and no E2E result'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('reports',nargs='+',type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();result=evaluate(a.reports)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
