"""Evaluate fixed piecewise architecture trained on earlier groups at V4."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .piecewise_gain_policy import PiecewiseGainPolicy
from .evaluate_fragmentation_mixture import join


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--target-decisions',type=Path,required=True)
    ap.add_argument('--target-results',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    rows,sig,ops,stages=join(a.operators,a.stages);target=json.loads(a.target_decisions.read_bytes())
    if list(sig)!=target['signature']:raise ValueError('route mismatch')
    models={op:PiecewiseGainPolicy().fit([r for r in rows if r['op']==op]) for op in sorted({r['op'] for r in rows})}
    predictions={tuple(q['identity']):models[q['op']].predict(q['query']) for q in target['predictions']}
    results=[]
    for path in a.target_results:
        data=json.loads(path.read_bytes());current=[]
        for r in data['rows']:
            p=predictions[tuple(r['identity'])];speed=r['actual_speedup'] if p['choice']=='overlap' else 1.
            current.append({'identity':r['identity'],'choice':p['choice'],'oracle':r['oracle'],'speedup':speed})
        if len(current)!=len(predictions):raise ValueError('inventory mismatch')
        results.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'summary':{'count':len(current),'correct':sum(r['choice']==r['oracle'] for r in current),
                        'slower':sum(r['speedup']<1 for r in current),
                        'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in current))},'rows':current})
    out={'classification':'exploratory retrospective V4 test of piecewise candidate',
         'warning':'architecture designed after V4 labels viewed; not blind; exact target stock/GEMM/comm measured',
         'train_sources':ops,'stage_sources':stages,'target_decisions_sha256':hashlib.sha256(a.target_decisions.read_bytes()).hexdigest(),
         'results':results}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps([r['summary'] for r in results],indent=2))


if __name__=='__main__':main()
