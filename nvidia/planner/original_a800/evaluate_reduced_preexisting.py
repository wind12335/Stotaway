"""Retrospective reduced-probe model evaluation against frozen V4/V5 runs."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .evaluate_fragmentation_mixture import join
from .reduced_probe_policy import ReducedProbePolicy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--target-decisions',type=Path,required=True)
    ap.add_argument('--target-results',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    data,sig,ops,stages=join(a.operators,a.stages);raw=a.target_decisions.read_bytes();target=json.loads(raw)
    if list(sig)!=target['signature']:raise ValueError('runtime mismatch')
    models={op:ReducedProbePolicy().fit([r for r in data if r['op']==op],op=op) for op in sorted({r['op'] for r in data})}
    predictions={}
    for r in target['predictions']:
        q={k:v for k,v in r['query'].items() if k not in ('comm_ms','stock_ms')};q['op']=r['op']
        predictions[tuple(r['identity'])]=models[r['op']].predict(q)
    out=[]
    for p in a.target_results:
        report=json.loads(p.read_bytes());rows=[]
        for r in report['rows']:
            pred=predictions[tuple(r['identity'])];speed=r['actual_speedup'] if pred['choice']=='overlap' else 1.
            rows.append({'identity':r['identity'],'choice':pred['choice'],'oracle':r['oracle'],'speedup':speed,'prediction':pred})
        if len(rows)!=len(predictions):raise ValueError('inventory mismatch')
        out.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'summary':{'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),
                    'slower':sum(r['speedup']<1 for r in rows),
                    'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in rows))},'rows':rows})
    output={'classification':'exploratory retrospective reduced probe evaluation',
            'warning':'target labels already inspected; target stock/comm not used; no probe costs in speedup',
            'train_sources':ops,'stage_sources':stages,'target_components_sha256':hashlib.sha256(raw).hexdigest(),'results':out}
    with a.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps([r['summary'] for r in out],indent=2))


if __name__=='__main__':main()
