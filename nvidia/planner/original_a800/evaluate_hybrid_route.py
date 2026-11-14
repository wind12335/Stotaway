"""Exact-shape versus family-leaveout evaluation using independent batches."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .evaluate_route_policy import flash
from .hybrid_route_policy import HybridRoutePolicy
from .orthogonal_predictor import Sample
from .route_policy import GroupedSample
from .validate_orthogonal_tp2 import route_for
from .validate_tp2_new import load


def run(train,test,hold_family):
    out=[]
    for row in test:
        group=[r for r in train if not hold_family or r.family!=row.family]
        if not group:raise ValueError('empty training set')
        policy=HybridRoutePolicy(row.sample.route).fit(group);p=policy.predict(row.sample.route,row.sample.shape)
        s=row.sample;gain=s.stock_ms/s.candidate_ms;choice='stock' if p['choice']=='stock' else 'overlap'
        out.append({'shape':s.shape,'family':row.family,'choice':choice,'source':p['source'],
                    'oracle':'overlap' if gain>1 else 'stock','speedup':gain if choice=='overlap' else 1.,
                    'actual_speedup':gain})
    return {'count':len(out),'correct':sum(r['choice']==r['oracle'] for r in out),
            'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in out)),
            'slower':sum(r['speedup']<1 for r in out),'exact_count':sum(r['source']=='exact_shape' for r in out),
            'rows':out}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dx-report',type=Path,required=True)
    ap.add_argument('--flash-report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    dx=json.loads(a.dx_report.read_bytes());fr=json.loads(a.flash_report.read_bytes())
    train,sig,_=load([Path(s['path']) for s in dx['train_sources']]);test,tsig,_=load([Path(s['path']) for s in dx['test_sources']])
    if sig!=tsig:raise ValueError('DX signature mismatch')
    sections={}
    for op in sorted({r['op'] for r in test}):
        route=route_for(sig,op)
        convert=lambda records:[GroupedSample(r['model'],Sample(route,(r['M'],r['N'],r['K']),r['stock_b2b'],r['dx_b2b'])) for r in records if r['op']==op]
        tr,te=convert(train),convert(test)
        sections['dx_'+op]={mode:run(tr,te,hold) for mode,hold in [('same_shape',False),('unseen_family',True)]}
    ftrain=[flash(Path(s['path']))[1] for s in fr['train_sources']];ftest=[flash(Path(s['path']))[1] for s in fr['test_sources']]
    sections['flashoverlap_rs']={mode:run(ftrain,ftest,hold) for mode,hold in [('same_shape',False),('unseen_family',True)]}
    out={'classification':'exploratory independent repeat, exact-shape calibration versus entire-family holdout',
         'warning':'data previously used in development; exact-shape uses target shape labels from earlier batch; no E2E claim',
         'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (a.dx_report,a.flash_report)],
         'sections':sections}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:{mode:{field:r[field] for field in ('count','correct','speedup_gmean','slower','exact_count')}
                         for mode,r in value.items()} for k,value in sections.items()},indent=2))


if __name__=='__main__':main()
