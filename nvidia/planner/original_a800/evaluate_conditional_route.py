"""Evaluate physical-component model with held-out shape families."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .conditional_route_policy import ConditionalRoutePolicy
from .orthogonal_predictor import Sample
from .route_policy import GroupedSample
from .validate_orthogonal_tp2 import route_for
from .validate_tp2_new import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dx-report',type=Path,required=True)
    ap.add_argument('--triton-report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sections={}
    for method,path in [('dx',a.dx_report),('triton_dist',a.triton_report)]:
        report=json.loads(path.read_bytes())
        if method=='dx':
            train,sig,_=load([Path(s['path']) for s in report['train_sources']]);test,tsig,_=load([Path(s['path']) for s in report['test_sources']])
            if sig!=tsig:raise ValueError('signature mismatch')
        else:
            train,sig,_=load([Path(s['path']) for s in report['sources']]);test=train
        for op in sorted({r['op'] for r in test}):
            route=route_for(sig,op)
            def convert(data):
                return [GroupedSample(r['model'],Sample(route,(r['M'],r['N'],r['K']),r['stock_b2b'],r['dx_b2b'],r['G'],r['C'])) for r in data if r['op']==op]
            tr,te=convert(train),convert(test)
            for use_components in (False,True):
                rows=[]
                for target in te:
                    sub=[r for r in tr if r.family!=target.family]
                    policy=ConditionalRoutePolicy(route,use_components=use_components).fit(sub)
                    pred=policy.predict(route,target.sample);s=target.sample
                    speed=s.stock_ms/s.candidate_ms if pred['choice']!='stock' else 1.
                    rows.append({'shape':s.shape,'family':target.family,'choice':pred['choice'],
                                 'oracle':route.method if s.candidate_ms<s.stock_ms else 'stock',
                                 'speedup':speed,'audit':policy.audit})
                name=f'{method}_{op}_{"components" if use_components else "shape"}'
                sections[name]={'summary':{'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),
                   'slower':sum(r['speedup']<1 for r in rows),
                   'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in rows))},'rows':rows}
    out={'classification':'exploratory grouped nested evaluation; target-shape component times externally measured',
         'warning':'component measurement cost excluded; prior data used in development; no FlashOverlap components in this dataset',
         'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (a.dx_report,a.triton_report)],'sections':sections}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v['summary'] for k,v in sections.items()},indent=2))


if __name__=='__main__':main()
