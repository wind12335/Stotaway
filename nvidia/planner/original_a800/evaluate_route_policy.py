"""Compare one generic route-local policy on the existing multi-backend data."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .orthogonal_predictor import Sample
from .route_policy import GroupedSample, RoutePolicy
from .predictor_registry import bind
from .validate_orthogonal_tp2 import route_for
from .validate_tp2_new import load


def flash(path):
    data=json.loads(path.read_bytes())
    route=bind({'method':'flashoverlap','actual_path':'cutlass_nccl_wave','direction':'rs','fallback':False},
               gpu=data['gpu'],topology='nv8',tp=data['tp'],dtype=data['dtype'],
               versions=(('torch',str(data['torch'])),),timing_mode='b2b16',
               warmup=data.get('warmup'),config_family='predictive_search')
    return route,GroupedSample(f"{data['N']}:{data['K']}",Sample(route,(data['M'],data['N'],data['K']),
        data['measurements']['stock']['median_ms'],data['measurements']['overlap']['median_ms']))


def metrics(out):
    return {'count':len(out),'correct':sum(r['choice']==r['oracle'] for r in out),
            'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in out)),
            'slower':sum(r['speedup']<1 for r in out),
            'always_overlap_gmean':math.exp(statistics.mean(math.log(r['actual_speedup']) for r in out)),
            'always_overlap_correct':sum(r['oracle']=='overlap' for r in out),
            'always_stock_correct':sum(r['oracle']=='stock' for r in out)}


def evaluate(train,test,policy_cls=RoutePolicy):
    out=[]
    for family in sorted({r.family for r in test}):
        group=[r for r in train if r.family!=family]
        if not group:continue
        route=group[0].sample.route
        try:policy=policy_cls(route).fit(group)
        except ValueError as exc:policy=None;reason=str(exc)
        for row in (r for r in test if r.family==family):
            s=row.sample
            if s.route!=route:raise ValueError('train/test route mismatch')
            result=policy.predict(route,s.shape) if policy else {'choice':'stock','reason':reason}
            actual=s.stock_ms/s.candidate_ms
            choice='stock' if result['choice']=='stock' else 'overlap'
            out.append({'family':family,'shape':s.shape,'actual_speedup':actual,'choice':choice,
                        'oracle':'overlap' if actual>1 else 'stock','speedup':actual if choice=='overlap' else 1.,
                        'prediction':result,'audit':policy.audit if policy else None})
    if not out:raise ValueError('empty evaluation')
    return {'summary':metrics(out),'rows':out}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dx-report',type=Path,required=True)
    ap.add_argument('--flash-report',type=Path,required=True);ap.add_argument('--triton-report',type=Path,required=True)
    ap.add_argument('--family-weighted',action='store_true')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    dx=json.loads(a.dx_report.read_bytes());triton=json.loads(a.triton_report.read_bytes());fr=json.loads(a.flash_report.read_bytes())
    dx_train,sig,_=load([Path(s['path']) for s in dx['train_sources']]);dx_test,tsig,_=load([Path(s['path']) for s in dx['test_sources']])
    if sig!=tsig:raise ValueError('DX signature mismatch')
    def convert(data,signature,op):
        r=route_for(signature,op)
        return [GroupedSample(c['model'],Sample(r,(c['M'],c['N'],c['K']),c['stock_b2b'],c['dx_b2b']))
                for c in data if c['op']==op]
    if a.family_weighted:
        from .multi_route_selector import MultiRouteSelector
        policy_cls=MultiRouteSelector
    else:policy_cls=RoutePolicy
    sections={}
    for op in sorted({c['op'] for c in dx_test}):sections['dx_'+op]=evaluate(convert(dx_train,sig,op),convert(dx_test,sig,op),policy_cls)
    sources=[Path(s['path']) for s in triton['sources']]
    td,tdsig,_=load(sources)
    for op in sorted({c['op'] for c in td}):
        data=convert(td,tdsig,op);sections['triton_dist_'+op]=evaluate(data,data,policy_cls)
    train=[flash(Path(s['path'])) for s in fr['train_sources']]
    test=[flash(Path(s['path'])) for s in fr['test_sources']]
    if len({r for r,_ in train+test})!=1:raise ValueError('FlashOverlap route mismatch')
    sections['flashoverlap_rs']=evaluate([x for _,x in train],[x for _,x in test],policy_cls)
    reports=[a.dx_report,a.flash_report,a.triton_report]
    out={'classification':'exploratory route-local grouped OOF policy, prior data used in development',
         'policy':'family_weighted_multi_ridge' if a.family_weighted else 'route_policy',
         'warning':'DX and FlashOverlap use independent repeat batches; Triton-dist uses grouped leaveout from same batch; no E2E or blind test',
         'source_reports':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in reports],
         'sections':sections}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v['summary'] for k,v in sections.items()},indent=2))


if __name__=='__main__':main()
