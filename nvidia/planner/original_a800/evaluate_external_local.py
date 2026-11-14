"""Compare local residual and base on existing FlashOverlap warm32 data."""
import argparse
import json
import math
import statistics
from pathlib import Path
from .evaluate_route_policy import flash
from .route_local_residual import RouteLocalResidual
from .validate_tp2_new import load
from .validate_orthogonal_tp2 import route_for
from .orthogonal_predictor import Sample
from .route_policy import GroupedSample


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--triton-dist',action='store_true')
    ap.add_argument('--independent-test',nargs='+',type=Path);a=ap.parse_args()
    data=json.loads(a.report.read_text());source=[Path(s['path']) for s in data['sources']]
    groups={}
    if a.triton_dist:
        cells,sig,_=load(source)
        if sig[0]!='triton_dist':raise ValueError('expected Triton-distributed route')
        for op in sorted({c['op'] for c in cells}):
            route=route_for(sig,op)
            groups[op]=(route,[GroupedSample(c['model'],Sample(route,(c['M'],c['N'],c['K']),c['stock_b2b'],c['dx_b2b'])) for c in cells if c['op']==op])
    else:
        pairs=[flash(p) for p in source]
        if len({route for route,_ in pairs})!=1:raise ValueError('mixed execution route')
        groups['rs']=(pairs[0][0],[sample for _,sample in pairs])
    test_groups=groups
    if a.independent_test:
        if a.triton_dist:
            cells,tsig,_=load(a.independent_test)
            if tsig!=sig:raise ValueError('test protocol mismatch')
            test_groups={}
            for op,(route,_) in groups.items():
                test_groups[op]=(route,[GroupedSample(c['model'],Sample(route,(c['M'],c['N'],c['K']),c['stock_b2b'],c['dx_b2b'])) for c in cells if c['op']==op])
        else:
            pairs=[flash(p) for p in a.independent_test]
            if any(route!=groups['rs'][0] for route,_ in pairs):raise ValueError('test route mismatch')
            test_groups={'rs':(groups['rs'][0],[sample for _,sample in pairs])}
    out=[]
    for op,(route,rows) in groups.items():
        test_rows=test_groups[op][1]
        for family in sorted({r.family for r in test_rows}):
            model=RouteLocalResidual(route).fit([r for r in rows if r.family!=family])
            for row in test_rows:
                if row.family!=family:continue
                s=row.sample;pred=model.predict(route,s.shape)
                out.append({'op':op,'family':family,'shape':s.shape,'prediction':pred,'actual_speedup':s.stock_ms/s.candidate_ms})
    def score(field):
        choices=[r['prediction'][field]>0 for r in out]
        speeds=[r['actual_speedup'] if c else 1. for c,r in zip(choices,out)]
        return {'count':len(out),'correct':sum(c==(r['actual_speedup']>1) for r,c in zip(out,choices)),
                'slower':sum(s<1 for s in speeds),'speedup_gmean':math.exp(statistics.mean(map(math.log,speeds)))}
    result={'classification':('Triton-distributed' if a.triton_dist else 'FlashOverlap')+' same-route leave-family-out development replay',
            'warning':'previously inspected data, not independent generalization; separate from DX; no E2E',
            'sources':data['sources'],'test_sources':[str(p) for p in (a.independent_test or [])],
            'summary':{'local':score('predicted_log_speedup'),'base':score('base_log_speedup')},'rows':out}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
