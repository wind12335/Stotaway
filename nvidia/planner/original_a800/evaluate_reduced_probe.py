"""Family holdout for reduced target measurements, with fold-local fitting."""
import argparse
import json
import math
import statistics
from pathlib import Path

from .evaluate_fragmentation_mixture import join
from .reduced_probe_policy import ReducedProbePolicy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--scale-floor',type=float,default=0.);a=ap.parse_args()
    data,sig,ops,stages=join(a.operators,a.stages);results=[]
    for row in data:
        train=[r for r in data if r['op']==row['op'] and r['family']!=row['family']]
        model=ReducedProbePolicy(scale_floor=a.scale_floor).fit(train,op=row['op'])
        query={k:v for k,v in row.items() if k not in ('comm_ms','stock_ms','overlap_ms')}
        pred=model.predict(query);actual=row['stock_ms']/row['overlap_ms']
        results.append({'identity':row['identity'],'family':row['family'],'choice':pred['choice'],
                        'oracle':'overlap' if actual>1 else 'stock','speedup':actual if pred['choice']=='overlap' else 1.,
                        'actual_speedup':actual,'prediction':pred})
    def summary(pick):
        choices=[r['choice'] if pick=='model' else pick for r in results]
        speeds=[r['actual_speedup'] if c=='overlap' else 1. for c,r in zip(choices,results)]
        return {'count':len(results),'correct':sum(c==r['oracle'] for c,r in zip(choices,results)),
                'slower':sum(v<1 for v in speeds),'speedup_gmean':math.exp(statistics.mean(math.log(v) for v in speeds))}
    out={'classification':'exploratory nested family-held-out reduced probe model',
         'warning':'development data; target full/split GEMM measured; per-fold communication and stock fitted from other families; cost excluded',
         'signature':sig,'scale_floor':a.scale_floor,'operator_sources':ops,'stage_sources':stages,
         'summary':{p:summary(p) for p in ('model','stock','overlap')},'rows':results}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':main()
