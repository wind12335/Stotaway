"""Nested family holdout for locally adapted residual shrinkage."""
import argparse
import json
import math
import statistics
from pathlib import Path
from .evaluate_fragmentation_mixture import join
from .adaptive_local_residual import AdaptiveLocalResidual


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();data,sig,ops,stages=join(a.operators,a.stages);out=[]
    for op in sorted({r['op'] for r in data}):
        rows=[r for r in data if r['op']==op]
        for family in sorted({r['family'] for r in rows}):
            model=AdaptiveLocalResidual().fit([r for r in rows if r['family']!=family],op=op)
            for row in rows:
                if row['family']!=family:continue
                query={k:v for k,v in row.items() if k not in ('comm_ms','stock_ms','overlap_ms')}
                out.append({'identity':row['identity'],'family':family,'op':op,'prediction':model.predict(query),
                            'actual_speedup':row['stock_ms']/row['overlap_ms'],'inner_audit':model.audit})
    def score(field):
        choices=[r['prediction'][field]>0 for r in out];speeds=[r['actual_speedup'] if c else 1. for c,r in zip(choices,out)]
        return {'count':len(out),'correct':sum(c==(r['actual_speedup']>1) for r,c in zip(out,choices)),
                'slower':sum(s<1 for s in speeds),'speedup_gmean':math.exp(statistics.mean(map(math.log,speeds)))}
    result={'classification':'nested physical-family holdout, training-only adaptive shrinkage',
            'warning':'development data; protocol changes were informed by earlier experiments; no probe cost',
            'signature':sig,'operator_sources':ops,'stage_sources':stages,
            'summary':{'adaptive':score('predicted_log_speedup'),'fixed_local':score('fixed_local_log_speedup'),
                       'base':score('base_log_speedup')},'rows':out}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
