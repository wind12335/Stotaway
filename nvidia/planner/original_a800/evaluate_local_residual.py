"""Strictly nested family holdout for route-local residual correction."""
import argparse
import json
import math
import statistics
from pathlib import Path

from .evaluate_fragmentation_mixture import join
from .local_residual_policy import LocalResidualPolicy


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-unknown-stage',action='store_true',help='historical metadata-only analysis; marks unknown fields')
    a=ap.parse_args();rows,sig,ops,stages=join(a.operators,a.stages,allow_unknown_stage=a.allow_unknown_stage);results=[]
    for op in sorted({r['op'] for r in rows}):
        group=[r for r in rows if r['op']==op]
        for family in sorted({r['family'] for r in group}):
            model=LocalResidualPolicy().fit([r for r in group if r['family']!=family],op=op)
            for row in group:
                if row['family']!=family:continue
                query={k:v for k,v in row.items() if k not in ('comm_ms','stock_ms','overlap_ms')}
                results.append({'identity':row['identity'],'family':family,'op':op,
                                'prediction':model.predict(query),'actual_speedup':row['stock_ms']/row['overlap_ms']})
    def summary(mode):
        choices=[r['prediction']['predicted_log_speedup' if mode=='local' else 'base_log_speedup']>0 for r in results]
        speeds=[r['actual_speedup'] if c else 1. for r,c in zip(results,choices)]
        return {'count':len(results),'correct':sum(c==(r['actual_speedup']>1) for r,c in zip(results,choices)),
                'slower':sum(v<1 for v in speeds),'speedup_gmean':math.exp(statistics.mean(map(math.log,speeds)))}
    out={'classification':'nested physical family holdout, local residual correction',
         'warning':'development data, not blind; target component and startup costs excluded',
         'allow_unknown_stage':a.allow_unknown_stage,
         'signature':sig,'operator_sources':ops,'stage_sources':stages,
         'summary':{k:summary(k) for k in ('local','base')},'rows':results}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':main()
