"""Evaluate family-jackknife vote rules on held-out physical families."""
import argparse
import json
import math
import statistics
from pathlib import Path

from .evaluate_fragmentation_mixture import join
from .reduced_family_ensemble import ReducedFamilyEnsemble


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    rows,sig,operators,stages=join(args.operators,args.stages)
    results=[]
    for op in sorted({r['op'] for r in rows}):
        group=[r for r in rows if r['op']==op]
        for held in sorted({r['family'] for r in group}):
            model=ReducedFamilyEnsemble().fit([r for r in group if r['family']!=held],op=op)
            for row in group:
                if row['family']!=held:continue
                query={k:v for k,v in row.items() if k not in ('comm_ms','stock_ms','overlap_ms')}
                pred=model.predict(query)
                results.append({'identity':row['identity'],'op':op,'family':held,'prediction':pred,
                                'actual_speedup':row['stock_ms']/row['overlap_ms']})
    def summary(mode):
        def choice(p):
            if mode=='full':return p['full_log_speedup']>0
            if mode=='majority':return p['positive_votes']*2>p['votes']
            if mode=='unanimous':return p['positive_votes']==p['votes']
            if mode=='overlap':return True
            return False
        selected=[choice(r['prediction']) for r in results]
        speeds=[r['actual_speedup'] if yes else 1. for r,yes in zip(results,selected)]
        return {'count':len(results),'correct':sum(yes==(r['actual_speedup']>1) for r,yes in zip(results,selected)),
                'slower':sum(v<1 for v in speeds),'speedup_gmean':math.exp(statistics.mean(map(math.log,speeds)))}
    out={'classification':'family held-out ensemble diagnostics on development data',
         'warning':'jackknife folds overlap; votes are not confidence intervals; no probe costs or E2E',
         'signature':sig,'operator_sources':operators,'stage_sources':stages,
         'summary':{m:summary(m) for m in ('full','majority','unanimous','overlap','stock')},'rows':results}
    with args.output.open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':main()
