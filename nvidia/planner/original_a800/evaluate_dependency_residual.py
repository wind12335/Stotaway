"""Evaluate grouped-calibrated residual model on held-out TP2 batches."""
import argparse
import json
import math
import statistics
from pathlib import Path

from .dependency_residual import DependencyResidual,Observation,Query
from .validate_orthogonal_tp2 import route_for
from .validate_tp2_new import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',nargs='+',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();training,sig,src=load(args.train);test,tsig,tsrc=load(args.test)
    if sig!=tsig:raise ValueError('training and test route signatures must match')
    out=[]
    for cell in test:
        route=route_for(sig,cell['op'])
        rows=[r for r in training if r['op']==cell['op'] and r['model']!=cell['model']]
        obs=[Observation(route,r['model'],(r['M'],r['N'],r['K']),r['stock_b2b'],r['dx_b2b'],r['G'],r['C']) for r in rows]
        query=Query(route,(cell['M'],cell['N'],cell['K']),cell['G'],cell['C'])
        try:prediction=DependencyResidual(route).fit(obs).predict(query)
        except ValueError as exc:prediction={'choice':'stock','reason':str(exc)}
        pick=prediction['choice'];speed=cell['stock_b2b']/cell['dx_b2b'] if pick!='stock' else 1.
        out.append({**cell,'prediction':prediction,'choice':pick,
                    'oracle':route.method if cell['dx_b2b']<cell['stock_b2b'] else 'stock','speedup':speed})
    gm=lambda values:math.exp(statistics.mean(math.log(x) for x in values))
    summary={'count':len(out),'correct':sum(r['choice']==r['oracle'] for r in out),
             'speedup_gmean':gm([r['speedup'] for r in out]),'slower':sum(r['speedup']<1 for r in out),
             'always_stock_correct':sum(r['oracle']=='stock' for r in out),
             'always_overlap_correct':sum(r['oracle']!='stock' for r in out),
             'always_overlap_gmean':gm([r['stock_b2b']/r['dx_b2b'] for r in out])}
    report={'classification':'exploratory independent batch, entire physical K/N family held out',
            'warning':'component times measured at target shape; data previously used in development',
            'train_sources':src,'test_sources':tsrc,'signature':sig,'summary':summary,'rows':out}
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
