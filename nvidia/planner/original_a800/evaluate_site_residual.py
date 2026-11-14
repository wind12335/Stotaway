"""Strict outer family-held-out evaluation of site-aware shape-only model."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .site_residual_policy import SiteResidualPolicy
from .validate_tp2_new import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-report',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    report=json.loads(a.train_report.read_bytes());train,sig,refs=load([Path(s['path']) for s in report['train_sources']]);test,tsig,tsources=load(a.test)
    if sig!=tsig:raise ValueError('protocol mismatch')
    rows=[]
    for r in test:
        group=[x for x in train if x['op']==r['op'] and x['model']!=r['model']]
        model=SiteResidualPolicy().fit(group);p=model.predict(r);gain=r['stock_b2b']/r['dx_b2b']
        rows.append({'site':r['site'],'shape':[r['M'],r['N'],r['K']],'family':r['model'],'choice':p['choice'],
                     'oracle':'overlap' if gain>1 else 'stock','speedup':gain if p['choice']=='overlap' else 1.,
                     'actual_speedup':gain,'prediction':p})
    summary={'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),'slower':sum(r['speedup']<1 for r in rows),
             'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in rows))}
    out={'classification':'exploratory site-residual shape-only grouped holdout',
         'warning':'site semantics must match; target data may have informed development; no E2E claim',
         'train_sources':refs,'test_sources':tsources,'summary':summary,'rows':rows}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
