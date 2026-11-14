"""Evaluate existing TP2 operator policies on Qwen3.6 static projection shapes.

No Qwen label is used for training. Each test excludes the same direction/K/N
from the training profiles; forward labels do not evaluate training E2E.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from validate_tp2_new import load
from relative_model import RelativeModel
from tp2_nested_policy import select_policy


SITE_OP = {'full_qkv':'ag_up', 'full_proj':'rs', 'gdn_in':'ag_up',
           'gdn_out':'rs', 'mlp_fc1':'ag_gu', 'mlp_fc2':'rs'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',nargs='+',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); train,sig,sources=load(args.train)
    tests=[]
    for p in args.test:
        d=json.loads(p.read_text());m=d['metadata']
        keys=('tp','dtype','gpu','torch','triton','cuda_build','cuda_device_max_connections','batch')
        actual=('dx',)+tuple(m[k] for k in keys)+('','')
        if actual!=sig or m['checks']<1:raise ValueError('Qwen timing signature mismatch')
        for v in d['results']:
            if v['site'] not in SITE_OP:raise ValueError('unknown site')
            z=v['measurements'];op=SITE_OP[v['site']]
            tests.append({'model':f"{v['op']}:{v['K']}:{v['N']}", 'site':v['site'],'op':op,
                          'M':v['M'],'K':v['K'],'N':v['N'],'G':z['gemm']['median_ms'],
                          'C':z['comm']['median_ms'],'B':v['payload_MB'],
                          'stock_b2b':z['stock']['median_ms'],'dx_b2b':z['dx']['median_ms'],
                          'source':str(p)})
    rows=[]
    for r in tests:
        eligible=[x for x in train if x['model']!=r['model']];subset=[x for x in eligible if x['op']==r['op']]
        prior=RelativeModel(mode='b2b').fit(eligible)
        try: old,_=prior.choose(r)
        except ValueError: old='stock'
        policy,threshold,_=select_policy(subset,2.0)
        pred=(RelativeModel._predict_op(RelativeModel(mode='b2b')._fit_op(subset),r)
              if policy=='threshold' else None)
        pick='dx' if policy=='dx' or (policy=='threshold' and pred>threshold) else 'stock'
        rows.append({**r,'oracle':'dx' if r['dx_b2b']<r['stock_b2b'] else 'stock',
                     'nested_pick':pick,'relative_pick':old,'always_dx':'dx','always_stock':'stock'})
    def summary(key):
        speeds=[r['stock_b2b']/(r['dx_b2b'] if r[key]=='dx' else r['stock_b2b']) for r in rows]
        return {'count':len(rows),'correct':sum(r[key]==r['oracle'] for r in rows),
                'gmean_speedup':math.exp(statistics.mean(map(math.log,speeds))),
                'slower':sum(x<1 for x in speeds)}
    result={'classification':'Qwen3.6 static projection, forward operator, exploratory holdout',
            'warning':'not Qwen3.6 training E2E; M values and matching projection sites are correlated',
            'training_sources':sources,'test_sources':[str(p) for p in args.test],
            'summary':{k:summary(k) for k in ('relative_pick','nested_pick','always_dx','always_stock')},'rows':rows}
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
