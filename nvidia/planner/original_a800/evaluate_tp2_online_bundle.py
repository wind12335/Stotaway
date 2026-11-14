"""Evaluate frozen operator bundle against a separate target-M batch."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from ..tp2_lookup import QWEN_SITES,key
from .tp2_online_bundle import OnlineBundle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--test',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    payload=json.loads(a.bundle.read_text());runtime=payload['runtime'];workload=payload['workload_requirements']
    bundle=OnlineBundle(a.bundle,runtime,workload)
    raw=a.test.read_bytes();data=json.loads(raw);md=data['metadata']
    warmups={route.warmup for route in bundle.routes.values()}
    if (any(md.get(k)!=v for k,v in runtime.items()) or
        md.get('M')!=workload['seq_length']*workload['micro_batch_size'] or md.get('batch')!=16 or
        len(warmups)!=1 or md.get('warmup')!=next(iter(warmups)) or md.get('checks',0)<1 or
        md.get('classification')!='diagnostic unfused site forward+backward'):
        raise ValueError('target protocol mismatch')
    predictions=bundle.choices([bundle.selector.predict(q.route,q.shape,capability=q.capability) for q in bundle.queries])
    rows=[]
    for r in data['results']:
        name=key(QWEN_SITES[r['site']],r['M'],r['K'],r['N'])
        choice=predictions.get(name,'stock')
        stock=r['measurements']['stock']['median_ms'];dx=r['measurements']['dx']['median_ms']
        rows.append({'key':name,'choice':choice,'stock_ms':stock,'dx_ms':dx,
                     'oracle':'dx' if dx<stock else 'stock','speedup':stock/dx if choice=='dx' else 1.})
    if len(rows)!=len(predictions) or {r['key'] for r in rows}!=set(predictions):raise ValueError('shape inventory differs')
    out={'classification':'independent target-M repeat batch; operator-only diagnostic',
         'warning':'target capability shapes previously measured and data previously inspected; not a blind test or training E2E',
         'bundle_sha256':bundle.digest,'test_source':str(a.test),'test_sha256':hashlib.sha256(raw).hexdigest(),
         'summary':{'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),
                    'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in rows)),
                    'slower':sum(r['speedup']<1 for r in rows),
                    'always_dx_gmean':math.exp(statistics.mean(math.log(r['stock_ms']/r['dx_ms']) for r in rows)),
                    'oracle_gmean':math.exp(statistics.mean(math.log(max(1,r['stock_ms']/r['dx_ms'])) for r in rows))},
         'rows':rows}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':main()
