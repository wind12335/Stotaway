"""FlashOverlap independent repeat, held-out N/K family and matching protocol."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from validate_flash_tp2 import fit
from predictor_registry import bind


def read(path):
    raw = path.read_bytes(); d = json.loads(raw)
    sig = tuple(d[k] for k in ('method','direction','tp','dtype','batch','gpu','torch')) + (d.get('warmup',4),d['repeats'])
    route = bind({'method':d['method'], 'actual_path':'cutlass_nccl_wave', 'direction':d['direction'], 'fallback':False},
                 gpu=d['gpu'], topology='nv8', tp=d['tp'], dtype=d['dtype'],
                 versions=(('torch',str(d['torch'])),), timing_mode='b2b16', warmup=d.get('warmup',4),
                 config_family='predictive_search')
    row = {k:d[k] for k in ('M','N','K')} | {'stock':d['measurements']['stock']['median_ms'],
           'overlap':d['measurements']['overlap']['median_ms'], 'route':route, 'source':str(path)}
    return sig, row, {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',nargs='+',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();train=[read(p) for p in args.train];test=[read(p) for p in args.test]
    if len({r[0] for r in train+test})!=1:raise ValueError('different measurement signatures')
    rows=[]
    for _,r,_ in test:
        group=[v for _,v,_ in train if (v['N'],v['K']) != (r['N'],r['K'])]
        if len(group)<2: raise ValueError('insufficient training data')
        pred=fit(group)(r);pick='overlap' if pred>0 else 'stock'
        rows.append({**{k:v for k,v in r.items() if k!='route'},'pred_log_speedup':pred,'choice':pick,
                     'oracle':'overlap' if r['overlap']<r['stock'] else 'stock',
                     'speedup':r['stock']/r['overlap'] if pick=='overlap' else 1.0})
    summary={'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),
             'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in rows)),
             'slower':sum(r['speedup']<1 for r in rows),
             'always_overlap_correct':sum(r['oracle']=='overlap' for r in rows),
             'always_overlap_gmean':math.exp(statistics.mean(math.log(r['stock']/r['overlap']) for r in rows)),
             'always_stock_correct':sum(r['oracle']=='stock' for r in rows)}
    result={'classification':'exploratory independent repeat; leave all N/K family out',
            'train_sources':[r[2] for r in train], 'test_sources':[r[2] for r in test],
            'signature':train[0][0], 'summary':summary, 'rows':rows}
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
