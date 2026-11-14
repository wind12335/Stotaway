"""Evaluate pre-frozen policies against new physical shape families."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np

from .validate_tp2_new import load
from .pipeline_gain_policy import PipelineGainPolicy
from .relative_model import RelativeModel


def metrics(rows):
    return {'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),
            'slower':sum(r['speedup']<1 for r in rows),
            'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in rows))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=a.frozen.read_bytes();frozen=json.loads(raw);test,sig,sources=load(a.test)
    # Collector currently uses three warmups but does not record that field;
    # retain the unknown signature instead of fabricating metadata.
    train_sig=frozen['signature']
    if list(sig)!=train_sig or sig[-1] is not None:
        raise ValueError('unexpected route/protocol')
    shape_predictions={(r['model'],r['site'],r['M'],r['K'],r['N']):r for r in frozen['precomputed_shape_predictions']}
    outputs={name:[] for name in ('shape','nested','pipeline','always_stock','always_overlap')}
    for row in test:
        identity=(row['source_model'],row['site'],row['M'],row['K'],row['N'])
        saved=shape_predictions[identity];model=frozen['models'][row['op']]
        shape_choice='overlap' if saved['shape_only_choice']!='stock' else 'stock'
        nested=model['nested_policy'];v=nested['model']
        params=(np.array(v['mu']),np.array(v['sd']),v['mean'],np.array(v['beta']),np.array(v['curve']),v['split'])
        score=RelativeModel._predict_op(params,row);nested_choice='overlap' if nested['kind']=='dx' or (nested['kind']=='threshold' and score>nested['threshold']) else 'stock'
        pl=model['pipeline_policy'];params=(np.array(pl['mu']),np.array(pl['sd']),np.array(pl['beta']),pl['mean'])
        estimate=PipelineGainPolicy._predict(params,row)
        pipeline_choice='overlap' if pl['kind']=='overlap' or (pl['kind']=='threshold' and estimate>pl['threshold']) else 'stock'
        actual=row['stock_b2b']/row['dx_b2b'];oracle='overlap' if actual>1 else 'stock'
        for name,choice in [('shape',shape_choice),('nested',nested_choice),('pipeline',pipeline_choice),('always_stock','stock'),('always_overlap','overlap')]:
            outputs[name].append({'identity':identity,'family':row['model'],'choice':choice,'oracle':oracle,
                                  'speedup':actual if choice=='overlap' else 1.,'actual_speedup':actual})
    if len(test)!=len(shape_predictions):raise ValueError('incomplete or duplicate target inventory')
    summary={name:metrics(rows) for name,rows in outputs.items()}
    # De-duplicate aliases by exact (direction,M,N,K), using median actual ratio
    # and requiring a single decision per policy for each physical point.
    unique={}
    for name,rows in outputs.items():
        groups={}
        for r in rows:
            op=r['family'].split(':')[0];ident=r['identity'];key=(op,ident[2],ident[3],ident[4]);groups.setdefault(key,[]).append(r)
        collapsed=[]
        for key,items in groups.items():
            if len({r['choice'] for r in items})!=1:raise ValueError('conflicting choices for same physical point')
            ratio=statistics.median(r['actual_speedup'] for r in items);choice=items[0]['choice']
            collapsed.append({'choice':choice,'oracle':'overlap' if ratio>1 else 'stock','speedup':ratio if choice=='overlap' else 1.})
        unique[name]=metrics(collapsed)
    result={'classification':'new physical families after frozen candidate models; exploratory TP2 forward operators',
            'warning':'training and test metadata omit warmup; collector code uses three test warmups; measured target G/C required for nested/pipeline; new families not full models',
            'frozen_sha256':hashlib.sha256(raw).hexdigest(),'test_sources':sources,'summary':summary,'physical_unique_summary':unique,'rows':outputs}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'summary':summary,'physical_unique_summary':unique},indent=2))


if __name__=='__main__':main()
