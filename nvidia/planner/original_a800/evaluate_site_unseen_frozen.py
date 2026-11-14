"""Evaluate frozen shape-only candidate against newly measured families."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .validate_tp2_new import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=a.frozen.read_bytes();frozen=json.loads(raw);test,sig,sources=load(a.test)
    if list(sig[:-1])!=frozen['signature'][:-1] or sig[-1]!=3 or frozen['signature'][-1] is not None:
        raise ValueError('unexpected runtime or warmup')
    predicted={(r['model'],r['site'],r['M'],r['K'],r['N']):r for r in frozen['predictions']}
    output=[]
    for row in test:
        identity=(row['source_model'],row['site'],row['M'],row['K'],row['N']);p=predicted[identity]['prediction']
        actual=row['stock_b2b']/row['dx_b2b'];choice=p['choice']
        output.append({'identity':identity,'family':row['model'],'choice':choice,
                       'oracle':'overlap' if actual>1 else 'stock','actual_speedup':actual,
                       'speedup':actual if choice=='overlap' else 1.,'prediction':p})
    if len(output)!=len(predicted) or len({tuple(r['identity']) for r in output})!=len(output):raise ValueError('inventory mismatch')
    def metrics(rows,pick):
        speeds=[r['actual_speedup'] if (r['choice']=='overlap' if pick=='model' else pick=='overlap') else 1. for r in rows]
        choices=[r['choice'] if pick=='model' else pick for r in rows]
        return {'count':len(rows),'correct':sum(c==r['oracle'] for c,r in zip(choices,rows)),
                'slower':sum(s<1 for s in speeds),'speedup_gmean':math.exp(statistics.mean(math.log(s) for s in speeds))}
    summary={pick:metrics(output,pick) for pick in ('model','stock','overlap')}
    out={'classification':'frozen shape-only candidate, new physical families, independent target collection',
         'warning':'training warmup metadata missing, target warmup=3; candidate designed after earlier test set; not E2E',
         'frozen_sha256':hashlib.sha256(raw).hexdigest(),'test_sources':sources,'summary':summary,'rows':output}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
