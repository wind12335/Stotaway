"""Evaluate pre-frozen component correction against independent target runs."""
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
    if sig[-1]!=3 or sig[0]!='dx' or sig[1]!=2 or sig[8]!=16:raise ValueError('protocol mismatch')
    choices={tuple(r['identity']):r for r in frozen['predictions']};rows=[]
    for row in test:
        key=(row['source_model'],row['site'],row['M'],row['K'],row['N']);pred=choices[key]
        speed=row['stock_b2b']/row['dx_b2b'];rows.append({'identity':key,'base':pred['base']['choice'],
            'corrected':pred['corrected']['choice'],'actual_speedup':speed,'oracle':'overlap' if speed>1 else 'stock'})
    if len(rows)!=len(choices) or len({r['identity'] for r in rows})!=len(rows):raise ValueError('inventory mismatch')
    def score(policy):
        speeds=[r['actual_speedup'] if (r[policy]=='overlap' if policy in ('base','corrected') else policy=='overlap') else 1. for r in rows]
        picks=[r[policy] if policy in ('base','corrected') else policy for r in rows]
        return {'count':len(rows),'correct':sum(x==r['oracle'] for x,r in zip(picks,rows)),
                'slower':sum(x<1 for x in speeds),'speedup_gmean':math.exp(statistics.mean(math.log(x) for x in speeds))}
    out={'classification':'prospective target families with pre-frozen component correction',
         'warning':'target GEMM components measured before decisions; cost excluded; model and heuristic developed on prior data; not E2E',
         'frozen_sha256':hashlib.sha256(raw).hexdigest(),'test_sources':sources,
         'summary':{p:score(p) for p in ('base','corrected','stock','overlap')},'rows':rows}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':main()
