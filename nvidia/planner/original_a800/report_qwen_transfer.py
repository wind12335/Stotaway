"""Analyze frozen target transfer against four separately measured arms."""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from .report_qwen36_tp2_e2e import parse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--logs',nargs=12,type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=a.profile.read_bytes();digest=hashlib.sha256(raw).hexdigest();profile=json.loads(raw)
    selected=profile['model_audit']['selection']['best']['sites']
    mapping={(): 'stock',('proj',):'proj',('fc2',):'fc2',('fc2','proj'):'pair'}
    choice=mapping[tuple(selected)]
    runs=[]
    for p in a.logs:
        match=re.search(r'_([0-9]{2})_(stock|proj|fc2|pair)_20261113\.log$',p.name)
        if not match:raise ValueError('unexpected log name')
        index,arm=int(match[1]),match[2]; rec=parse(p,3);rec['arm']=arm;rec['index']=index
        text=p.read_text(errors='replace')
        if arm!='stock' and not rec['counts']:raise ValueError('missing adapter counts')
        if arm=='fc2' and any(f'STOWAWAY_TP2_LOOKUP rank={rank} sha256={digest}' not in text for rank in (0,1)):
            raise ValueError('frozen plan not loaded by both ranks')
        runs.append(rec)
    runs.sort(key=lambda r:r['index'])
    if [r['index'] for r in runs] != list(range(12)):raise ValueError('expected complete sequence')
    blocks=[]
    for i in range(0,12,4):
        group=runs[i:i+4]; by={r['arm']:{n:(t,l) for n,t,l in r['iterations'] if n>3} for r in group}
        if set(by)!={'stock','proj','fc2','pair'}:raise ValueError('four arms required')
        ids=sorted(by['stock'])
        if any(set(v)!=set(ids) for v in by.values()):raise ValueError('iteration mismatch')
        loss_diff={m:max(abs(by[m][n][1]-by['stock'][n][1]) for n in ids) for m in by}
        deltas={m:[by[m][n][0]-by['stock'][n][0] for n in ids] for m in by}
        med={m:statistics.median(d) for m,d in deltas.items()}
        speeds={m:statistics.median(by['stock'][n][0]/by[m][n][0] for n in ids) for m in by}
        interaction=[by['pair'][n][0]-by['proj'][n][0]-by['fc2'][n][0]+by['stock'][n][0] for n in ids]
        oracle=min(med,key=lambda m:(med[m],m!='stock',m))
        blocks.append({'sequence':[r['arm'] for r in group],'iterations':ids,'deltas_ms':deltas,'median_delta_ms':med,
                       'paired_speedup':speeds,'interaction_median_ms':statistics.median(interaction),'oracle':oracle,
                       'max_absolute_printed_loss_difference':loss_diff,
                       'choice':choice,'run_median_speedup':next(r['after_warmup_median_ms'] for r in group if r['arm']=='stock')/
                                                          next(r['after_warmup_median_ms'] for r in group if r['arm']==choice)})
    baselines={m:{'geomean_speedup':math.exp(statistics.mean(math.log(b['paired_speedup'][m]) for b in blocks)),
                  'slower':sum(b['paired_speedup'][m]<1 for b in blocks)} for m in ('stock','proj','fc2','pair')}
    out={'classification':'exploratory cross-M transfer, independent target runs, 3 blocks, four-layer mock training',
         'warning':'target data previously available; matching iteration numbers does not establish identical inputs; printed loss differs for some arms; no blind-test or numerical-equivalence claim',
         'profile_sha256':digest,'selection':choice,'correct':sum(b['oracle']==choice for b in blocks),
         'blocks':blocks,'baselines':baselines,'runs':runs}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'selection':choice,'correct':out['correct'],'baselines':baselines,
                      'blocks':[{k:b[k] for k in ('sequence','median_delta_ms','interaction_median_ms','oracle','run_median_speedup')} for b in blocks]},indent=2))


if __name__=='__main__':main()
