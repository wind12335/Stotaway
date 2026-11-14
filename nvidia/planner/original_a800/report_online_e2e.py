"""Summarize paired stock/online TP2 diagnostics without masking loss drift."""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from .report_qwen36_tp2_e2e import parse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wall',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    wall=json.loads(a.wall.read_bytes());digest=hashlib.sha256(a.bundle.read_bytes()).hexdigest()
    if [x['arm'] for x in wall]!=['stock','online','online','stock','stock','online','online','stock']:
        raise ValueError('unexpected run sequence')
    runs=[]
    for item in wall:
        path=Path(item['log']);r=parse(path,3);r.update({'arm':item['arm'],'elapsed_seconds':item['elapsed_seconds']})
        text=path.read_text(errors='replace')
        if item['arm']=='online':
            times=[]
            for rank in (0,1):
                p=rf'STOWAWAY_TP2_ONLINE rank={rank} sha256={digest} queries=6 decision_ms=([0-9.]+)'
                found=re.search(p,text)
                if not found:raise ValueError('missing rank online decision')
                times.append(float(found[1]))
            r['decision_ms']=times
        runs.append(r)
    blocks=[]
    for i in range(0,8,2):
        pair=runs[i:i+2];stock=next(r for r in pair if r['arm']=='stock');online=next(r for r in pair if r['arm']=='online')
        s={n:(t,l) for n,t,l in stock['iterations'] if n>3};o={n:(t,l) for n,t,l in online['iterations'] if n>3}
        if set(s)!=set(o):raise ValueError('iteration mismatch')
        ids=sorted(s);speeds=[s[n][0]/o[n][0] for n in ids]
        blocks.append({'sequence':[r['arm'] for r in pair], 'paired_speedup':statistics.median(speeds),
                       'run_median_speedup':stock['after_warmup_median_ms']/online['after_warmup_median_ms'],
                       'loss_max_abs':max(abs(s[n][1]-o[n][1]) for n in ids),
                       'per_step_delta_ms':[o[n][0]-s[n][0] for n in ids]})
    out={'classification':'four paired TP2 Qwen four-layer random mock training diagnostics',
         'warning':'operator-only policy; matched iteration IDs do not prove identical execution state; no full-model claim',
         'bundle_sha256':digest,'blocks':blocks,'runs':runs,
         'paired_speedup_gmean':math.exp(statistics.mean(math.log(b['paired_speedup']) for b in blocks)),
         'run_median_speedup_gmean':math.exp(statistics.mean(math.log(b['run_median_speedup']) for b in blocks)),
         'startup_decision_ms':[r['decision_ms'] for r in runs if r['arm']=='online']}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:out[k] for k in ('paired_speedup_gmean','run_median_speedup_gmean','startup_decision_ms')},indent=2))


if __name__=='__main__':main()
