"""Compare one exact-shape correction with the uncorrected online plan."""
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
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--calibrated',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    wall=json.loads(a.wall.read_bytes());digests={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in [('base',a.base),('calibrated',a.calibrated)]}
    if [v['arm'] for v in wall]!=['base','calibrated','calibrated','base','base','calibrated','calibrated','base']:
        raise ValueError('wrong experimental sequence')
    runs=[]
    for entry in wall:
        p=Path(entry['log']);r=parse(p,3);r['arm']=entry['arm'];r['elapsed_seconds']=entry['elapsed_seconds']
        content=p.read_text(errors='replace')
        if any(f'STOWAWAY_TP2_ONLINE rank={i} sha256={digests[entry["arm"]]}' not in content for i in (0,1)):
            raise ValueError('missing online digest')
        runs.append(r)
    blocks=[]
    for i in range(0,len(runs),2):
        pair=runs[i:i+2];base=next(r for r in pair if r['arm']=='base');cal=next(r for r in pair if r['arm']=='calibrated')
        x={n:(t,l) for n,t,l in base['iterations'] if n>3};y={n:(t,l) for n,t,l in cal['iterations'] if n>3}
        if set(x)!=set(y):raise ValueError('iteration mismatch')
        ids=sorted(x)
        blocks.append({'sequence':[v['arm'] for v in pair], 'speedup':statistics.median(x[n][0]/y[n][0] for n in ids),
                       'median_time_speedup':base['after_warmup_median_ms']/cal['after_warmup_median_ms'],
                       'max_abs_loss_diff':max(abs(x[n][1]-y[n][1]) for n in ids),
                       'delta_ms':[y[n][0]-x[n][0] for n in ids]})
    report={'classification':'paired online MLP fc2 correction versus original operator model',
            'warning':'exploratory four-layer random mock, same-iteration pairing not a proof of equal state',
            'bundle_sha256':digests,'blocks':blocks,'runs':runs,
            'paired_speedup_gmean':math.exp(statistics.mean(math.log(b['speedup']) for b in blocks)),
            'run_median_speedup_gmean':math.exp(statistics.mean(math.log(b['median_time_speedup']) for b in blocks))}
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'paired_speedup_gmean':report['paired_speedup_gmean'],
                      'block_speedups':[b['speedup'] for b in blocks],
                      'run_median_speedup_gmean':report['run_median_speedup_gmean']},indent=2))


if __name__=='__main__':main()
