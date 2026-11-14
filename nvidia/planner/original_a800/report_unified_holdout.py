"""Report fixed two-arm alternating paired training diagnostics."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .report_qwen36_tp2_e2e import parse


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stock',nargs='+',type=Path,required=True)
    ap.add_argument('--plan',nargs='+',type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if len(a.stock)!=len(a.plan) or not a.stock:raise ValueError('paired runs required')
    digest=hashlib.sha256(a.profile.read_bytes()).hexdigest(); blocks=[]
    for stock_path,plan_path in zip(a.stock,a.plan):
        stock,plan=parse(stock_path,3),parse(plan_path,3)
        raw=plan_path.read_text(errors='replace')
        if any(f'STOWAWAY_TP2_LOOKUP rank={rank} sha256={digest}' not in raw for rank in (0,1)):
            raise ValueError('plan not loaded by both ranks')
        s={i:(t,l) for i,t,l in stock['iterations'] if i>3};p={i:(t,l) for i,t,l in plan['iterations'] if i>3}
        if set(s)!=set(p) or not s:raise ValueError('iteration mismatch')
        if any(not math.isclose(s[i][1],p[i][1],rel_tol=1e-6,abs_tol=1e-6) for i in s):
            raise ValueError('printed losses mismatch')
        ratios=[s[i][0]/p[i][0] for i in sorted(s)]
        blocks.append({'stock':stock,'plan':plan,'paired_speedup_median':statistics.median(ratios),
                       'ratio_of_run_medians':stock['after_warmup_median_ms']/plan['after_warmup_median_ms'],
                       'paired_deltas_ms':[p[i][0]-s[i][0] for i in sorted(s)]})
    geo=lambda key:math.exp(statistics.mean(math.log(b[key]) for b in blocks))
    out={'classification':'independent paired TP2 four-layer random-init mock training',
         'warning':'only three blocks, rounded equal losses do not prove identical execution state; not full Qwen 27B',
         'profile_sha256':digest,'blocks':blocks,'paired_speedup_gmean':geo('paired_speedup_median'),
         'run_median_speedup_gmean':geo('ratio_of_run_medians')}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'paired_speedup_gmean':out['paired_speedup_gmean'],
                      'run_median_speedup_gmean':out['run_median_speedup_gmean'],
                      'paired_block_speedups':[b['paired_speedup_median'] for b in blocks]},indent=2))


if __name__=='__main__':main()
