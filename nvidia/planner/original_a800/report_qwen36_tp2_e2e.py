"""Summarize explicitly provided Qwen TP2 diagnostic run logs."""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


def parse(path, warmup):
    raw=path.read_bytes();text=raw.decode(errors='replace')
    pattern=r'iteration\s+(\d+)/\s*\d+\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?lm loss:\s*([\d.eE+-]+)'
    iteration_lines=re.findall(r'iteration\s+(\d+)/\s*\d+\s*\|[^\n]*elapsed time per iteration \(ms\):[^\n]*',text)
    values=[(int(i),float(t),float(loss)) for i,t,loss in re.findall(pattern,text)]
    selected=[v for v in values if v[0]>warmup]
    if (not selected or len(iteration_lines) != len(values) or len({v[0] for v in values}) != len(values)
            or any(not math.isfinite(v) for _,t,l in values for v in (t,l))):
        raise ValueError(f'missing or invalid training iterations: {path}')
    if '[after training is done]' not in text:
        raise ValueError(f'incomplete run: {path}')
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'iterations':values,
            'after_warmup_median_ms':statistics.median(t for _,t,_ in selected),
            'choices':re.findall(r'STOWAWAY_TP2_CHOICE rank=0[^\n]*',text),
            'counts':re.findall(r'STOWAWAY_DIAGNOSTIC_COUNTS rank=0[^\n]*',text)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stock',nargs='+',type=Path,required=True)
    ap.add_argument('--plan',nargs='+',type=Path,required=True);ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    groups={name:[parse(p,a.warmup) for p in paths] for name,paths in [('stock',a.stock),('plan',a.plan)]}
    if any(not r['counts'] for r in groups['plan']):raise ValueError('missing adapter diagnostics')
    stock=statistics.median(r['after_warmup_median_ms'] for r in groups['stock'])
    plan=statistics.median(r['after_warmup_median_ms'] for r in groups['plan'])
    out={'classification':'TP2 Qwen3.6 four-layer local-spec random-init mock training diagnostic',
         'warning':'same-shape lookup calibrated on operator timings, not full-model 27B or independent E2E training',
         'warmup_steps':a.warmup,'stock_median_ms':stock,'plan_median_ms':plan,'speedup':stock/plan,
         'groups':groups}
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:out[k] for k in ('classification','stock_median_ms','plan_median_ms','speedup')},indent=2))


if __name__=='__main__':main()
