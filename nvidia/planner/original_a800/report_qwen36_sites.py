"""Summarize site-ablation training logs by explicit arm and seed run."""
import argparse
import json
import statistics
from pathlib import Path

from .report_qwen36_tp2_e2e import parse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('inputs',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    valid={'stock','qkv','proj','fc1','fc2'};groups={}
    for p in args.inputs:
        parts=p.name.split('_')
        if len(parts)<5 or parts[3] not in valid:raise ValueError(f'unrecognized arm: {p}')
        arm=parts[3];record=parse(p,3)
        if arm!='stock' and not record['counts']:raise ValueError('missing adapter counts')
        groups.setdefault(arm,[]).append(record)
    if set(groups)!=valid:raise ValueError('all five arms required')
    med={k:statistics.median(r['after_warmup_median_ms'] for r in v) for k,v in groups.items()}
    out={'classification':'Qwen3.6 four-layer TP2 mock training, site ablation diagnostic',
         'warning':'site switches do not independently distinguish GDN/MLP; sequential arms, limited batches',
         'warmup_steps':3,'median_ms':med,'speedup':{k:med['stock']/v for k,v in med.items()},'groups':groups}
    with args.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:out[k] for k in ('median_ms','speedup')},indent=2))


if __name__=='__main__':main()
