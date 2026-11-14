"""Analyze explicitly ordered four-arm step blocks and preserve raw provenance."""
import argparse
import json
import statistics
from pathlib import Path

from .report_qwen36_tp2_e2e import parse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('inputs',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    parsed=[]
    for p in sorted(a.inputs,key=lambda x:int(x.stem.split('_')[-1])):
        arm=p.stem.split('_')[3]
        if arm not in ('stock','proj','fc2','pair'):raise ValueError(f'unknown arm {p}')
        record=parse(p,3);record['arm']=arm
        if arm!='stock' and not record['counts']:raise ValueError('missing counts')
        parsed.append(record)
    if len(parsed)%4:raise ValueError('complete four-arm blocks required')
    blocks=[]
    for i in range(0,len(parsed),4):
        group=parsed[i:i+4];values={r['arm']:r['after_warmup_median_ms'] for r in group}
        if set(values)!={'stock','proj','fc2','pair'}:raise ValueError('each block needs four arms')
        j=values['pair']-values['proj']-values['fc2']+values['stock']
        blocks.append({'sequence':[r['arm'] for r in group],'ms':values,'interaction_ms':j,
                       'delta_proj':values['proj']-values['stock'],'delta_fc2':values['fc2']-values['stock'],
                       'delta_pair':values['pair']-values['stock']})
    med=lambda field:statistics.median(b[field] for b in blocks)
    out={'classification':'Qwen3.6 TP2 four-layer M2048 diagnostic, blocked interaction study',
         'warning':'same adapter fc2 controls GDN and MLP; no causal attribution; few blocks',
         'blocks':blocks,'median':{k:med(k) for k in ('interaction_ms','delta_proj','delta_fc2','delta_pair')},
         'runs':parsed}
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'blocks':blocks,'median':out['median']},indent=2))


if __name__=='__main__':main()
