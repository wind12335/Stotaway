"""Evaluate pipeline features against independent DX repeat and TD leaveout."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .pipeline_gain_policy import PipelineGainPolicy
from .validate_tp2_new import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dx-report',type=Path,required=True)
    ap.add_argument('--triton-report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sections={}
    for method,path in [('dx',a.dx_report),('triton_dist',a.triton_report)]:
        report=json.loads(path.read_bytes())
        if method=='dx':
            tr,sig,_=load([Path(s['path']) for s in report['train_sources']]);te,tsig,_=load([Path(s['path']) for s in report['test_sources']])
            if sig!=tsig:raise ValueError('protocol mismatch')
        else:tr,sig,_=load([Path(s['path']) for s in report['sources']]);te=tr
        for op in sorted({r['op'] for r in te}):
            train=[r for r in tr if r['op']==op];test=[r for r in te if r['op']==op];rows=[]
            for target in test:
                model=PipelineGainPolicy().fit([r for r in train if r['model']!=target['model']])
                p=model.predict(target);actual=target['stock_b2b']/target['dx_b2b']
                rows.append({'shape':[target['M'],target['N'],target['K']],'family':target['model'],
                             'choice':p['choice'],'oracle':'overlap' if actual>1 else 'stock',
                             'speedup':actual if p['choice']=='overlap' else 1.,'prediction':p,'audit':model.audit})
            sections[method+'_'+op]={'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),
                'slower':sum(r['speedup']<1 for r in rows),
                'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in rows)),'rows':rows}
    out={'classification':'exploratory grouped pipeline-feature nested policy',
         'warning':'target shape G and C measured, measurement cost excluded; prior development data, not blind',
         'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (a.dx_report,a.triton_report)],'sections':sections}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:{x:v[x] for x in ('count','correct','speedup_gmean','slower')} for k,v in sections.items()},indent=2))


if __name__=='__main__':main()
