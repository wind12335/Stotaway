"""Assess pre-frozen component-based decisions against newly collected DX."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .validate_tp2_new import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--posthoc',action='store_true',help='target measurements were available before decision freeze');a=ap.parse_args()
    raw=a.frozen.read_bytes();frozen=json.loads(raw);test,sig,sources=load(a.test)
    if list(sig)!=frozen['signature']:raise ValueError('runtime signature mismatch')
    predictions={tuple(p['identity']):p for p in frozen['predictions']};rows=[]
    for r in test:
        key=(r['source_model'],r['site'],r['M'],r['N'],r['K']);p=predictions[key]
        ratio=r['stock_b2b']/r['dx_b2b'];rows.append({'identity':key,'op':r['op'],'choice':p['prediction']['choice'],
          'oracle':'overlap' if ratio>1 else 'stock','actual_speedup':ratio,'component_wall_ms':p['component_wall_ms'],
          'predicted_log_speedup':p['prediction']['predicted_log_speedup']})
    if len(rows)!=len(predictions) or len({tuple(r['identity']) for r in rows})!=len(rows):raise ValueError('inventory mismatch')
    def stats(group,method):
        choices=[r['choice'] if method=='model' else method for r in group]
        speeds=[r['actual_speedup'] if c=='overlap' else 1. for c,r in zip(choices,group)]
        return {'count':len(group),'correct':sum(c==r['oracle'] for c,r in zip(choices,group)),
                'slower':sum(s<1 for s in speeds),'speedup_gmean':math.exp(statistics.mean(math.log(s) for s in speeds))}
    out={'classification':('retrospective evaluation; target labels predate decisions' if a.posthoc else
                           'new physical shape families, component decisions frozen before target DX measurement'),
         'warning':('target timings already inspected; ' if a.posthoc else '')+
                   'component cost excluded from speedup, target baseline measured separately; forward TP2 only',
         'frozen_sha256':hashlib.sha256(raw).hexdigest(),'test_sources':sources,
         'summary':{p:stats(rows,p) for p in ('model','stock','overlap')},
         'by_M':{str(m):{p:stats([r for r in rows if r['identity'][2]==m],p) for p in ('model','stock','overlap')} for m in sorted({r['identity'][2] for r in rows})},
         'median_component_wall_ms':statistics.median(r['component_wall_ms'] for r in rows),'rows':rows}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:out[k] for k in ('summary','by_M','median_component_wall_ms')},indent=2))


if __name__=='__main__':main()
