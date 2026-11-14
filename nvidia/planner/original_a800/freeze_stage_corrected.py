"""Freeze split-cost decisions before target overlap labels are measured."""
import argparse
import hashlib
import json
from pathlib import Path
from .site_residual_policy import SiteResidualPolicy,PipelineCorrectedSitePolicy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    model_raw=a.model.read_bytes();model=json.loads(model_raw);stage={};sources=[]
    for p in a.stages:
        raw=p.read_bytes();data=json.loads(raw);sources.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
        if data['metadata']['batch']!=16 or data['metadata']['warmup']!=3:raise ValueError('component timing mismatch')
        for r in data['results']:
            key=(r['model'],r['site'],r['M'],r['K'],r['N'])
            if key in stage:raise ValueError('duplicate component')
            stage[key]=r
    predictions=[]
    for r in model['predictions']:
        key=(r['model'],r['site'],r['M'],r['K'],r['N']);s=stage[key]['measurements']
        base=SiteResidualPolicy.from_dict(model['models'][r['op']]);policy=PipelineCorrectedSitePolicy(base)
        pred=policy.predict(r,full_ms=s['full']['median_ms'],split_ms=s['split']['median_ms'])
        predictions.append({'identity':key,'op':r['op'],'base':r['prediction'],'corrected':pred})
    if len(predictions)!=len(stage):raise ValueError('inventory mismatch')
    out={'schema':'stowaway-stage-corrected-freeze-v1','source_model_sha256':hashlib.sha256(model_raw).hexdigest(),
         'sources':sources,'coefficient':1.,'predictions':predictions,
         'warning':'fixed heuristic selected after previous dataset; target component measurements are not free'}
    raw=(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(raw).hexdigest(),'predictions':len(predictions)},indent=2))


if __name__=='__main__':main()
