"""Freeze target predictions from components before measuring overlap."""
import argparse
import hashlib
import json
from pathlib import Path
from .fragmentation_mixture import FragmentationMixture


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--components',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw_model=a.model.read_bytes();model=json.loads(raw_model);sig=model['signature'];components={};sources=[]
    for p in a.components:
        raw=p.read_bytes();data=json.loads(raw);md=data['metadata']
        if (md['tp'],md['batch'],md['warmup'],md['gpu'],md['torch'],md['triton'],md['cuda_build'],md['cuda_device_max_connections'])!=(sig[1],sig[8],sig[-1],sig[3],sig[4],sig[5],sig[6],sig[7]):
            raise ValueError('component protocol mismatch')
        sources.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
        for r in data['results']:
            key=(r['model'],r['site'],r['M'],r['N'],r['K'])
            if key in components:raise ValueError('duplicate component')
            components[key]=r
    results=[]
    for t in model['targets']:
        key=(t['model'],t['site'],t['M'],t['N'],t['K']);r=components[key];x=r['measurements']
        payload=model['models'][t['op']]
        query={'M':t['M'],'N':t['N'],'K':t['K'],'full_ms':x['full']['median_ms'],'split_ms':x['split']['median_ms']}
        if payload['schema']=='stowaway-local-residual-v1':
            from .local_residual_policy import LocalResidualPolicy
            query['op']=t['op'];policy=LocalResidualPolicy.from_dict(payload)
        elif payload['schema']=='stowaway-reduced-probe-v1':
            from .reduced_probe_policy import ReducedProbePolicy
            query['op']=t['op'];policy=ReducedProbePolicy.from_dict(payload)
        elif payload['schema']=='stowaway-piecewise-gain-v1':
            from .piecewise_gain_policy import PiecewiseGainPolicy
            policy=PiecewiseGainPolicy.from_dict(payload)
        else:policy=FragmentationMixture.from_dict(payload)
        if payload['schema'] not in ('stowaway-reduced-probe-v1','stowaway-local-residual-v1'):
            query.update({'comm_ms':x['comm']['median_ms'],'stock_ms':x['stock']['median_ms']})
        prediction=policy.predict(query)
        results.append({'identity':key,'op':t['op'],'query':query,'prediction':prediction,'component_wall_ms':r['elapsed_wall_ms']})
    if len(results)!=len(components):raise ValueError('inventory mismatch')
    out={'schema':'stowaway-fragmentation-decisions-v1','signature':sig,'model_sha256':hashlib.sha256(raw_model).hexdigest(),
         'component_sources':sources,'predictions':results,'warning':'frozen before measuring target DX; probe cost excluded from speedup'}
    raw=(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(raw).hexdigest(),'predictions':len(results)},indent=2))


if __name__=='__main__':main()
