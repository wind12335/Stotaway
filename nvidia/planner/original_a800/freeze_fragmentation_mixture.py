"""Freeze route models before collecting target component data or labels."""
import argparse
import hashlib
import json
from pathlib import Path
from bench.tp2_cells import shapes
from .evaluate_fragmentation_mixture import join
from .fragmentation_mixture import FragmentationMixture


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--shapes',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--m-values',type=int,nargs='+',default=[1024,4096])
    ap.add_argument('--scale-floor',type=float,default=0.)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--piecewise',action='store_true');group.add_argument('--reduced',action='store_true')
    group.add_argument('--local-residual',action='store_true');a=ap.parse_args()
    rows,sig,ops,stages=join(a.operators,a.stages)
    inventory={(r['op'],r['K'],r['N']) for r in rows};raw_shapes=a.shapes.read_bytes();vehicles=json.loads(raw_shapes)
    if not a.m_values or len(set(a.m_values)) != len(a.m_values) or any(m<=0 or m%2 for m in a.m_values):
        raise ValueError('M values must be unique positive even integers')
    targets=[]
    for name,dims in vehicles.items():
        for site,direction,_,k,n in shapes(*dims,1024):
            op='rs' if direction=='rs' else 'ag_up' if site=='qkv' else 'ag_gu'
            if (op,k,n) in inventory:raise ValueError('target family in training')
            for m in a.m_values:targets.append({'model':name,'site':site,'op':op,'M':m,'N':n,'K':k})
    if a.local_residual:
        from .local_residual_policy import LocalResidualPolicy
        models={op:LocalResidualPolicy(scale_floor=a.scale_floor or .05).fit([r for r in rows if r['op']==op],op=op).to_dict() for op in sorted({r['op'] for r in rows})}
    elif a.reduced:
        from .reduced_probe_policy import ReducedProbePolicy
        models={op:ReducedProbePolicy(scale_floor=a.scale_floor).fit([r for r in rows if r['op']==op],op=op).to_dict() for op in sorted({r['op'] for r in rows})}
    elif a.piecewise:
        from .piecewise_gain_policy import PiecewiseGainPolicy
        models={op:PiecewiseGainPolicy(scale_floor=a.scale_floor).fit([r for r in rows if r['op']==op]).to_dict() for op in sorted({r['op'] for r in rows})}
    else:
        if a.scale_floor:raise ValueError('scale floor requires reduced or piecewise')
        models={op:FragmentationMixture().fit([r for r in rows if r['op']==op]).to_dict() for op in sorted({r['op'] for r in rows})}
    out={'schema':'stowaway-fragmentation-freeze-v1','signature':sig,'training_operator_sources':ops,
         'training_stage_sources':stages,'target_shape_sha256':hashlib.sha256(raw_shapes).hexdigest(),
         'models':models,'targets':targets,'warning':'models frozen before target component and overlap labels'}
    raw=(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(raw).hexdigest(),'targets':len(targets)},indent=2))


if __name__=='__main__':main()
