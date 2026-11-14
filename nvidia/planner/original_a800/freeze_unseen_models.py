"""Freeze candidate models and physical family inventory before GPU collection."""
import argparse
import hashlib
import json
from pathlib import Path

from .validate_tp2_new import load
from .validate_orthogonal_tp2 import route_for
from .route_policy import GroupedSample,RoutePolicy
from .orthogonal_predictor import Sample,RouteRegressor
from .pipeline_gain_policy import PipelineGainPolicy
from .tp2_nested_policy import select_policy
from .relative_model import RelativeModel


def pack(params):
    mu,sd,mean,beta,curve,split=params
    return {'mu':mu.tolist(),'sd':sd.tolist(),'mean':mean,'beta':beta.tolist(),'curve':curve.tolist(),'split':split}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-report',type=Path,required=True)
    ap.add_argument('--shapes',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    source=json.loads(a.source_report.read_bytes());paths=[Path(s['path']) for s in source['train_sources']]
    rows,sig,refs=load(paths);models={};predicted=[]
    shape_raw=a.shapes.read_bytes();shapes=json.loads(shape_raw)
    from bench.tp2_cells import shapes as make_shapes
    training_families={(r['op'],r['K'],r['N']) for r in rows}
    for op in sorted({r['op'] for r in rows}):
        group=[r for r in rows if r['op']==op];route=route_for(sig,op)
        samples=[GroupedSample(r['model'],Sample(route,(r['M'],r['N'],r['K']),r['stock_b2b'],r['dx_b2b'])) for r in group]
        shape_policy=RoutePolicy(route).fit(samples)
        kind,threshold,audit=select_policy(group,2.)
        pipeline=PipelineGainPolicy().fit(group)
        mu,sd,beta,mean=pipeline.params
        models[op]={'shape_policy':shape_policy.to_dict(), 'nested_policy':{'kind':kind,'threshold':threshold,'audit':audit,
                       'model':pack(RelativeModel(mode='b2b')._fit_op(group))},
                    'pipeline_policy':{'kind':pipeline.policy,'threshold':pipeline.threshold,'audit':pipeline.audit,
                        'mu':mu.tolist(),'sd':sd.tolist(),'beta':beta.tolist(),'mean':mean}}
        for model,dims in shapes.items():
            for site,kind_op,m,k,n in make_shapes(*dims,1024):
                logical='rs' if kind_op=='rs' else ('ag_up' if site=='qkv' else 'ag_gu')
                if logical!=op:continue
                if (op,k,n) in training_families:raise ValueError('new shape overlaps training family')
                for m in (1024,4096):
                    result=shape_policy.predict(route,(m,n,k))
                    predicted.append({'model':model,'site':site,'op':op,'M':m,'K':k,'N':n,
                                      'shape_only_choice':result['choice'],'shape_prediction':result})
    encoded=(json.dumps({'schema':'stowaway-unseen-freeze-v1','signature':sig,'training_sources':refs,
                         'shape_source':{'path':str(a.shapes),'sha256':hashlib.sha256(shape_raw).hexdigest()},
                         'models':models,'precomputed_shape_predictions':predicted,
                         'warning':'frozen before new target timing; prior training data inspected; component models need measured G/C'},
                        indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(encoded)
    print(json.dumps({'sha256':hashlib.sha256(encoded).hexdigest(),'rows':len(predicted),'path':str(a.output)},indent=2))


if __name__=='__main__':main()
