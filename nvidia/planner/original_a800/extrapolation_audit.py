"""Audit distance of frozen low-cost model queries to their route's training data.

Descriptive diagnostic only: no calibrated uncertainty or automatic rejection.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
from .evaluate_fragmentation_mixture import join
from .local_residual_policy import feature


def audit(model_path,decision_path,eval_paths):
    model=json.loads(Path(model_path).read_bytes());decisions=json.loads(Path(decision_path).read_bytes())
    digest=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if decisions['model_sha256']!=digest:raise ValueError('model and decisions mismatch')
    ops=[Path(x['path']) for x in model['training_operator_sources']]
    stages=[Path(x['path']) for x in model['training_stage_sources']]
    for file,ref in zip(ops+stages,model['training_operator_sources']+model['training_stage_sources']):
        if hashlib.sha256(file.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('training source changed')
    train,signature,_,_=join(ops,stages)
    if list(signature)!=model['signature']:raise ValueError('training signature mismatch')
    observations={}
    for path in eval_paths:
        d=json.loads(Path(path).read_bytes())
        if d['frozen_sha256']!=hashlib.sha256(Path(decision_path).read_bytes()).hexdigest():
            raise ValueError('evaluation and frozen decisions mismatch')
        evaluation={}
        for row in d['rows']:
            identity=tuple(row['identity'])
            if identity in evaluation:raise ValueError('duplicate evaluation identity')
            speed=row['actual_speedup']
            if not isinstance(speed,(int,float)) or isinstance(speed,bool) or not math.isfinite(speed) or speed<=0:
                raise ValueError('invalid observed speedup')
            evaluation[identity]=speed
        for identity,speed in evaluation.items():observations.setdefault(identity,[]).append(speed)
    rows=[]
    for record in decisions['predictions']:
        query=record['query'];same=[r for r in train if r['op']==record['op']]
        matrix=np.stack([feature(r) for r in same]);q=feature(query)
        outside=[i for i in range(len(q)) if q[i]<matrix[:,i].min() or q[i]>matrix[:,i].max()]
        scale=np.maximum(matrix.std(0),[.25,.15,.15,.10])
        distances=np.linalg.norm((matrix-q)/scale,axis=1)
        speed=observations.get(tuple(record['identity']),[])
        rows.append({'identity':record['identity'],'op':record['op'],'outside_dimensions':outside,
                     'nearest_normalized_distance':float(distances.min()),
                     'predicted_log_speedup':record['prediction']['predicted_log_speedup'],
                     'choice':record['prediction']['choice'],'observed_speedups':speed})
    if len({tuple(r['identity']) for r in rows})!=len(rows) or set(observations)!={tuple(r['identity']) for r in rows}:
        raise ValueError('decision and evaluation inventories differ')
    if any(len(r['observed_speedups'])!=len(eval_paths) for r in rows):raise ValueError('incomplete evaluations')
    summary={}
    for name,group in [('inside',[r for r in rows if not r['outside_dimensions']]),
                       ('outside',[r for r in rows if r['outside_dimensions']])]:
        outcomes=[(r,s) for r in group for s in r['observed_speedups']]
        if not outcomes:summary[name]={'physical_points':len(group),'observations':0};continue
        selected=[s if r['choice']=='overlap' else 1. for r,s in outcomes]
        summary[name]={'physical_points':len(group),'observations':len(outcomes),
                       'correct':sum((r['choice']=='overlap')==(s>1) for r,s in outcomes),
                       'slower':sum(s<1 for s in selected),
                       'geomean_selected':math.exp(statistics.mean(math.log(s) for s in selected))}
    return {'classification':'post-hoc range and nearest-family distance diagnostic',
            'feature_order':['log_M','log_N','log_K','log_split_over_full'],
            'model_sha256':digest,'summary':summary,'rows':rows,
            'warning':'axis-aligned training range, not a probabilistic guarantee; repeated same-shape observations are not independent shapes; no threshold selected'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,type=Path);ap.add_argument('--decisions',required=True,type=Path)
    ap.add_argument('--evaluations',required=True,nargs='+',type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();result=audit(a.model,a.decisions,a.evaluations)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
