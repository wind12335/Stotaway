"""Read-only SHA verification for reduced forward plans and their source chain.

A verified hash proves file identity only; it does not validate GPU support,
performance, actual topology, numerical correctness, or E2E speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(plan_path, model_path):
    plan=json.loads(Path(plan_path).read_bytes());model=json.loads(Path(model_path).read_bytes())
    if plan.get('plan_source')!='reduced-probe-model' or plan.get('phase')!='forward':
        raise ValueError('reduced forward plan required')
    if plan.get('model_sha256')!=digest(model_path):raise ValueError('model hash mismatch')
    if model.get('schema')!='stowaway-fragmentation-freeze-v1':raise ValueError('invalid model schema')
    refs=[]
    inventories=(('training_operator',model.get('training_operator_sources',[])),
                 ('training_component',model.get('training_stage_sources',[])),
                 ('target_component',plan.get('component_sources',[])))
    if any(not isinstance(items,list) or not items for _,items in inventories):
        raise ValueError('all three source inventories required')
    for kind,items in inventories:
        for item in items:
            path=Path(item['path']);actual=digest(path);expected=item['sha256']
            refs.append({'kind':kind,'path':str(path),'expected_sha256':expected,'actual_sha256':actual,
                         'match':actual==expected})
    if any(not r['match'] for r in refs):raise ValueError('source file hash mismatch')
    target=plan['target_model'];w=plan['workload_requirements'];m=w['seq_length']*w['micro_batch_size']
    targets=[r for r in model['targets'] if r['model']==target and r['M']==m]
    if not targets:raise ValueError('plan target absent from model')
    return {'classification':'reduced-forward-plan source-chain integrity audit',
            'plan_path':str(plan_path),'plan_sha256':digest(plan_path),
            'model_path':str(model_path),'model_sha256':digest(model_path),
            'target_model':target,'target_M':m,'target_count':len(targets),'verified_sources':refs,
            'warning':'hashes only establish artifact identity; they do not prove topology or performance'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path);ap.add_argument('--model',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();r=audit(a.plan,a.model)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({'plan_sha256':r['plan_sha256'],'model_sha256':r['model_sha256'],
                      'verified_sources':len(r['verified_sources']),'target_count':r['target_count']},indent=2))


if __name__=='__main__':main()
