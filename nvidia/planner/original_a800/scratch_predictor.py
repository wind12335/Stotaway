"""Freeze/evaluate TP2 RS scratch models with exact protocol and data isolation.

The target phase uses only full and split GEMM before label collection. The
scratch path is intentionally incompatible with the older DX plan schema.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .local_residual_policy import LocalResidualPolicy
from .reduced_probe_policy import ReducedProbePolicy


def raw(path):
    data=Path(path).read_bytes();return json.loads(data),hashlib.sha256(data).hexdigest()


def identity(row):return (row['model'],row['site'],row['M'],row['N'],row['K'])


def protocol(md,*,operators):
    fields=('tp','gpu','torch','triton','cuda_build','cuda_device_max_connections','batch','warmup','rs_actual_path')
    sig={k:md[k] for k in fields}
    if sig['tp']!=2 or sig['batch']!=16 or sig['warmup']!=3 or sig['rs_actual_path']!='direct_exchange_rs_scratch':
        raise ValueError('scratch TP2 b2b16 warmup3 protocol required')
    if operators and (md.get('dtype')!='bfloat16' or md.get('checks',0)<1 or not md.get('rs_scratch')):
        raise ValueError('checked BF16 scratch operator measurements required')
    return sig


def rows(paths,*,operators,signature=None):
    out={};sources=[]
    for path in paths:
        report,sha=raw(path);sig=protocol(report['metadata'],operators=operators)
        if signature is not None and sig!=signature:raise ValueError('measurement signature mismatch')
        signature=sig;sources.append({'path':str(path),'sha256':sha})
        for row in report['results']:
            key=identity(row)
            if key in out or row['op']!='rs' or row['site'] not in ('o_proj','fc2'):
                raise ValueError('duplicate or invalid RS record')
            out[key]=row
    if not out:raise ValueError('empty inventory')
    return out,signature,sources


def freeze(train_operators,train_stages,target_stages):
    ops,sig,ops_sources=rows(train_operators,operators=True)
    stages,_,stage_sources=rows(train_stages,operators=False,signature=sig)
    targets,_,target_sources=rows(target_stages,operators=False,signature=sig)
    if set(ops)!=set(stages):raise ValueError('training inventory mismatch')
    training_families={(x[3],x[4]) for x in ops}
    if any((x[3],x[4]) in training_families for x in targets):raise ValueError('target physical family in training')
    data=[]
    for key,row in ops.items():
        measurements=row['measurements'];features=stages[key]['measurements']
        data.append({'family':f'rs:{row["K"]}:{row["N"]}','op':'rs','M':row['M'],'N':row['N'],'K':row['K'],
                     'full_ms':features['full']['median_ms'],'split_ms':features['split']['median_ms'],
                     'comm_ms':measurements['comm']['median_ms'],'stock_ms':measurements['stock']['median_ms'],
                     'overlap_ms':measurements['dx']['median_ms']})
    model=LocalResidualPolicy().fit(data,op='rs');predictions=[]
    for key,row in sorted(targets.items()):
        f=row['measurements'];query={'op':'rs','M':row['M'],'N':row['N'],'K':row['K'],
                     'full_ms':f['full']['median_ms'],'split_ms':f['split']['median_ms']}
        predictions.append({'identity':key,'prediction':model.predict(query),'query':query,
                            'component_wall_ms':row['elapsed_wall_ms']})
    return {'schema':'stowaway-scratch-rs-frozen-v1','signature':sig,'model':model.to_dict(),
            'training_operator_sources':ops_sources,'training_stage_sources':stage_sources,
            'target_component_sources':target_sources,'predictions':predictions,
            'warning':'operator-only scratch RS, frozen before target timing; component costs excluded'}


def evaluate(frozen_path,target_paths):
    frozen,sha=raw(frozen_path)
    if frozen.get('schema')!='stowaway-scratch-rs-frozen-v1':raise ValueError('wrong model schema')
    targets,_,sources=rows(target_paths,operators=True,signature=frozen['signature'])
    predictions={tuple(p['identity']):p for p in frozen['predictions']}
    if set(targets)!=set(predictions):raise ValueError('target inventory mismatch')
    results=[]
    for key,row in targets.items():
        prediction=predictions[key]['prediction'];m=row['measurements'];speed=m['stock']['median_ms']/m['dx']['median_ms']
        results.append({'identity':key,'choice':prediction['choice'],'actual_speedup':speed,
                        'predicted_log_speedup':prediction['predicted_log_speedup']})
    def score(arm):
        choices=[r['choice'] if arm=='model' else arm for r in results]
        values=[r['actual_speedup'] if c=='overlap' else 1. for r,c in zip(results,choices)]
        return {'count':len(values),'correct':sum((c=='overlap')==(r['actual_speedup']>1) for r,c in zip(results,choices)),
                'slower':sum(x<1 for x in values),'speedup_gmean':math.exp(statistics.mean(math.log(x) for x in values))}
    return {'classification':'frozen scratch RS predictor against later operator measurements',
            'frozen_sha256':sha,'target_sources':sources,'summary':{x:score(x) for x in ('model','stock','overlap')},
            'rows':results,'warning':'physical shapes previously studied with old DX; component and initialization cost excluded; no E2E'}


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--train-operators',type=Path,nargs='+',required=True)
    f.add_argument('--train-stages',type=Path,nargs='+',required=True);f.add_argument('--target-stages',type=Path,nargs='+',required=True)
    f.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--frozen',type=Path,required=True);e.add_argument('--target',type=Path,nargs='+',required=True);e.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();value=freeze(a.train_operators,a.train_stages,a.target_stages) if a.command=='freeze' else evaluate(a.frozen,a.target)
    payload=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as stream:stream.write(payload)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(payload).hexdigest(),
                      'summary':value.get('summary'),'predictions':len(value.get('predictions',[]))},indent=2))


if __name__=='__main__':main()
