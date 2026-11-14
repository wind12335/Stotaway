"""Prepare a route-bound two-probe decision bundle for safe TP2 startup.

This interface consumes independently collected GEMM components; it never
performs GPU probes implicitly inside a kernel call or a CUDA graph capture.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .reduced_probe_policy import ReducedProbePolicy
from ..tp2_lookup import key


def prepare(model_file,component_files,*,runtime,workload,capabilities,phase,target_model):
    raw=Path(model_file).read_bytes();bundle=json.loads(raw);sig=bundle['signature']
    if phase!='forward':
        raise ValueError('current model is calibrated for forward operators only')
    expected={'tp':sig[1],'dtype':sig[2],'gpu':sig[3],'torch':sig[4],'triton':sig[5],
              'cuda_build':sig[6],'cuda_device_max_connections':sig[7]}
    if bundle.get('schema')!='stowaway-fragmentation-freeze-v1' or sig[0]!='dx' or runtime!=expected:
        raise ValueError('model runtime mismatch')
    schemas={p.get('schema') for p in bundle.get('models',{}).values()}
    if schemas not in ({'stowaway-reduced-probe-v1'}, {'stowaway-local-residual-v1'}):
        raise ValueError('one consistent reduced or local residual model family required')
    if not isinstance(workload,dict) or not workload:raise ValueError('explicit workload required')
    if not isinstance(target_model,str) or not target_model:raise ValueError('explicit target model required')
    if any(not isinstance(workload.get(k), int) or isinstance(workload.get(k), bool) or workload[k] <= 0
           for k in ('seq_length','micro_batch_size')):
        raise ValueError('positive integer sequence length and micro batch size required')
    m=workload['seq_length']*workload['micro_batch_size']
    targets=[t for t in bundle['targets'] if t['model']==target_model and t['M']==m]
    if not targets:raise ValueError('target inventory/workload mismatch')
    expected_op={'qkv':'ag_up','fc1':'ag_gu','o_proj':'rs','fc2':'rs'}
    if any(t.get('op')!=expected_op.get(t.get('site')) for t in targets):
        raise ValueError('target site/operation mismatch')
    if schemas=={'stowaway-local-residual-v1'}:
        from .local_residual_policy import LocalResidualPolicy
        models={op:LocalResidualPolicy.from_dict(p) for op,p in bundle['models'].items()}
    else:
        models={op:ReducedProbePolicy.from_dict(p) for op,p in bundle['models'].items()}
    if any(op!=model.op for op,model in models.items()):raise ValueError('model operation mismatch')
    records={};sources=[]
    for path in component_files:
        raw_component=Path(path).read_bytes();d=json.loads(raw_component);md=d['metadata']
        if (md['tp'],md['gpu'],md['torch'],md['triton'],md['cuda_build'],md['cuda_device_max_connections'],md['batch'],md['warmup'])!=(
                sig[1],sig[3],sig[4],sig[5],sig[6],sig[7],sig[8],sig[-1]):
            raise ValueError('component runtime or timing mismatch')
        sources.append({'path':str(path),'sha256':hashlib.sha256(raw_component).hexdigest()})
        for row in d['results']:
            ident=(row['model'],row['site'],row['M'],row['N'],row['K'])
            if ident in records:raise ValueError('duplicate component row')
            records[ident]=row
    entries={};details=[]
    adapters={'qkv':'qkv','o_proj':'proj','fc1':'fc1','fc2':'fc2'}
    identities={(t['model'],t['site'],t['M'],t['N'],t['K']) for t in targets}
    if len(identities)!=len(targets):raise ValueError('duplicate target')
    if not identities.issubset(records):raise ValueError('incomplete target inventory')
    for target in targets:
        ident=(target['model'],target['site'],target['M'],target['N'],target['K'])
        row=records[ident];shape=(target['M'],target['N'],target['K']);site=adapters[target['site']]
        if row['op'] != ('rs' if target['op']=='rs' else 'ag'):
            raise ValueError('component operation mismatch')
        name=key(site,shape[0],shape[2],shape[1]);evidence=capabilities.get(name)
        if not isinstance(evidence,dict) or evidence.get('support') not in ('supported','unknown','unsupported') or not evidence.get('evidence'):
            raise ValueError('explicit per-shape capability evidence required')
        values=row['measurements'];query={'op':target['op'],'M':shape[0],'N':shape[1],'K':shape[2],
            'full_ms':values['full']['median_ms'],'split_ms':values['split']['median_ms']}
        pred=models[target['op']].predict(query)
        choice='dx' if evidence['support']=='supported' and pred['choice']=='overlap' else 'stock'
        if name in entries and entries[name]['choice']!=choice:raise ValueError('conflicting same runtime shape')
        entries[name]={'choice':choice};details.append({'key':name,'prediction':pred,'capability':evidence})
    return {'schema':'stowaway-tp2-forward-lookup-v1','classification':'diagnostic-forward-only',
            'runtime':runtime,'workload_requirements':workload,'entries':entries,'plan_source':'reduced-probe-model',
            'predictor_model_family':next(iter(schemas)),
            'target_model':target_model,'route':'dx-direct-exchange-split-gemm',
            'model_sha256':hashlib.sha256(raw).hexdigest(),'component_sources':sources,'decisions':details,
            'phase':'forward','warning':'forward operator prediction only; not a training plan or E2E result'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--components',type=Path,nargs='+',required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--workload',type=Path,required=True)
    parser.add_argument('--capabilities',type=Path,required=True)
    parser.add_argument('--target-model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    def load(path):return json.loads(path.read_text())
    result=prepare(args.model,args.components,runtime=load(args.runtime),workload=load(args.workload),
                   capabilities=load(args.capabilities),phase='forward',target_model=args.target_model)
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with args.output.open('xb') as stream:stream.write(raw)
    print(json.dumps({'path':str(args.output),'sha256':hashlib.sha256(raw).hexdigest(),
                      'entries':len(result['entries'])}))


if __name__=='__main__':main()
