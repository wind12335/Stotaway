"""Cross-validate the component model by holding out entire physical families."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .fragmentation_mixture import FragmentationMixture
from .validate_tp2_new import load


def join(operator_paths,stage_paths,*,allow_unknown_stage=False):
    operators,sig,operator_sources=load(operator_paths)
    if sig[0]!='dx' or sig[1]!=2 or sig[-1]!=3:raise ValueError('required DX TP2 warmup=3')
    stage={};stage_sources=[]
    for path in stage_paths:
        raw=path.read_bytes();data=json.loads(raw);md=data['metadata']
        if (md['tp'],md['batch'],md['warmup'],md['gpu'],md['torch'],md['cuda_build'])!=(sig[1],sig[8],sig[-1],sig[3],sig[4],sig[6]):
            raise ValueError('stage protocol mismatch')
        for field,expected in (('triton',sig[5]),('cuda_device_max_connections',sig[7])):
            if field not in md:
                if not allow_unknown_stage:raise ValueError('stage metadata missing '+field)
            elif md[field]!=expected:raise ValueError('stage protocol mismatch: '+field)
        stage_sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
        for row in data['results']:
            key=(row['model'],row['site'],row['M'],row['N'],row['K'])
            if key in stage:raise ValueError('duplicate stage data')
            stage[key]=row
    output=[]
    for row in operators:
        key=(row['source_model'],row['site'],row['M'],row['N'],row['K']);s=stage[key]['measurements']
        output.append({'family':row['model'],'op':row['op'],'identity':key,'M':row['M'],'N':row['N'],'K':row['K'],
                       'full_ms':s['full']['median_ms'],'split_ms':s['split']['median_ms'],'comm_ms':row['C'],
                       'stock_ms':row['stock_b2b'],'overlap_ms':row['dx_b2b']})
    if len(output)!=len(stage):raise ValueError('operator/stage inventory mismatch')
    return output,sig,operator_sources,stage_sources


def evaluate(rows,model_class=FragmentationMixture):
    out=[]
    for row in rows:
        # All M and all site aliases of same physical AG/RS K/N held out.
        train=[r for r in rows if r['op']==row['op'] and r['family']!=row['family']]
        model=model_class().fit(train);query={k:v for k,v in row.items() if k!='overlap_ms'}
        pred=model.predict(query);speed=row['stock_ms']/row['overlap_ms']
        out.append({'identity':row['identity'],'family':row['family'],'choice':pred['choice'],
                    'oracle':'overlap' if speed>1 else 'stock','actual_speedup':speed,
                    'speedup':speed if pred['choice']=='overlap' else 1.,'prediction':pred})
    return out


def summarize(rows,mode):
    chosen=[r['choice'] if mode=='model' else mode for r in rows]
    speeds=[r['actual_speedup'] if c=='overlap' else 1. for c,r in zip(chosen,rows)]
    return {'count':len(rows),'correct':sum(c==r['oracle'] for c,r in zip(chosen,rows)),
            'slower':sum(s<1 for s in speeds),'speedup_gmean':math.exp(statistics.mean(math.log(s) for s in speeds))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--piecewise',action='store_true');a=ap.parse_args()
    if a.piecewise:
        from .piecewise_gain_policy import PiecewiseGainPolicy
        model_class=PiecewiseGainPolicy
    else:model_class=FragmentationMixture
    data,sig,ops,stages=join(a.operators,a.stages);rows=evaluate(data,model_class)
    result={'classification':'exploratory leave-physical-family-out component model','model_type':model_class.__name__,
            'warning':'existing development data, target stock and G/C and split GEMM are measured; cost excluded; not E2E',
            'signature':sig,'operator_sources':ops,'stage_sources':stages,
            'summary':{p:summarize(rows,p) for p in ('model','stock','overlap')},'rows':rows}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
