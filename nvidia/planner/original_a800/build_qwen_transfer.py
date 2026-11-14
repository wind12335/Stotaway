"""Freeze a cross-M transfer experiment before reading target step timings.

This diagnostic transfers absolute site residuals and pair interaction from
the measured M2048 plan to M3072. Counts and capabilities are assumptions
validated separately by runtime counters; source and target are one method.
"""
import argparse
import hashlib
import json
import statistics
import math
from pathlib import Path

from .export_paircal_mask import RUNTIME, WORKLOAD
from .export_tp2_step_plan import export_step_plan, write_json_new
from .orthogonal_predictor import ContextRequirement, RouteRegressor, Sample
from .predictor_registry import PredictorRegistry, bind
from .unified_step import Callsite, choose_step
from ..tp2_lookup import QWEN_SITES


def load(path):
    raw=path.read_bytes()
    return json.loads(raw), {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--operator-data',nargs='+',type=Path,required=True)
    ap.add_argument('--source-paircal',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    source, source_ref=load(a.source_paircal)
    if source.get('context')!='qwen36-local4-random-mock-training-tp2-m2048-conn1-dx-sites' or len(source.get('blocks',[])) < 3:
        raise ValueError('source context differs')
    rows=[];refs=[];protocol=None
    for path in a.operator_data:
        data,ref=load(path);md=data['metadata'];refs.append(ref)
        if any(md.get(k)!=v for k,v in RUNTIME.items()) or md.get('classification')!='diagnostic unfused site forward+backward' or md.get('batch')!=16 or md.get('checks',0)<1:
            raise ValueError('operator context mismatch')
        timing=(md['batch'],md.get('warmup'))
        if protocol is not None and protocol!=timing:raise ValueError('operator protocol mismatch')
        protocol=timing
        if any(row['M']==3072 for row in data['results']):raise ValueError('target M must be held out')
        rows.extend(data['results'])
    registry=PredictorRegistry();routes={}
    for site in ('proj','fc2'):
        route=bind({'method':'dx','actual_path':'direct_exchange','direction':'rs','fallback':False},
                   gpu=RUNTIME['gpu'],topology='nv8',tp=2,dtype='bfloat16',
                   versions=(('cuda',RUNTIME['cuda_build']),('torch',RUNTIME['torch']),('triton',RUNTIME['triton'])),
                   timing_mode='b2b16',warmup=protocol[1],config_family=f'{site};connection=1;forward_backward')
        samples=[Sample(route,(r['M'],r['N'],r['K']),r['measurements']['stock']['median_ms'],
                        r['measurements']['dx']['median_ms']) for r in rows if QWEN_SITES[r['site']]==site]
        registry.register(RouteRegressor(route).fit(samples));routes[site]=route
    calls=[];source_operator={'proj':0.,'fc2':0.}
    for site,count,k in (('proj',2,3072),('fc2',6,3072),('fc2',8,8704)):
        route=routes[site];shape=(3072,5120,k)
        matches=[r for r in rows if QWEN_SITES[r['site']]==site and r['K']==k and r['N']==5120 and r['M'] in (2048,4096)]
        if {r['M'] for r in matches}!={2048,4096}:raise ValueError('missing stock bracket')
        stock=statistics.mean(r['measurements']['stock']['median_ms'] for r in matches)
        calls.append(Callsite(site,'forward_backward',route,shape,count,stock,
                              ContextRequirement(route,shape,'supported','M2048 and M4096 site diagnostics; target assumed'),0.))
        source_rows=[r for r in matches if r['M']==2048]
        source_stock=statistics.mean(r['measurements']['stock']['median_ms'] for r in source_rows)
        source_shape=(2048,5120,k)
        pred=registry.predict(route,source_shape,capability=ContextRequirement(route,source_shape,'supported','source diagnostics'))['predicted_log_speedup']
        source_operator[site]+=count*source_stock*(math.exp(-pred)-1)
    workload='qwen36-local4-random-mock-training-tp2-m3072-conn1-transfer'
    provenance='transferred-from-source-paircal-sha256:'+source_ref['sha256']
    pair=statistics.median(b['median_paired_interaction_ms'] for b in source['blocks'])
    residual={site:statistics.median(b['median_delta_ms'][site] for b in source['blocks'])-source_operator[site]
              for site in ('proj','fc2')}
    result=choose_step(registry,calls,workload,common_ms=0.,pairs_ms={('fc2','proj'):pair},
                       site_residual_ms=residual,calibration_provenance=provenance)
    cal={'workload':workload,'common_ms':0.,'pairs_ms':{'fc2|proj':pair},
         'site_residual_ms':residual,'provenance':provenance}
    profile=export_step_plan(result,calls,runtime=RUNTIME,workload=workload,model_sources=refs,
                             step_calibration=cal,workload_requirements={**WORKLOAD,'seq_length':3072})
    inventory={'qkv:3072:5120:7168','proj:3072:3072:5120','fc1:3072:5120:8240',
               'fc2:3072:3072:5120','fc1:3072:5120:17408','fc2:3072:8704:5120'}
    for k in inventory:profile['entries'].setdefault(k,{'choice':'stock'})
    if set(profile['entries'])!=inventory:raise ValueError('inventory mismatch')
    profile['model_audit']={'operator_sources':refs,'source_paircal':source_ref,'source_operator_delta_ms':source_operator,'selection':result,
                            'target_operator_and_step_measurements_used':False,
                            'counts_assumption':{'proj':2,'gdn_fc2':6,'mlp_fc2':8},
                            'warning':'transferred absolute residual; target support/counts need runtime validation'}
    digest=write_json_new(a.output,profile)
    print(json.dumps({'path':str(a.output),'sha256':digest,'best':result['best'],'candidates':result['candidates']},indent=2))


if __name__=='__main__':main()
