"""Build an auditable unified-model TP2 Qwen diagnostic from existing data.

Leave-one-M-out operator regression, then fit site residuals to measured
whole-step singleton deltas.  The same M2048 step blocks calibrate the step
model, so they are not a held-out performance evaluation.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .export_paircal_mask import CONTEXT, INVENTORY, RUNTIME, WORKLOAD
from .export_tp2_step_plan import export_step_plan, write_json_new
from .orthogonal_predictor import ContextRequirement, RouteRegressor, Sample
from .predictor_registry import PredictorRegistry, bind
from .unified_step import Callsite, choose_step
from ..tp2_lookup import QWEN_SITES, key


def read(path):
    raw=path.read_bytes()
    return json.loads(raw), {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--operator-data',nargs='+',type=Path,required=True)
    ap.add_argument('--paircal',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report, step_source=read(args.paircal)
    if report.get('context') != CONTEXT or len(report.get('blocks',[])) < 3:
        raise ValueError('matching three-block paircal required')
    rows=[]; sources=[]; protocol=None
    for path in args.operator_data:
        data,source=read(path); md=data['metadata'];sources.append(source)
        if any(md.get(k)!=v for k,v in RUNTIME.items()) or md.get('classification')!='diagnostic unfused site forward+backward' or md.get('batch')!=16 or md.get('checks',0)<1:
            raise ValueError(f'operator runtime or phase mismatch: {path}')
        p=(md['batch'],md.get('warmup'))
        if protocol is not None and protocol!=p:raise ValueError('operator protocols differ')
        protocol=p
        for r in data['results']:
            if r['M'] == 2048:raise ValueError('hold out target M from operator model')
            rows.append(r)
    # Aggregating each measured adapter site, not each invocation, matches the
    # scope of STOWAWAY_DX_SITES in the step calibration.
    registry=PredictorRegistry();routes={}
    for site in ('proj','fc2'):
        route=bind({'method':'dx','actual_path':'direct_exchange','direction':'rs','fallback':False},
                   gpu=RUNTIME['gpu'],topology='nv8',tp=2,dtype='bfloat16',
                   versions=(('cuda',RUNTIME['cuda_build']),('torch',RUNTIME['torch']),('triton',RUNTIME['triton'])),
                   timing_mode='b2b16',warmup=protocol[1],config_family=f'{site};connection=1;forward_backward')
        samples=[Sample(route,(r['M'],r['N'],r['K']),r['measurements']['stock']['median_ms'],
                        r['measurements']['dx']['median_ms']) for r in rows if QWEN_SITES[r['site']]==site]
        if len(samples)<2:raise ValueError(f'insufficient operator calibration for {site}')
        registry.register(RouteRegressor(route).fit(samples));routes[site]=route
    # Five-step runtime shape counts subsequently confirmed the 2/6/8 counts
    # for this exact diagnostic; callers must not extrapolate them elsewhere.
    calls=[]
    for site,count,k in (('proj',2,3072),('fc2',6,3072),('fc2',8,8704)):
        route=routes[site];shape=(2048,5120,k)
        matches=[r for r in rows if QWEN_SITES[r['site']]==site and r['K']==k and r['N']==5120]
        at_m=[r for r in matches if r['M'] in (1024,3072)]
        if {r['M'] for r in at_m}!={1024,3072}:raise ValueError('bracketing stock samples required')
        stock=statistics.mean(r['measurements']['stock']['median_ms'] for r in at_m)
        calls.append(Callsite(site,'forward_backward',route,shape,count,stock,
                              ContextRequirement(route,shape,'supported','prior DX site diagnostics'),0.))
    # Calibrate each logical site's actual difference minus operator estimate;
    # interaction is derived from the medians of paired within-step differences.
    med=lambda name:statistics.median(b['median_delta_ms'][name] for b in report['blocks'])
    operator_delta={'proj':0.,'fc2':0.}
    for c in calls:
        pred=registry.predict(c.route,c.shape,capability=c.capability)['predicted_log_speedup']
        operator_delta[c.site]+=c.count*c.stock_ms*(math.exp(-pred)-1)
    site_residuals={s:med(s)-operator_delta[s] for s in ('proj','fc2')}
    interaction=statistics.median(b['median_paired_interaction_ms'] for b in report['blocks'])
    provenance=f"paired-step-sha256:{step_source['sha256']}"
    result=choose_step(registry,calls,CONTEXT,common_ms=0.,pairs_ms={('fc2','proj'):interaction},
                       site_residual_ms=site_residuals,calibration_provenance=provenance)
    cal={'workload':CONTEXT,'common_ms':0.,'pairs_ms':{'fc2|proj':interaction},
         'site_residual_ms':site_residuals,'provenance':provenance}
    profile=export_step_plan(result,calls,runtime=RUNTIME,workload=CONTEXT,model_sources=sources,
                             step_calibration=cal,workload_requirements=WORKLOAD)
    # Explicitly cover the other measured callsites with stock. The exported
    # proj and both fc2 choices come from their individual Callsite evidence.
    for name in INVENTORY:
        profile['entries'].setdefault(name,{'choice':'stock'})
    if set(profile['entries']) != INVENTORY:raise ValueError('unexpected target inventory')
    profile['model_audit']={'target_M_excluded_from_operator_training':True,'operator_sources':sources,
                            'step_source':step_source,'operator_delta_ms':operator_delta,
                            'counts_per_step':{'proj':2,'gdn_fc2':6,'mlp_fc2':8},'selection':result,
                            'counts_evidence':'qwen36_tp2_unified_v2_shape_counts_20261112.log',
                            'warning':'step residual calibrated on the same blocks; not held-out step evaluation'}
    digest=write_json_new(args.output,profile)
    print(json.dumps({'path':str(args.output),'sha256':digest,'best':result['best'],
                      'operator_delta_ms':operator_delta,'site_residual_ms':site_residuals,
                      'interaction_ms':interaction},indent=2))


if __name__=='__main__':main()
