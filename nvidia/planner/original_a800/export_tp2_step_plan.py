"""Export one calibrated DX step decision to the TP2 diagnostic lookup format.

The selected set must come from choose_step; the caller must supply the same
call inventory and real step calibration. This adapter doesn't infer runtime
configuration or assert that the exported policy improves performance.
"""
import hashlib
import json
import math
from pathlib import Path

from ..tp2_lookup import TRAIN_SCHEMA, QWEN_SITES, key
from .unified_step import _inventory_digest


def export_step_plan(result, calls, *, runtime, workload, model_sources, step_calibration, workload_requirements=None):
    calls=tuple(calls)
    if not calls or result.get('method') != 'dx' or result.get('workload') != workload:
        raise ValueError('matching DX workload and call inventory required')
    if result.get('call_inventory_sha256')!=_inventory_digest(calls):
        raise ValueError('call inventory differs from the one used to select the plan')
    if not isinstance(step_calibration, dict) or not step_calibration.get('provenance'):
        raise ValueError('real step calibration and provenance are required')
    if step_calibration.get('workload') != workload:
        raise ValueError('step calibration workload mismatch')
    common=step_calibration.get('common_ms')
    if not isinstance(common,(int,float)) or isinstance(common,bool) or not math.isfinite(common):
        raise ValueError('finite step common overhead required')
    pairs=step_calibration.get('pairs_ms')
    if not isinstance(pairs,dict):
        raise ValueError('explicit interaction mapping required')
    if not model_sources or any(not isinstance(s,dict) or not isinstance(s.get('sha256'),str)
                                or len(s['sha256'])!=64 or any(ch not in '0123456789abcdef' for ch in s['sha256'])
                                for s in model_sources):
        raise ValueError('model provenance required')
    required_runtime={'tp','dtype','gpu','torch','triton','cuda_build','cuda_device_max_connections'}
    if set(runtime)!=required_runtime or runtime['tp'] != 2 or runtime['dtype'] != 'bfloat16':
        raise ValueError('supported TP2 BF16 runtime required')
    selected=set(result.get('best',{}).get('sites',[]))
    known={c.site for c in calls}
    residuals=step_calibration.get('site_residual_ms', {})
    if not isinstance(residuals,dict) or any(k not in known or not isinstance(v,(int,float)) or isinstance(v,bool)
                                            or not math.isfinite(v) for k,v in residuals.items()):
        raise ValueError('invalid per-site residuals')
    if not selected <= known:
        raise ValueError('selection contains unknown site')
    result_order=result.get('sites_order')
    if not isinstance(result_order,list) or set(result_order)!=known or len(result_order)!=len(known):
        raise ValueError('selection inventory differs')
    # Verify selection came from the calibrated candidate set and its actual
    # measured/provided step calibration, not a hand-constructed arbitrary mask.
    best=result.get('best')
    if best not in result.get('candidates',[]):
        raise ValueError('selected mask is not a predicted candidate')
    if selected and (not math.isfinite(best.get('conservative_delta_ms',float('nan')))
                     or best['conservative_delta_ms'] >= 0):
        raise ValueError('selected mask does not predict a conservative gain')
    if selected and not result.get('step_calibrated',False):
        raise ValueError('non-stock step must be calibrated')
    actual_digest=hashlib.sha256(json.dumps(step_calibration,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if result.get('calibration_sha256')!=actual_digest:
        raise ValueError('step calibration differs from the one used to select the plan')
    canonical_pairs={tuple(sorted(k.split('|'))):v for k,v in pairs.items() if isinstance(k,str) and k.count('|')==1}
    if len(canonical_pairs)!=len(pairs) or any(a==b or not {a,b}<=known or
            not isinstance(v,(float,int)) or isinstance(v,bool) or not math.isfinite(v)
            for (a,b),v in canonical_pairs.items()):
        raise ValueError('invalid interactions')
    if any((a,b) not in canonical_pairs for a in sorted(selected) for b in sorted(selected) if a < b):
        raise ValueError('selected pair lacks calibration')
    entries={}
    for c in calls:
        route=c.route
        versions=dict(route.versions)
        if (route.method!='dx' or route.actual_path!='direct_exchange' or route.compute!='vendor_gemm'
                or route.transport!='cuda_peer_copy' or route.memory!='nvshmem_symmetric'
                or route.tp!=2 or route.dtype!='bfloat16' or route.gpu!=runtime['gpu']
                or versions.get('torch')!=runtime['torch'] or versions.get('triton')!=runtime['triton']
                or versions.get('cuda')!=runtime['cuda_build'] or c.phase!='forward_backward'
                or f"connection={runtime['cuda_device_max_connections']}" not in route.config_family):
            raise ValueError('only matching DX forward+backward route is exportable')
        adapter=QWEN_SITES.get(c.site,c.site)
        if adapter not in ('qkv','proj','fc1','fc2'):
            raise ValueError('unsupported adapter site')
        if route.direction != ('ag' if adapter in ('qkv','fc1') else 'rs'):
            raise ValueError('site/collective mismatch')
        if (c.capability.route!=route or c.capability.shape!=c.shape or
                (c.site in selected and c.capability.support!='supported')):
            raise ValueError('capability mismatch')
        m,n,k=c.shape
        if m%2:
            raise ValueError('TP2 requires even M')
        name=key(adapter,m,k,n)
        choice='dx' if c.site in selected else 'stock'
        if name in entries and entries[name]['choice']!=choice:
            raise ValueError('conflicting choices for the same runtime adapter shape')
        entries[name]={'choice':choice}
    if workload_requirements is not None and (not isinstance(workload_requirements,dict) or not workload_requirements):
        raise ValueError('workload requirements must be a non-empty mapping')
    profile = {'schema':TRAIN_SCHEMA, 'classification':'diagnostic-forward-backward', 'runtime':runtime,
            'entries':entries, 'plan_source':'experimental-unified-step', 'workload':workload,
            'predicted_delta_ms':best['delta_ms'], 'conservative_delta_ms':best['conservative_delta_ms'],
            'step_calibration':step_calibration, 'model_sources':model_sources,
            'warning':'diagnostic only; conservative delta is an input allowance, not a statistical guarantee; requires independent validation'}
    if workload_requirements is not None:
        profile['workload_requirements'] = workload_requirements
    return profile


def write_json_new(path, profile):
    encoded=(json.dumps(profile,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()
