"""Validated operator-only inference bundle for an explicit TP2 workload.

The bundle enumerates capability-verified shapes; it does not infer shape
support, build a step model, or select a runtime configuration automatically.
"""
import hashlib
import json
from pathlib import Path

from ..tp2_lookup import QWEN_SITES, key
from .orthogonal_predictor import ContextRequirement, RouteRegressor
from .hybrid_route_policy import HybridRoutePolicy
from .predictor_registry import CachedRouteSelector, PredictorRegistry
from .distributed_inference import Query


class OnlineBundle:
    def __init__(self,path,runtime,workload):
        raw=Path(path).read_bytes();self.digest=hashlib.sha256(raw).hexdigest()
        profile=json.loads(raw)
        if profile.get('schema')!='stowaway-tp2-online-bundle-v1' or profile.get('runtime')!=runtime:
            raise ValueError('online bundle schema/runtime mismatch')
        if profile.get('workload_requirements')!=workload:
            raise ValueError('online bundle workload mismatch')
        expected_runtime={'tp','dtype','gpu','torch','triton','cuda_build','cuda_device_max_connections'}
        if set(runtime)!=expected_runtime or runtime['tp']!=2 or runtime['dtype']!='bfloat16':
            raise ValueError('unsupported runtime')
        self.registry=PredictorRegistry();models={}
        for site,payload in profile['models'].items():
            if site not in ('qkv','proj','fc1','fc2'):raise ValueError('invalid site')
            model=(HybridRoutePolicy.from_dict(payload) if payload.get('schema')=='stowaway-hybrid-route-v1'
                   else RouteRegressor.from_dict(payload));r=model.route;vers=dict(r.versions)
            if (r.method!='dx' or r.actual_path!='direct_exchange' or r.compute!='vendor_gemm'
                or r.transport!='cuda_peer_copy' or r.memory!='nvshmem_symmetric' or r.tp!=2
                or r.dtype!='bfloat16' or r.gpu!=runtime['gpu']
                or r.direction!=('ag' if site in ('qkv','fc1') else 'rs')
                or vers.get('torch')!=runtime['torch'] or vers.get('triton')!=runtime['triton']
                or vers.get('cuda')!=runtime['cuda_build']
                or f"connection={runtime['cuda_device_max_connections']}" not in r.config_family
                or 'forward_backward' not in r.config_family):
                raise ValueError('model route does not match runtime/site/phase')
            self.registry.register(model);models[site]=r
        self.routes=models
        self.queries=[];self.keys=[]
        for entry in profile['queries']:
            site=entry['site'];shape=tuple(entry['shape']);route=models[site]
            capability=ContextRequirement(route,shape,entry['support'],entry['evidence'])
            if capability.support!='supported':raise ValueError('only verified shapes belong in bundle')
            if shape[0]%2 or shape[0]!=workload['seq_length']*workload['micro_batch_size']:
                raise ValueError('query M inconsistent with workload')
            name=key(site,shape[0],shape[2],shape[1])
            if name in self.keys:raise ValueError('duplicate query')
            self.keys.append(name);self.queries.append(Query(route,shape,capability))
        if not self.queries:raise ValueError('empty inference bundle')
        self.selector=CachedRouteSelector(self.registry)
        override=profile.get('overrides',{})
        if not isinstance(override,dict) or not set(override)<=set(self.keys):raise ValueError('invalid overrides')
        for name,item in override.items():
            if not isinstance(item,dict) or item.get('choice') not in ('stock','dx') or not isinstance(item.get('source_sha256'),str) or len(item['source_sha256'])!=64:
                raise ValueError('invalid override evidence')
        self.overrides=override

    def choices(self,decisions):
        if len(decisions)!=len(self.keys):raise ValueError('incomplete decisions')
        out={}
        for name,result in zip(self.keys,decisions):
            if result.get('choice') not in ('stock','dx'):raise ValueError('invalid choice')
            out[name]=self.overrides.get(name,{}).get('choice',result['choice'])
        return out

    def choose(self,choices,site,m,k,n,*,dtype='bfloat16'):
        """Unlisted shapes and unsupported dtypes use the original operator."""
        if dtype!='bfloat16':return 'stock'
        return choices.get(key(site,m,k,n),'stock')


def build_bundle(train_reports, capability_report, runtime, workload, *, hybrid=False):
    """Build a demonstrator from disjoint training M and checked target shapes."""
    from .predictor_registry import bind
    from .orthogonal_predictor import Sample
    from .route_policy import GroupedSample
    capability=json.loads(Path(capability_report).read_bytes());md=capability['metadata']
    topology=md.get('topology','unverified')
    if not isinstance(topology,str) or not topology.strip():
        raise ValueError('topology must be a nonempty explicitly reported string')
    def check(m):
        if any(m.get(k)!=v for k,v in runtime.items()) or m.get('batch')!=16 or m.get('checks',0)<1 or m.get('classification')!='diagnostic unfused site forward+backward':
            raise ValueError('incompatible checked forward+backward report')
        if m.get('topology','unverified')!=topology:
            raise ValueError('training and capability topology metadata mismatch')
    check(md);target_m=md['M'];protocol=(md['batch'],md.get('warmup'))
    if target_m!=workload['seq_length']*workload['micro_batch_size']:
        raise ValueError('capability M/workload mismatch')
    rows=[];sources=[]
    for path in train_reports:
        raw=Path(path).read_bytes();report=json.loads(raw);check(report['metadata'])
        if (report['metadata']['batch'],report['metadata'].get('warmup'))!=protocol:
            raise ValueError('timing protocol mismatch')
        if any(r['M']==target_m for r in report['results']):raise ValueError('target M in training data')
        rows.extend(report['results']);sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
    sites=sorted({QWEN_SITES[r['site']] for r in capability['results']})
    models={};queries=[]
    cap_sha=hashlib.sha256(Path(capability_report).read_bytes()).hexdigest()
    for site in sites:
        direction='ag' if site in ('qkv','fc1') else 'rs'
        route=bind({'method':'dx','actual_path':'direct_exchange','direction':direction,'fallback':False},
                   gpu=runtime['gpu'],topology=topology,tp=2,dtype='bfloat16',
                   versions=(('cuda',runtime['cuda_build']),('torch',runtime['torch']),('triton',runtime['triton'])),
                   timing_mode='b2b16',warmup=protocol[1],config_family=f'{site};connection={runtime["cuda_device_max_connections"]};forward_backward')
        grouped=[GroupedSample(f"{r['K']}:{r['N']}",Sample(route,(r['M'],r['N'],r['K']),r['measurements']['stock']['median_ms'],
                        r['measurements']['dx']['median_ms'])) for r in rows if QWEN_SITES[r['site']]==site]
        models[site]=(HybridRoutePolicy(route).fit(grouped).to_dict() if hybrid else
                      RouteRegressor(route).fit([r.sample for r in grouped]).to_dict())
        for r in capability['results']:
            if QWEN_SITES[r['site']]==site:
                queries.append({'site':site,'shape':[r['M'],r['N'],r['K']],
                                'support':'supported','evidence':f'checked-training-site-sha256:{cap_sha}'})
    return {'schema':'stowaway-tp2-online-bundle-v1','runtime':runtime,'workload_requirements':workload,
            'models':models,'queries':queries,'training_sources':sources,
            'capability_source':{'path':str(capability_report),'sha256':cap_sha},
            'warning':'operator-only predictions; target M timing labels excluded from model fit; not step optimized',
            'model_type':'hybrid_exact_and_grouped' if hybrid else 'ridge'}
