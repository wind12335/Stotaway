"""TP2 Triton-distributed AG+GEMM with route-policy decision before execution."""
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton_dist
from triton_dist.kernels.nvidia import ag_gemm,create_ag_gemm_context
from triton_dist.utils import initialize_distributed,finalize_distributed

from planner.original_a800.orthogonal_predictor import ContextRequirement
from planner.original_a800.predictor_registry import PredictorRegistry,CachedRouteSelector
from planner.original_a800.route_policy import RoutePolicy
from planner.original_a800.hybrid_route_policy import HybridRoutePolicy
from planner.original_a800.route_local_residual import RouteLocalResidual
from planner.original_a800.guarded_hybrid_route import GuardedHybridRoute
from planner.original_a800.distributed_inference import Query,prepare_collectively
from bench.tp2_cells import measure


def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    group=initialize_distributed();m=int(os.environ.get('TD_M','1024'));n=int(os.environ.get('TD_N','5120'));k=int(os.environ.get('TD_K','8192'))
    path=Path(os.environ['TD_POLICY']);raw=path.read_bytes();payload=json.loads(raw);data=payload['policy']
    policy=(HybridRoutePolicy.from_dict(data) if data.get('schema')=='stowaway-hybrid-route-v1' else
            RouteLocalResidual.from_dict(data) if data.get('schema')=='stowaway-route-local-residual-v1' else
            GuardedHybridRoute.from_dict(data) if data.get('schema')=='stowaway-guarded-hybrid-route-v1' else
            RoutePolicy.from_dict(data));route=policy.route
    vers=dict(route.versions)
    if (route.method,route.compute,route.transport,route.memory,route.direction,route.tp,route.dtype,route.gpu)!=(
        'triton_dist','triton','cuda_peer_copy','nvshmem_symmetric','ag',2,'bfloat16',torch.cuda.get_device_name()):
        raise ValueError('route mismatch')
    if vers.get('torch')!=str(torch.__version__) or vers.get('triton')!=str(triton.__version__) or vers.get('cuda')!=str(torch.version.cuda):
        raise ValueError('version mismatch')
    if f'path={triton_dist.__file__}' not in route.config_family or f"connection={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset')}" not in route.config_family:
        raise ValueError('implementation path mismatch')
    if route.timing_mode!='b2b16' or route.config_family.find('autotune=False')<0:
        raise ValueError('timing mode or autotune mismatch')
    shape=(m,n,k);evidence=None
    for source in payload.get('sources',[]):
        p=Path(source['path']);source_raw=p.read_bytes()
        if hashlib.sha256(source_raw).hexdigest()!=source['sha256']:raise ValueError('source checksum mismatch')
        record=json.loads(source_raw)
        if record['metadata'].get('checks',0)>0 and any((r['M'],r['N'],r['K'])==shape for r in record['results']):
            evidence=source['sha256'];break
    cap=ContextRequirement(route,shape,'supported' if evidence else 'unknown',
                           f'checked source sha256:{evidence}' if evidence else 'shape not checked')
    registry=PredictorRegistry();registry.register(policy);selector=CachedRouteSelector(registry)
    start=time.perf_counter();decision=prepare_collectively(selector,[Query(route,shape,cap)],group=group)
    decision_ms=(time.perf_counter()-start)*1000;choice=decision['decisions'][0]['choice']
    ctx=create_ag_gemm_context(m,n*2,k,torch.bfloat16,rank,2,num_local_ranks=2)
    shard=torch.randn((m//2,k),device='cuda',dtype=torch.bfloat16)*.05
    weight=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)*.05
    full=torch.empty((m,k),device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    dist.all_gather_into_tensor(full,shard,group=group);reference=full@weight.T
    # Verify actual overlap even when policy chose stock, then measure selected arm.
    actual=ag_gemm(shard,weight.T,ctx=ctx,autotune=False)
    torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
    def stock():
        dist.all_gather_into_tensor(full,shard,group=group);torch.matmul(full,weight.T,out=out)
    def overlap():return ag_gemm(shard,weight.T,ctx=ctx,autotune=False)
    if choice not in ('stock','triton_dist'):raise ValueError('invalid selection')
    result=measure(stock if choice=='stock' else overlap,8,16)
    info=[None]*2;dist.all_gather_object(info,{'rank':rank,'choice':choice,'decision_ms':decision_ms},group=group)
    if len({x['choice'] for x in info})!=1:raise ValueError('rank mismatch')
    if rank==0:
        report={'classification':'actual Triton-dist selected TP2 AG arm','warning':'operator only, target previously measured',
                'shape':shape,'policy_sha256':hashlib.sha256(raw).hexdigest(),'prediction':decision['decisions'][0],
                'rank_decisions':info,'measurement':result,'correctness':'overlap allclose atol=rtol=.02'}
        p=Path(os.environ['TD_OUT'])
        with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(json.dumps(report,indent=2),flush=True)
    ctx.finalize();finalize_distributed()


if __name__=='__main__':main()
