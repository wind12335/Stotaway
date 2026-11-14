"""Run an actual FlashOverlap RS arm selected by a frozen route policy."""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0,os.getcwd())
from test import reorder_indices,generate_row_remap_array

from planner.original_a800.orthogonal_predictor import ContextRequirement
from planner.original_a800.predictor_registry import PredictorRegistry,CachedRouteSelector
from planner.original_a800.route_policy import RoutePolicy
from planner.original_a800.hybrid_route_policy import HybridRoutePolicy
from planner.original_a800.route_local_residual import RouteLocalResidual
from planner.original_a800.guarded_hybrid_route import GuardedHybridRoute
from planner.original_a800.distributed_inference import Query,prepare_collectively


def worker(rank,world,nccl_id,m,n,k,config,policy_data,batch,repeats,warmup,config_sha,verified,output):
    torch.cuda.set_device(rank)
    os.environ.update(RANK=str(rank),WORLD_SIZE=str(world),MASTER_ADDR='127.0.0.1',MASTER_PORT='29659')
    dist.init_process_group('nccl')
    policy=(HybridRoutePolicy.from_dict(policy_data) if policy_data.get('schema')=='stowaway-hybrid-route-v1' else
            RouteLocalResidual.from_dict(policy_data) if policy_data.get('schema')=='stowaway-route-local-residual-v1' else
            GuardedHybridRoute.from_dict(policy_data) if policy_data.get('schema')=='stowaway-guarded-hybrid-route-v1' else
            RoutePolicy.from_dict(policy_data));route=policy.route
    if (route.method,route.compute,route.transport,route.actual_path,route.direction,route.tp,route.dtype,route.gpu)!=(
        'flashoverlap','cutlass','nccl','cutlass_nccl_wave','rs',2,'float16',torch.cuda.get_device_name(rank)):
        raise ValueError('route does not match verified FlashOverlap implementation')
    if (dict(route.versions).get('torch')!=str(torch.__version__) or route.timing_mode!=f'b2b{batch}'
            or route.warmup!=warmup or route.config_family!='predictive_search'):
        raise ValueError('policy runtime or measurement protocol mismatch')
    registry=PredictorRegistry();registry.register(policy);selector=CachedRouteSelector(registry)
    shape=(m,n,k);cap=ContextRequirement(route,shape,'supported' if verified else 'unknown',
        f'checked config sha256:{config_sha}' if verified else 'no explicit correctness evidence')
    started=time.perf_counter();pred=prepare_collectively(selector,[Query(route,shape,cap)]);decision_ms=(time.perf_counter()-started)*1000
    chosen=pred['decisions'][0]['choice']
    a=torch.randn((m,k),device='cuda',dtype=torch.float16)*.5
    b=torch.randn((n,k),device='cuda',dtype=torch.float16)*.5
    c=torch.empty((m,n),device='cuda',dtype=torch.float16);d=torch.empty((m//world,n),device='cuda',dtype=torch.float16)
    if chosen=='stock':
        impl=torch.classes.flashoverlap_class.BaselineImpl();impl.nccl_init(rank,world,nccl_id);impl.cublas_init()
        def run():impl.gemm_reducescatter(a,b,c,d)
    elif chosen=='flashoverlap':
        impl=torch.classes.flashoverlap_class.OverlapImpl();impl.nccl_init(rank,world,nccl_id);impl.cutlass_init();impl.overlap_init()
        bm,bn,algo,seg,hint=(config[x] for x in ('BM','BN','Algo','cSeg','hint'))
        if len(seg)<=1:raise ValueError('not an overlap config')
        counter=torch.zeros((n//bn,),dtype=torch.int32,device='cuda')
        order=reorder_indices((m//bm)*(n//bn),hint).reshape((m//bm,n//bn))
        row=generate_row_remap_array(m,n,bm,bn,seg,world)
        seg_cpu=torch.tensor(seg,dtype=torch.int32);seg_gpu=seg_cpu.cuda(rank)
        def run():impl.gemm_reducescatter_overlap(a,b,c,d,counter,order,row,1,seg_cpu,seg_gpu,algo,False)
    else:raise ValueError('invalid decision')
    for _ in range(warmup):run()
    torch.cuda.synchronize();dist.barrier();times=[]
    for _ in range(repeats):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch):run()
        end.record();end.synchronize()
        value=torch.tensor([start.elapsed_time(end)/batch],device='cuda');dist.all_reduce(value,op=dist.ReduceOp.MAX)
        times.append(value.item())
    gathered=[None]*world;dist.all_gather_object(gathered,{'rank':rank,'choice':chosen,'decision_ms':decision_ms})
    if len({r['choice'] for r in gathered})!=1:raise RuntimeError('different decisions across ranks')
    if rank==0:output.update({'rank_decisions':gathered,'prediction':pred['decisions'][0],
                              'samples_ms':times,'median_ms':statistics.median(times)})
    dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--m',type=int,required=True);ap.add_argument('--n',type=int,required=True)
    ap.add_argument('--k',type=int,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--policy',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--warmup',type=int,default=32);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--repeats',type=int,default=8);a=ap.parse_args()
    raw=a.policy.read_bytes();policy_file=json.loads(raw);policy=policy_file['policy'];config=a.config.read_bytes()
    config_sha=hashlib.sha256(config).hexdigest()
    verified=False
    for source in policy_file.get('sources',[]):
        p=Path(source['path']);record=json.loads(p.read_bytes())
        if (record['M'],record['N'],record['K'],record['config_sha256'])==(a.m,a.n,a.k,config_sha):
            # Only the supplied, previously checked shapes are in this example
            # evidence inventory; checksum binds the exact historical source.
            verified=hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
            if verified:break
    if torch.cuda.device_count()!=2:raise ValueError('requires TP2')
    manager=mp.Manager();out=manager.dict();nccl_id=torch.ops.flashoverlap_op.generate_nccl_id()
    mp.spawn(worker,args=(2,nccl_id,a.m,a.n,a.k,json.loads(config),policy,a.batch,a.repeats,a.warmup,config_sha,verified,out),nprocs=2)
    payload={'classification':'actual FlashOverlap/stock selected arm TP2 microbenchmark',
             'warning':'correctness verified separately; selected arm only; not E2E',
             'shape':[a.m,a.n,a.k],'policy_sha256':hashlib.sha256(raw).hexdigest(),
             'config_sha256':config_sha,'shape_evidence_found':verified,'batch':a.batch,'warmup':a.warmup,'repeats':a.repeats,**dict(out)}
    with a.output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
