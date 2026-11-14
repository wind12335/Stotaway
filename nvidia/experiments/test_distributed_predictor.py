"""Small two-rank process-group validation, no model E2E performance claim."""
import os
import time
import json
import torch
import torch.distributed as dist

from planner.original_a800.distributed_inference import Query, prepare_collectively
from planner.original_a800.orthogonal_predictor import ContextRequirement, RouteRegressor, Sample
from planner.original_a800.predictor_registry import CachedRouteSelector, PredictorRegistry, bind
from planner.original_a800.route_policy import GroupedSample, RoutePolicy


def main():
    rank=int(os.environ['LOCAL_RANK']);backend=os.environ.get('PREDICTOR_TEST_BACKEND','nccl')
    if backend=='nccl':torch.cuda.set_device(rank)
    dist.init_process_group(backend)
    route=bind({'method':'dx','actual_path':'direct_exchange','direction':'ag','fallback':False},
               gpu='test',topology='test',tp=2,dtype='bfloat16',versions=(('torch',str(torch.__version__)),),
               timing_mode='b2b16',warmup=3,config_family='test')
    shape=(1024,4096,4096);cap=ContextRequirement(route,shape,'supported','test')
    model=RouteRegressor(route).fit([Sample(route,shape,1,.8),Sample(route,(2048,4096,4096),2,1.6)])
    registry=PredictorRegistry();registry.register(model);selector=CachedRouteSelector(registry)
    q=Query(route,shape,cap)
    start=time.perf_counter();result=prepare_collectively(selector,[q]);elapsed=(time.perf_counter()-start)*1000
    assert result['decisions'][0]['choice']=='dx'
    from planner.original_a800.dynamic_query import make_query,update_cache
    dynamic=make_query(route,(2048,4096,4096),support='supported',evidence='test-safe-point')
    dynamic_result=prepare_collectively(selector,[dynamic])
    cache={};update_cache(cache,'qkv',dynamic,dynamic_result['decisions'][0])
    assert cache['qkv:2048:4096:4096']=='dx'
    flash=bind({'method':'flashoverlap','actual_path':'cutlass_nccl_wave','direction':'rs','fallback':False},
               gpu='test',topology='test',tp=2,dtype='bfloat16',versions=(('torch',str(torch.__version__)),),
               timing_mode='b2b16',warmup=32,config_family='test')
    policy=RoutePolicy(flash).fit([GroupedSample(str(i),Sample(flash,(1024*(i+1),4096,4096),1,.8)) for i in range(4)])
    registry.register(policy)
    fc=ContextRequirement(flash,shape,'supported','test')
    flash_result=prepare_collectively(selector,[Query(flash,shape,fc)])
    assert flash_result['decisions'][0]['choice']=='flashoverlap'
    policy_path=os.environ.get('STOWAWAY_ROUTE_POLICY')
    if policy_path:
        with open(policy_path) as f:payload=json.load(f)
        actual_policy=RoutePolicy.from_dict(payload['policy'])
        registry.register(actual_policy)
        actual_shape=(4096,8192,4096)
        actual_cap=ContextRequirement(actual_policy.route,actual_shape,'supported','previous FlashOverlap TP2 run')
        actual=prepare_collectively(selector,[Query(actual_policy.route,actual_shape,actual_cap)])
        print(f'PREDICTOR_LOADED_POLICY rank={rank} result={actual["decisions"][0]}',flush=True)
    try:
        altered=Query(route,shape,ContextRequirement(route,shape,'unknown' if rank else 'supported','test'))
        prepare_collectively(selector,[altered])
        raise AssertionError('mismatch was not rejected')
    except RuntimeError as exc:
        assert 'inconsistent predictor query batch' in str(exc)
    try:
        prepare_collectively(None,[q])
        raise AssertionError('missing leader was not rejected')
    except RuntimeError as exc:
        assert 'leader predictor error' in str(exc)
    try:
        bad=Query(route,shape,cap,margin_log=float('nan')) if rank==1 else q
        prepare_collectively(selector,[bad])
        raise AssertionError('invalid local query was not rejected')
    except RuntimeError as exc:
        assert 'inconsistent predictor query batch' in str(exc)
    dist.barrier()
    warm=[]
    for _ in range(10):
        start=time.perf_counter();prepare_collectively(selector,[q]);warm.append((time.perf_counter()-start)*1000)
    import statistics
    print(f'PREDICTOR_COLLECTIVE_OK rank={rank} backend={backend} cold_ms={elapsed:.3f} '
          f'warm_median_ms={statistics.median(warm):.3f} samples_ms={warm}',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
