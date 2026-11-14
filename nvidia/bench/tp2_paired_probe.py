"""Interleaved stock/DX TP2 probe on explicit shape, one collector process."""
import datetime
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import triton
from bench.tp2_cells import configured_sites
from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import initialize, finalize


def main():
    outer_start=time.monotonic();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180));cuda=initialize()
    m=int(os.environ['CELL_M']);blocks=int(os.environ.get('PROBE_BLOCKS','4'));batch=16
    seed=int(os.environ.get('PROBE_SEED','1'));warmup=int(os.environ.get('PROBE_WARMUP','8'))
    if blocks<2 or warmup<1:raise ValueError('at least two blocks and positive warmup')
    inventory=list(configured_sites(m))
    if len(inventory)!=1:raise ValueError('probe requires exactly one selected shape')
    model,site,op,m,k,n=inventory[0];gen=torch.Generator(device='cuda').manual_seed(799+rank)
    context=DirectExchange(cuda,rank,2,m,k,n,torch.bfloat16)
    w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16,generator=gen)*.05
    full=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=gen)*.05
    # The all-gather destination must not alias the input shard: rank 1's
    # first half is overwritten before the candidate consumes its input.
    shard=full[:m//2].clone();partial=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    output=torch.empty((m//2,n),device='cuda',dtype=torch.bfloat16)
    if op=='ag':
        def stock():
            dist.all_gather_into_tensor(full,shard);torch.matmul(full,w.T,out=partial);return partial
        def candidate():return context.all_gather_gemm(shard,w)
    else:
        def stock():
            torch.matmul(full,w.T,out=partial);dist.reduce_scatter_tensor(output,partial);return output
        def candidate():return context.gemm_reduce_scatter(full,w)
    start=time.monotonic()
    for _ in range(2):
        reference=stock().clone();actual=candidate();torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
    funcs={'stock':stock,'dx':candidate};rng=random.Random(seed);first=['stock','dx'];rng.shuffle(first)
    schedule=[first if i%2==0 else first[::-1] for i in range(blocks)]
    for name in first:
        for _ in range(warmup):funcs[name]()
    torch.cuda.synchronize();dist.barrier()
    pre_components=None
    if os.environ.get('PROBE_PRE_COMPONENTS')=='1':
        if op=='ag':
            def pre_comm():dist.all_gather_into_tensor(full,shard)
        else:
            def pre_comm():dist.reduce_scatter_tensor(output,partial)
        def pre_gemm():torch.matmul(full,w.T,out=partial)
        from bench.tp2_cells import measure
        pre_start=time.monotonic()
        pre_components={name:measure(fn,8,batch) for name,fn in (('comm',pre_comm),('gemm',pre_gemm))}
        pre_components['wall_ms']=(time.monotonic()-pre_start)*1000
    result=[]
    for index,order in enumerate(schedule):
        times={};windows={}
        for name in order:
            wall_begin=time.time()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            for _ in range(batch):funcs[name]()
            end.record();end.synchronize();t=torch.tensor([begin.elapsed_time(end)/batch],device='cuda')
            dist.all_reduce(t,op=dist.ReduceOp.MAX);times[name]=t.item()
            windows[name]=[wall_begin,time.time()]
        result.append({'block':index,'order':order,'stock_ms':times['stock'],'dx_ms':times['dx'],
                       'speedup':times['stock']/times['dx'],'rank0_wall_window_unix_s':windows})
    components=None
    if os.environ.get('PROBE_COMPONENTS')=='1':
        if op=='ag':
            def comm():dist.all_gather_into_tensor(full,shard)
            def gemm():torch.matmul(full,w.T,out=partial)
        else:
            def comm():dist.reduce_scatter_tensor(output,partial)
            def gemm():torch.matmul(full,w.T,out=partial)
        # These isolated timings are diagnostic; they are not assumed additive.
        from bench.tp2_cells import measure
        components={}
        for name,fn in (('comm',comm),('gemm',gemm)):
            component_start=time.time()
            components[name]=measure(fn,8,batch)
            components[name]['rank0_wall_window_unix_s']=[component_start,time.time()]
    duration_sample=None
    duration=float(os.environ.get('PROBE_COMPONENT_DURATION','0'))
    if duration:
        if not components or not .5<=duration<=10:
            raise ValueError('component duration requires components and must be 0.5–10 seconds')
        # Deliberately long same-process GEMM sampling to correlate external
        # coarse GPU clock telemetry; this is a diagnostic phase, not training data.
        start_s=time.time();end_s=time.monotonic()+duration;durations=[]
        while time.monotonic()<end_s:
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(batch):gemm()
            end.record();end.synchronize();durations.append(begin.elapsed_time(end)/batch)
        torch.cuda.synchronize();finish_s=time.time()
        duration_sample={'window_unix_s':[start_s,finish_s],'local_samples':len(durations),
                         'local_median_ms':statistics.median(durations),'rank':rank}
    torch.cuda.synchronize();wall=(time.monotonic()-start)*1000
    if rank==0:
        payload={'classification':'interleaved TP2 direct-exchange operator probe','model':model,'site':site,'op':op,
                 'M':m,'N':n,'K':k,'batch':batch,'blocks':blocks,'warmup':warmup,'checks':2,'seed':seed,
                 'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'triton':triton.__version__,
                 'cuda_build':torch.version.cuda,'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset'),
                 'point_wall_ms':wall,'collector_wall_ms':(time.monotonic()-outer_start)*1000,
                 'median_speedup':statistics.median(r['speedup'] for r in result),'block_results':result,'components':components,
                 'rank0_long_gemm':duration_sample,'pre_components':pre_components,
                 'warning':'same-shape exploratory probe, excludes full process startup'}
        with Path(os.environ['PROBE_OUT']).open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
        print(json.dumps(payload,indent=2),flush=True)
    context.close();finalize()


if __name__=='__main__':main()
