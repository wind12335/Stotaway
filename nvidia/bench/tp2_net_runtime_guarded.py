"""TP2 experiment: charge in-process probing to actual executed work.

Two independent process arms share shape, initialization, correctness checks and
warmup. 'adaptive' measures paired stock/DX before selecting one path; 'stock'
performs no probe. Wall time includes selection/probe in the adaptive arm.
"""
import datetime
import json
import math
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
from planner.original_a800.latency_guard import LatencyGuard
from planner.original_a800.net_probe_policy import select_paired


def maximum(value):
    tensor=torch.tensor([value],dtype=torch.float64,device='cuda')
    dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
    return tensor.item()


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180));cuda=initialize()
    m=int(os.environ['CELL_M']);inventory=list(configured_sites(m))
    if len(inventory)!=1:raise ValueError('exactly one explicit shape required')
    model,site,op,m,k,n=inventory[0];arm=os.environ['NET_ARM']
    if arm not in ('stock','adaptive','guarded'):raise ValueError('invalid arm')
    calls=int(os.environ.get('NET_CALLS','2048'));warmup=int(os.environ.get('NET_WARMUP','32'))
    blocks=int(os.environ.get('NET_PROBE_BLOCKS','4'));batch=16;seed=int(os.environ['NET_SEED'])
    if calls<1 or warmup<1 or blocks<2:raise ValueError('invalid counts')
    gen=torch.Generator(device='cuda').manual_seed(799+rank)
    ctx=DirectExchange(cuda,rank,2,m,k,n,torch.bfloat16)
    w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16,generator=gen)*.05
    full=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=gen)*.05
    shard=full[:m//2].clone();partial=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    output=torch.empty((m//2,n),device='cuda',dtype=torch.bfloat16)
    if op=='ag':
        def stock():dist.all_gather_into_tensor(full,shard);torch.matmul(full,w.T,out=partial);return partial
        def dx():return ctx.all_gather_gemm(shard,w)
    else:
        def stock():torch.matmul(full,w.T,out=partial);dist.reduce_scatter_tensor(output,partial);return output
        def dx():return ctx.gemm_reduce_scatter(full,w)
    for _ in range(2):
        reference=stock().clone();actual=dx();torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
    # Common warmup is equal for both arms, including potential context effects.
    for _ in range(warmup):stock();dx()
    torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
    samples=[];choice='stock';probe_ms=0.;reason='baseline arm';predicted_saving=0.
    if arm in ('adaptive','guarded'):
        funcs={'stock':stock,'dx':dx};order=['stock','dx'];random.Random(seed).shuffle(order)
        for i in range(blocks):
            times={}
            for name in (order if i%2==0 else order[::-1]):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                for _ in range(batch):funcs[name]()
                end.record();end.synchronize();times[name]=maximum(begin.elapsed_time(end)/batch)
            samples.append(times)
        torch.cuda.synchronize();probe_ms=maximum((time.perf_counter()-start)*1000)
        # The elapsed max-rank wall includes probe and synchronization. The
        # saved compute must recover the entire measured incremental probe cost.
        policy=select_paired(samples,probe_ms=probe_ms,remaining_calls=calls)
        predicted_saving=policy['conservative_saving_ms'];choice=policy['choice']
        reason='median paired saving minus MAD, after probe cost'
        choice_obj=[choice if rank==0 else None];dist.broadcast_object_list(choice_obj,src=0);choice=choice_obj[0]
    initial_choice=choice
    selected=dx if choice=='dx' else stock
    run_start=time.perf_counter();gpu_begin,gpu_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    gpu_begin.record()
    chunks=[];transitions=[];limit=128;stock_reference=statistics.median(x['stock'] for x in samples) if samples else None
    guard=LatencyGuard(stock_reference,initial=choice) if arm=='guarded' else None
    for offset in range(0,calls,limit if arm=='guarded' else calls):
        count=min(limit,calls-offset) if arm=='guarded' else calls
        chunk_start,chunk_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        if arm=='guarded':chunk_start.record()
        for _ in range(count):selected()
        if arm=='guarded':
            chunk_end.record();chunk_end.synchronize();elapsed=maximum(chunk_start.elapsed_time(chunk_end)/count)
            chunks.append({'offset':offset,'count':count,'choice':choice,'ms_per_call':elapsed})
            next_choice=guard.observe(choice,elapsed)
            if next_choice!=choice:
                choice=next_choice;selected=stock;transitions.append({'after_calls':offset+count,'choice':choice})
    gpu_end.record();gpu_end.synchronize();run_wall_ms=(time.perf_counter()-run_start)*1000
    total_wall_ms=(time.perf_counter()-start)*1000
    gpu_ms=maximum(gpu_begin.elapsed_time(gpu_end));run_wall_ms=maximum(run_wall_ms);total_wall_ms=maximum(total_wall_ms)
    checks=[None]*2;dist.all_gather_object(checks,choice)
    if checks!=[choice,choice]:raise ValueError('rank decision mismatch')
    if rank==0:
        result={'classification':'executed same-process probe and selected TP2 operator workload',
          'arm':arm,'model':model,'site':site,'op':op,'M':m,'N':n,'K':k,'calls':calls,'batch':batch,
          'warmup':warmup,'probe_blocks':blocks,'seed':seed,'gpu':torch.cuda.get_device_name(),
          'torch':str(torch.__version__),'triton':triton.__version__,'cuda_build':torch.version.cuda,
          'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset'),
          'choice':choice,'initial_choice':initial_choice,'decision_reason':reason,'predicted_conservative_saving_ms_per_call':predicted_saving,
          'probe_max_rank_wall_ms':probe_ms,'chosen_gpu_max_rank_ms':gpu_ms,
          'chosen_run_max_rank_wall_ms':run_wall_ms,'total_max_rank_wall_ms':total_wall_ms,
          'samples':samples,'chunks':chunks,'transitions':transitions,'stock_reference_ms':stock_reference,'checks':2,'rank_choices':checks,
          'warning':'operator loop, common initialization and warmup excluded; not model E2E; independent processes may drift'}
        with Path(os.environ['NET_OUT']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result,indent=2),flush=True)
    ctx.close();finalize()

if __name__=='__main__':main()
