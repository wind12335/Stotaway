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
    if arm not in ('stock','adaptive','dx','dx_prealloc','dx_partial'):raise ValueError('invalid arm')
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
    if arm in ('dx_prealloc','dx_partial'):
        if op!='rs':raise ValueError('RS only')
        from stowaway_nvidia.dx import _sum_pair, _publish
        import triton as tr
        permanent_partial=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        permanent_output=torch.empty((m//2,n),device='cuda',dtype=torch.bfloat16)
        def preallocated_dx():
            turn=ctx._next_round('rs_round');current=torch.cuda.current_stream()
            result_output=(torch.empty((m//2,n),device='cuda',dtype=torch.bfloat16) if arm=='dx_partial' else permanent_output)
            half=ctx.block_rows;peer=ctx.peer;rank=ctx.rank
            torch.matmul(full[peer*half:(peer+1)*half],w.T,out=permanent_partial[peer*half:(peer+1)*half])
            ready=torch.cuda.Event();ready.record(current)
            torch.matmul(full[rank*half:(rank+1)*half],w.T,out=permanent_partial[rank*half:(rank+1)*half])
            size=half*n*permanent_partial.element_size()
            with torch.cuda.stream(ctx.send):
                ctx.send.wait_event(ready)
                if turn>1:ctx.cuda.wait(ctx.rs_ack[rank].data_ptr()+peer*4,turn-1,ctx.send)
                ctx.cuda.copy(ctx.rs_slots[peer]+rank*size,permanent_partial.data_ptr()+peer*size,size,ctx.send)
                ctx.cuda.copy(ctx.rs_arrival[peer]+rank*4,ctx.tickets.data_ptr()+turn*4,4,ctx.send)
            ctx.cuda.wait(ctx.rs_arrival[rank].data_ptr()+peer*4,turn,current)
            count=half*n
            _sum_pair[(tr.cdiv(count,1024),)](result_output,permanent_partial[rank*half:(rank+1)*half],ctx.rs_slots[rank][peer],count,SCALE=1.0,BLOCK=1024)
            _publish[(1,)](ctx.rs_ack[peer],rank,turn,num_warps=1)
            done=torch.cuda.Event();done.record(ctx.send);current.wait_event(done)
            return result_output
        dx=preallocated_dx
    for _ in range(2):
        reference=stock().clone();actual=dx();torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
    if arm=='dx_partial':
        saved=dx();snapshot=saved.clone();dx();torch.cuda.synchronize()
        torch.testing.assert_close(saved,snapshot,rtol=0,atol=0)
    if arm=='dx_partial' and os.environ.get('PARTIAL_STRESS')=='1':
        # Keep every output alive; change activations between asynchronous
        # calls and only compare after all calls have been submitted.
        refs=[];outs=[]
        for _ in range(12):
            full.normal_(generator=gen).mul_(.05)
            refs.append(stock().clone());outs.append(dx())
        torch.cuda.synchronize()
        for got,want in zip(outs,refs):torch.testing.assert_close(got,want,rtol=.02,atol=.02)
    # Common warmup is equal for both arms, including potential context effects.
    for _ in range(warmup):stock();dx()
    torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
    samples=[];choice='stock';probe_ms=0.;reason='baseline arm';predicted_saving=0.
    if arm=='adaptive':
        funcs={'stock':stock,'dx':dx};order=['stock','dx'];random.Random(seed).shuffle(order)
        for i in range(blocks):
            times={}
            for name in (order if i%2==0 else order[::-1]):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                for _ in range(batch):funcs[name]()
                end.record();end.synchronize();times[name]=maximum(begin.elapsed_time(end)/batch)
            samples.append(times)
        torch.cuda.synchronize();probe_ms=maximum((time.perf_counter()-start)*1000)
        savings=[x['stock']-x['dx'] for x in samples]
        center=statistics.median(savings);mad=statistics.median(abs(x-center) for x in savings)
        predicted_saving=center-mad
        # The elapsed max-rank wall includes probe and synchronization. The
        # saved compute must recover the entire measured incremental probe cost.
        choice='dx' if predicted_saving*calls>probe_ms else 'stock'
        reason='median paired saving minus MAD, after probe cost'
        choice_obj=[choice if rank==0 else None];dist.broadcast_object_list(choice_obj,src=0);choice=choice_obj[0]
    if arm in ('dx','dx_prealloc','dx_partial'):choice='dx'
    selected=dx if choice=='dx' else stock
    torch.cuda.synchronize();allocator_before=torch.cuda.memory_stats();run_start=time.perf_counter();gpu_begin,gpu_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    gpu_begin.record()
    for _ in range(calls):selected()
    gpu_end.record();submission_wall_ms=(time.perf_counter()-run_start)*1000;gpu_end.synchronize();run_wall_ms=(time.perf_counter()-run_start)*1000
    allocator_after=torch.cuda.memory_stats()
    total_wall_ms=(time.perf_counter()-start)*1000
    gpu_ms=maximum(gpu_begin.elapsed_time(gpu_end));run_wall_ms=maximum(run_wall_ms);total_wall_ms=maximum(total_wall_ms)
    local_stats={'rank':rank,'submission_wall_ms':submission_wall_ms,'run_wall_ms':run_wall_ms,
        'allocated_before':allocator_before.get('allocated_bytes.all.current',0),'allocated_after':allocator_after.get('allocated_bytes.all.current',0),
        'reserved_before':allocator_before.get('reserved_bytes.all.current',0),'reserved_after':allocator_after.get('reserved_bytes.all.current',0),
        'num_alloc_retries_delta':allocator_after.get('num_alloc_retries',0)-allocator_before.get('num_alloc_retries',0),
        'num_device_alloc_delta':allocator_after.get('num_device_alloc',0)-allocator_before.get('num_device_alloc',0)}
    rank_stats=[None]*2;dist.all_gather_object(rank_stats,local_stats)
    checks=[None]*2;dist.all_gather_object(checks,choice)
    if checks!=[choice,choice]:raise ValueError('rank decision mismatch')
    if rank==0:
        result={'classification':'executed same-process probe and selected TP2 operator workload',
          'arm':arm,'model':model,'site':site,'op':op,'M':m,'N':n,'K':k,'calls':calls,'batch':batch,
          'warmup':warmup,'probe_blocks':blocks,'seed':seed,'gpu':torch.cuda.get_device_name(),
          'torch':str(torch.__version__),'triton':triton.__version__,'cuda_build':torch.version.cuda,
          'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset'),
          'choice':choice,'decision_reason':reason,'predicted_conservative_saving_ms_per_call':predicted_saving,
          'probe_max_rank_wall_ms':probe_ms,'chosen_gpu_max_rank_ms':gpu_ms,
          'chosen_run_max_rank_wall_ms':run_wall_ms,'total_max_rank_wall_ms':total_wall_ms,
          'samples':samples,'rank_stats':rank_stats,'checks':2,'retained_output_stress':os.environ.get('PARTIAL_STRESS')=='1' and arm=='dx_partial','rank_choices':checks,
          'warning':'operator loop, common initialization and warmup excluded; not model E2E; independent processes may drift'}
        with Path(os.environ['NET_OUT']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result,indent=2),flush=True)
    ctx.close();finalize()

if __name__=='__main__':main()
