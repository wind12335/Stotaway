"""One explicit AG+GEMM arm for same-shape replacement comparison."""
import datetime,hashlib,json,os,statistics,time
from pathlib import Path
import torch
import torch.distributed as dist
import triton


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    arm=os.environ['REPLACE_ARM'];manifest_raw=Path(os.environ['REPLACE_MANIFEST']).read_bytes();manifest=json.loads(manifest_raw)
    target=manifest['targets'][int(os.environ['REPLACE_INDEX'])]
    m,n,k=(target[x] for x in ('M','N','K'));batch=manifest['batch'];repeats=manifest['repeats'];warm=manifest['warmup']
    if arm not in ('stock','triton_dist','dx'):raise ValueError('unknown arm')
    if arm=='triton_dist':
        import triton_dist
        from triton_dist.utils import initialize_distributed,finalize_distributed
        from triton_dist.kernels.nvidia import create_ag_gemm_context,ag_gemm
        group=initialize_distributed()
        ctx=create_ag_gemm_context(m,n*2,k,torch.bfloat16,rank,2,num_local_ranks=2)
    else:
        from stowaway_nvidia.runtime import initialize,finalize
        dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180));group=dist.group.WORLD
        ctx=None
        if arm=='dx':
            from stowaway_nvidia.dx import DirectExchange
            ctx=DirectExchange(initialize(),rank,2,m,k,n,torch.bfloat16)
    gen=torch.Generator(device='cuda').manual_seed(4200+rank)
    shard=torch.randn((m//2,k),generator=gen,device='cuda',dtype=torch.bfloat16)*.05
    w=torch.randn((n,k),generator=gen,device='cuda',dtype=torch.bfloat16)*.05
    full=torch.empty((m,k),device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    def stock():dist.all_gather_into_tensor(full,shard,group=group);torch.matmul(full,w.T,out=out);return out
    if arm=='triton_dist':
        def run():return ag_gemm(shard,w.T,ctx=ctx,autotune=False)
    elif arm=='dx':
        def run():return ctx.all_gather_gemm(shard,w)
    else:run=stock
    for _ in range(2):
        reference=stock().clone();actual=run();torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
    for _ in range(warm):run()
    torch.cuda.synchronize();dist.barrier(group=group);t0=time.perf_counter();samples=[]
    for _ in range(repeats):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(batch):run()
        end.record();end.synchronize();value=torch.tensor([start.elapsed_time(end)/batch],device='cuda')
        dist.all_reduce(value,op=dist.ReduceOp.MAX,group=group);samples.append(value.item())
    duration=(time.perf_counter()-t0)*1000
    if rank==0:
        result={'classification':'single arm TP2 AG replacement operator measurement','arm':arm,'target':target,
          'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'dtype':'bfloat16','tp':2,'batch':batch,'warmup':warm,'repeats':repeats,
          'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'triton':triton.__version__,'cuda_build':torch.version.cuda,
          'triton_dist_path':triton_dist.__file__ if arm=='triton_dist' else None,
          'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset'),
          'samples_ms':samples,'median_ms':statistics.median(samples),'measurement_wall_ms':duration,'checks':2,
          'warning':'separate processes; no plan or E2E execution; target was chosen from prior observed data'}
        with Path(os.environ['REPLACE_OUT']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result),flush=True)
    if ctx is not None:ctx.finalize() if arm=='triton_dist' else ctx.close()
    if arm=='triton_dist':finalize_distributed()
    elif arm=='dx':finalize()
    else:dist.destroy_process_group()

if __name__=='__main__':main()
