"""TP2 FP16 stock/DX RS arm, reference for FlashOverlap replacement trial."""
import datetime,hashlib,json,os,statistics,time
from pathlib import Path
import torch
import torch.distributed as dist
import triton
from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import initialize,finalize


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    manifest_raw=Path(os.environ['REPLACE_MANIFEST']).read_bytes();manifest=json.loads(manifest_raw)
    target=manifest['targets'][int(os.environ['REPLACE_INDEX'])];m,n,k=(target[x] for x in ('M','N','K'))
    arm=os.environ['REPLACE_ARM'];batch=manifest['batch'];warm=manifest['warmup'];repeats=manifest['repeats']
    if arm not in ('stock','dx'):raise ValueError('invalid arm')
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180));ctx=None
    if arm=='dx':ctx=DirectExchange(initialize(),rank,2,m,k,n,torch.float16,reuse_rs_scratch=True)
    gen=torch.Generator(device='cuda').manual_seed(5500+rank)
    x=torch.randn((m,k),device='cuda',dtype=torch.float16,generator=gen)*.05
    w=torch.randn((n,k),device='cuda',dtype=torch.float16,generator=gen)*.05
    partial=torch.empty((m,n),device='cuda',dtype=torch.float16);out=torch.empty((m//2,n),device='cuda',dtype=torch.float16)
    def stock():torch.matmul(x,w.T,out=partial);dist.reduce_scatter_tensor(out,partial);return out
    if arm=='dx':
        def run():return ctx.gemm_reduce_scatter(x,w)
    else:run=stock
    for _ in range(2):
        reference=stock().clone();actual=run();torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
    for _ in range(warm):run()
    torch.cuda.synchronize();dist.barrier();start_wall=time.perf_counter();samples=[]
    for _ in range(repeats):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
        for _ in range(batch):run()
        end.record();end.synchronize();v=torch.tensor([begin.elapsed_time(end)/batch],device='cuda');dist.all_reduce(v,op=dist.ReduceOp.MAX);samples.append(v.item())
    if rank==0:
        result={'classification':'single arm TP2 RS replacement measurement','arm':arm,'target':target,
                'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'dtype':'float16','tp':2,'batch':batch,'warmup':warm,'repeats':repeats,
                'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'triton':triton.__version__,'cuda_build':torch.version.cuda,
                'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset'),
                'samples_ms':samples,'median_ms':statistics.median(samples),'measurement_wall_ms':(time.perf_counter()-start_wall)*1000,
                'checks':2,'output_semantics':'contiguous rank shard','dx_scratch':arm=='dx',
                'warning':'independent process; no E2E; FlashOverlap reorders outputs and requires separate normalization'}
        with Path(os.environ['REPLACE_OUT']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result),flush=True)
    if ctx is not None:ctx.close();finalize()
    else:dist.destroy_process_group()

if __name__=='__main__':main()
