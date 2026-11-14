"""Measure repeated independent AG/RS pairs, direct Async-TP versus DX substitution."""
import datetime, json, os, statistics
from pathlib import Path
import torch
import torch.distributed as dist
from stowaway_nvidia.runtime import initialize
from stowaway_nvidia.dx8 import DirectExchange8
from stowaway_nvidia.async_tp_exchange import AsyncTPExchange

rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
world=dist.get_world_size()
if world!=8:raise RuntimeError('TP8 required')
async_ex=AsyncTPExchange(dist.group.WORLD)
cuda=initialize(expected_world_size=8)
m,k,n=8192,1792,4096
ctx=DirectExchange8(cuda,rank,8,m,k,n,torch.bfloat16)
g=torch.Generator(device='cuda').manual_seed(7610+rank)
x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=g).mul_(.05)
w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16,generator=g).mul_(.05)
partial=x@w.T;ref=torch.empty((m//8,n),device='cuda',dtype=x.dtype)
dist.reduce_scatter_tensor(ref,partial)
for name,fn in [('async_tp',async_ex.gemm_reduce_scatter),('dx',ctx.gemm_reduce_scatter)]:
 for _ in range(2):torch.testing.assert_close(fn(x,w),ref,rtol=.03,atol=.02)
samples={}
for name,fn in [('async_tp',async_ex.gemm_reduce_scatter),('dx',ctx.gemm_reduce_scatter)]:
 for _ in range(4):fn(x,w)
 torch.cuda.synchronize();dist.barrier()
 vals=[]
 for _ in range(8):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(16):fn(x,w)
  b.record();b.synchronize();value=torch.tensor([a.elapsed_time(b)/16],device='cuda')
  dist.all_reduce(value,op=dist.ReduceOp.MAX);vals.append(value.item())
 samples[name]={'samples_ms':vals,'median_ms':statistics.median(vals)}
if rank==0:
 out={'tp':8,'dtype':'bfloat16','M':m,'N':n,'K':k,'op':'rs','batch':16,'measurements':samples,
      'dx_over_async_speedup':samples['async_tp']['median_ms']/samples['dx']['median_ms'],
      'note':'same process direct operator timing; not model E2E or predictor'}
 with Path(os.environ['REPLACEMENT_BENCH_OUT']).open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
ctx.close();torch.cuda.synchronize();dist.barrier()
import nvshmem.core
nvshmem.core.finalize();dist.destroy_process_group()
