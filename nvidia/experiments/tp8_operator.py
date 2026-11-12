"""8-rank NVSHMEM AG/GEMM and GEMM/RS correctness plus paired timing."""
import datetime, json, os, random, statistics
from pathlib import Path
import torch
import torch.distributed as dist
from stowaway_nvidia.runtime import initialize, finalize
from stowaway_nvidia.dx8 import DirectExchange8

rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
cuda=initialize(expected_world_size=8)
rows=int(os.getenv('TP8_M','1024')); k=int(os.getenv('TP8_K','1024')); n=int(os.getenv('TP8_N','1024'))
checks=int(os.getenv('TP8_CHECKS','4')); repeats=int(os.getenv('TP8_REPEATS','6')); batch=int(os.getenv('TP8_BATCH','3'))
requested=os.getenv('TP8_ONLY','both')
if requested not in ('both','ag','rs'): raise ValueError('TP8_ONLY must be both, ag or rs')
torch.manual_seed(1123+rank)
x=torch.randn((rows//8,k),device='cuda',dtype=torch.bfloat16).mul_(.05)
a=torch.randn((rows,k),device='cuda',dtype=torch.bfloat16).mul_(.05)
w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16).mul_(.05)
ctx=DirectExchange8(cuda,rank,world,rows,k,n,x.dtype)
def stock_ag():
 full=torch.empty((rows,k),device='cuda',dtype=x.dtype);dist.all_gather_into_tensor(full,x);return full@w.t()
def stock_rs():
 partial=a@w.t();out=torch.empty((rows//8,n),device='cuda',dtype=x.dtype);dist.reduce_scatter_tensor(out,partial);return out
def dx_ag():return ctx.all_gather_gemm(x,w)
def dx_rs():return ctx.gemm_reduce_scatter(a,w)
try:
 for i in range(checks):
  x.normal_().mul_(.05);a.normal_().mul_(.05)
  for name,baseline,candidate in [('ag',stock_ag,dx_ag),('rs',stock_rs,dx_rs)]:
   if requested != 'both' and name != requested: continue
   ref=baseline();got=candidate()
   try:torch.testing.assert_close(got,ref,rtol=.03,atol=.02)
   except AssertionError as exc:
    print(f'FAIL rank={rank} turn={i} op={name} {exc}',flush=True);raise
  print(f'CHECK rank={rank} turn={i} PASS',flush=True)
 if os.environ.get('TP8_TRACE'):
  from torch.profiler import profile, ProfilerActivity, record_function
  op = os.environ.get('TP8_TRACE_OP', 'rs')
  if op not in ('ag', 'rs'): raise ValueError('TP8_TRACE_OP must be ag or rs')
  fn = dx_ag if op == 'ag' else dx_rs
  for _ in range(2): fn()
  torch.cuda.synchronize(); dist.barrier()
  if rank == 0:
   with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
    for _ in range(3):
     with record_function('DX8_'+op): fn()
    torch.cuda.synchronize()
   prof.export_chrome_trace(os.environ['TP8_TRACE'])
  else:
   for _ in range(3): fn()
   torch.cuda.synchronize()
  dist.barrier()
 def timed(fn):
  for _ in range(2):fn()
  torch.cuda.synchronize();dist.barrier(); vals=[]
  for _ in range(repeats):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record()
   for _ in range(batch):fn()
   end.record();end.synchronize()
   v=torch.tensor([start.elapsed_time(end)/batch],device='cuda');dist.all_reduce(v,op=dist.ReduceOp.MAX);vals.append(v.item())
  return vals
 result={}
 for op,stock,dx in [('ag',stock_ag,dx_ag),('rs',stock_rs,dx_rs)]:
  if requested != 'both' and op != requested: continue
  arms=[('stock',stock),('dx',dx)];random.Random(20261111+len(op)).shuffle(arms)
  samples={name:timed(fn) for name,fn in arms}
  result[op]={'samples_ms':samples,'stock_median_ms':statistics.median(samples['stock']), 'dx_median_ms':statistics.median(samples['dx'])}
  result[op]['speedup']=result[op]['stock_median_ms']/result[op]['dx_median_ms']
 if rank==0:
  path=Path(os.environ['TP8_OUT']);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('x') as f:json.dump({'M':rows,'K':k,'N':n,'world':world,'checks':checks,'repeats':repeats,'batch':batch,'schedule':ctx.schedule,'batch_ack':ctx.batch_ack,'result':result},f,indent=2)
  print('TP8_RESULT',json.dumps(result),flush=True)
finally:
 ctx.close();finalize()
