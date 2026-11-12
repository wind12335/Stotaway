"""A800 measurements required by the original costmodel: G, C, and Cdx."""
import datetime
import json
import os
import statistics
from pathlib import Path
import torch
import torch.distributed as dist
from stowaway_nvidia.brs8 import ReduceScatter8
from stowaway_nvidia.runtime import initialize, finalize

rank = int(os.environ['RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
cuda = initialize(expected_world_size=8)
m,k,n=(int(os.environ[x]) for x in ('TP8_M','TP8_K','TP8_N'))
op=os.environ['TP8_OP']
if op not in ('ag','rs'): raise ValueError('TP8_OP must be ag or rs')
dtype=torch.bfloat16
torch.manual_seed(511+rank)
x=torch.randn((m,k),device='cuda',dtype=dtype).mul_(.05)
w=torch.randn((n,k),device='cuda',dtype=dtype).mul_(.05)
gemm_out=torch.empty((m,n),device='cuda',dtype=dtype)
rs_input=torch.randn((m,n),device='cuda',dtype=dtype).mul_(.05) if op=='rs' else None
rs_out=torch.empty((m//8,n),device='cuda',dtype=dtype) if op=='rs' else None
ag_in=torch.randn((m//8,k),device='cuda',dtype=dtype).mul_(.05) if op=='ag' else None
ag_out=torch.empty((m,k),device='cuda',dtype=dtype) if op=='ag' else None
ctx=ReduceScatter8(cuda,rank,8,m*n,dtype) if op=='rs' else None

def gemm(): torch.matmul(x,w.t(),out=gemm_out)
def comm():
    if op=='rs': dist.reduce_scatter_tensor(rs_out,rs_input)
    else: dist.all_gather_into_tensor(ag_out,ag_in)
def dx_comm(): ctx(rs_input,rs_out)
def timed(fn,batch):
    for _ in range(3): fn()
    torch.cuda.synchronize();dist.barrier()
    values=[]
    for _ in range(6):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch): fn()
        end.record();end.synchronize()
        t=torch.tensor([start.elapsed_time(end)/batch],device='cuda')
        dist.all_reduce(t,op=dist.ReduceOp.MAX)
        values.append(t.item())
    return values
try:
    if ctx:
        ref=torch.empty_like(rs_out);dist.reduce_scatter_tensor(ref,rs_input)
        got,_=ctx(rs_input)
        torch.testing.assert_close(got.view_as(ref),ref,rtol=.03,atol=.02)
    result={}
    for name,fn in [('G',gemm),('C',comm)]+([('Cdx',dx_comm)] if ctx else []):
        samples=timed(fn,1)
        result[name]={'samples_ms':samples,'median_ms':statistics.median(samples)}
    if rank==0:
        path=Path(os.environ['TP8_OUT']);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump({'M':m,'K':k,'N':n,'op':op,'tp':8,'components':result},f,indent=2)
        print('TP8_COMPONENTS',json.dumps(result),flush=True)
finally:
    if ctx:ctx.close()
    finalize()
