"""Compare TP8 standalone reduce-scatter with NCCL under repeated reuse."""
import datetime
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from stowaway_nvidia.brs8 import ReduceScatter8
from stowaway_nvidia.runtime import initialize, finalize

rank = int(os.environ['RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
cuda = initialize(expected_world_size=8)
n = int(os.getenv('CHECK_NUMEL', '1048576'))
turns = int(os.getenv('CHECK_TURNS', '8'))
torch.manual_seed(2381+rank)
ctx = ReduceScatter8(cuda,rank,8,n,torch.bfloat16)
records=[]
try:
    for i in range(turns):
        x = torch.randn(n,device='cuda',dtype=torch.bfloat16).mul_(.05)
        ref = torch.empty(n//8,device='cuda',dtype=x.dtype)
        dist.reduce_scatter_tensor(ref,x)
        got, handle = ctx(x, defer_fence=True)
        handle.wait()
        torch.testing.assert_close(got,ref,rtol=.03,atol=.02)
        records.append(float((got.float()-ref.float()).abs().max()))
        print(f'BRS_CHECK rank={rank} turn={i} PASS',flush=True)
    all_records=[None]*8
    dist.all_gather_object(all_records,records)
    if rank==0:
        path=Path(os.environ['CHECK_OUT']);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f: json.dump({'numel':n,'turns':turns,'max_abs_by_rank':all_records},f,indent=2)
finally:
    ctx.close();finalize()
