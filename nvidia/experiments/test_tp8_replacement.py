"""Exercise TP8 Async-TP/DX/stock dispatch for one model inventory."""
import datetime, json, os
from pathlib import Path
import torch
import torch.distributed as dist
from stowaway_nvidia.replacement_plan8 import ReplacementExecutor

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
exe = ReplacementExecutor(os.environ['REPLACEMENT_PLAN'])
gen = torch.Generator(device='cuda').manual_seed(3210 + rank)
for entry in exe.plan['entries']:
    m,n,k,op = (entry[x] for x in ('M','N','K','op'))
    x = torch.randn((m//8 if op == 'ag' else m,k), generator=gen, device='cuda', dtype=torch.bfloat16).mul_(.05)
    w = torch.randn((n,k), generator=gen, device='cuda', dtype=torch.bfloat16).mul_(.05)
    if op == 'ag':
        full = torch.empty((m,k), device='cuda', dtype=x.dtype)
        dist.all_gather_into_tensor(full,x)
        ref = full @ w.T
    else:
        partial = x @ w.T
        ref = torch.empty((m//8,n), device='cuda', dtype=x.dtype)
        dist.reduce_scatter_tensor(ref,partial)
    for _ in range(2):
        actual = exe.run(entry['site'],op,x,w)
        torch.testing.assert_close(actual,ref,rtol=.03,atol=.02)
    dist.barrier()
    if rank == 0: print('PASS',entry,flush=True)
report={'rank':rank,'hits':exe.hits,'fallbacks':exe.fallbacks,'plan_sha256':exe.digest}
reports=[None]*8
dist.all_gather_object(reports,report,group=exe.group)
exe.close()
if rank == 0:
    with Path(os.environ['REPLACEMENT_RESULT']).open('x') as f: json.dump({'reports':reports},f,indent=2)
dist.barrier()
dist.destroy_process_group()
