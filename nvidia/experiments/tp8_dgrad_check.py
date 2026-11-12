"""Eight-rank AG+dgrad/wgrad comparison, including repeated buffer reuse."""
import datetime
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from stowaway_nvidia.runtime import initialize, finalize
from stowaway_nvidia.dx8 import DirectExchange8

rank = int(os.environ['RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
cuda = initialize(expected_world_size=8)
m = int(os.getenv('CHECK_M','1024'))
k = int(os.getenv('CHECK_K','1024'))
n = int(os.getenv('CHECK_N','512'))
turns = int(os.getenv('CHECK_TURNS','8'))
torch.manual_seed(8123+rank)
ctx = DirectExchange8(cuda, rank, 8, m, k, n, torch.bfloat16)
records = []
try:
    for turn in range(turns):
        grad = torch.randn((m//8,k), device='cuda', dtype=torch.bfloat16).mul_(.05)
        weight = torch.randn((k,n), device='cuda', dtype=torch.bfloat16).mul_(.05)
        activation = torch.randn((m,n), device='cuda', dtype=torch.bfloat16).mul_(.05)
        full = torch.empty((m,k), device='cuda', dtype=torch.bfloat16)
        dist.all_gather_into_tensor(full, grad)
        ref_input = full @ weight
        ref_weight = full.t() @ activation
        got_input, got_weight, got_full = ctx.all_gather_dgrad(grad, weight, activation)
        torch.testing.assert_close(got_full, full, rtol=0, atol=0)
        torch.testing.assert_close(got_input, ref_input, rtol=.03, atol=.02)
        torch.testing.assert_close(got_weight, ref_weight, rtol=.03, atol=.02)
        torch.cuda.synchronize()
        records.append({'turn':turn, 'input_max_abs':float((got_input.float()-ref_input.float()).abs().max()),
                        'weight_max_abs':float((got_weight.float()-ref_weight.float()).abs().max())})
        print(f'DGRAD_CHECK rank={rank} turn={turn} PASS', flush=True)
    gathered = [None]*8
    dist.all_gather_object(gathered, records)
    if rank == 0:
        path = Path(os.environ['CHECK_OUT']); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f:
            json.dump({'M':m,'K':k,'N':n,'turns':turns,'ranks':gathered}, f, indent=2)
finally:
    ctx.close()
    finalize()
