"""Validate one external backend and DX through the same automatic dispatcher."""
import datetime
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from stowaway_nvidia.multi_backend8 import MultiBackend8

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=240))
backend = os.environ['TEST_EXTERNAL_BACKEND']
plan = Path(os.environ['TEST_EXTERNAL_PLAN'])
dispatch = MultiBackend8(plan, 'operator_smoke')
generator = torch.Generator(device='cuda').manual_seed(3750 + rank)
results = []
for (site, op, m, n, k), choice in sorted(dispatch.plan.entries.items()):
    x = torch.randn((m // 8 if op == 'ag' else m, k), device='cuda', dtype=dispatch.dtype, generator=generator).mul_(.05)
    w = torch.randn((n, k), device='cuda', dtype=dispatch.dtype, generator=generator).mul_(.05)
    if op == 'ag':
        gathered = torch.empty((m, k), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(gathered, x)
        expected = gathered @ w.T
    else:
        partial = x @ w.T
        expected = torch.empty((m // 8, n), device=x.device, dtype=x.dtype)
        dist.reduce_scatter_tensor(expected, partial.contiguous())
    for turn in range(2):
        actual = dispatch.run(site, op, x, w)
        torch.testing.assert_close(actual, expected, rtol=.05, atol=.03)
    full_expected = expected.clone()
    # Downstream activations can remain live across later layers: a cached
    # communication buffer must not mutate a previously returned result.
    saved = actual
    other = dispatch.run(site, op, x * 0.5, w)
    torch.cuda.synchronize()
    torch.testing.assert_close(saved, full_expected, rtol=.05, atol=.03)
    del other, saved, full_expected
    def stock():
        if op == 'ag':
            dist.all_gather_into_tensor(gathered, x)
            return gathered @ w.T
        dist.reduce_scatter_tensor(expected, (x @ w.T).contiguous())
        return expected
    timings = {}
    for label, fn in [('stock', stock), ('selected', lambda: dispatch.run(site, op, x, w))]:
        for _ in range(4): fn()
        torch.cuda.synchronize(); dist.barrier()
        samples = []
        for _ in range(8):
            dist.barrier()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(16): fn()
            end.record(); end.synchronize()
            elapsed = torch.tensor([begin.elapsed_time(end) / 16], device='cuda')
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
        timings[label] = {'samples_ms': samples, 'median_ms': statistics.median(samples)}
    results.append({'site': site, 'op': op, 'shape': [m, n, k], 'choice': choice, 'passes': 2,
                    'measurements': timings, 'speedup': timings['stock']['median_ms'] / timings['selected']['median_ms']})
    dist.barrier()
all_hits = [None] * 8
dist.all_gather_object(all_hits, {str(k): v for k, v in dispatch.hits.items()})
if rank == 0:
    dest = Path(os.environ['TEST_EXTERNAL_OUT'])
    with dest.open('x') as f:
        json.dump({'backend': backend, 'dtype': dispatch.plan.dtype, 'results': results, 'rank_hits': all_hits}, f, indent=2)
        f.write('\n')
    print(f'{backend} TP8 automatic dispatcher PASS', flush=True)
dispatch.close()
dist.destroy_process_group()
