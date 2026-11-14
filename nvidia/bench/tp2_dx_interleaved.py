"""TP2 DX RS directly emits rank-interleaved logical rows."""
import datetime
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import triton
from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import initialize, finalize


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    raw = Path(os.environ['LAYOUT_MANIFEST']).read_bytes()
    design = json.loads(raw)
    target = design['targets'][int(os.environ['LAYOUT_INDEX'])]
    m, n, k = (target[x] for x in ('M', 'N', 'K'))
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    ctx = DirectExchange(initialize(), rank, 2, m, k, n, torch.float16, reuse_rs_scratch=True)
    gen = torch.Generator(device='cuda').manual_seed(5500 + rank)
    x = torch.randn((m, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    w = torch.randn((n, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    partial = torch.matmul(x, w.T)
    local_ref = torch.empty((m // 2, n), device='cuda', dtype=torch.float16)
    dist.reduce_scatter_tensor(local_ref, partial)
    ranks = [torch.empty_like(local_ref) for _ in range(2)]
    dist.all_gather(ranks, local_ref)
    expected = torch.cat(ranks)[rank::2].contiguous()
    for _ in range(2):
        actual = ctx.gemm_reduce_scatter(x, w, output_layout='interleaved')
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    old = ctx.gemm_reduce_scatter(x, w, output_layout='interleaved')
    old_copy = old.clone()
    ctx.gemm_reduce_scatter(x * .9, w, output_layout='interleaved')
    torch.testing.assert_close(old, old_copy, rtol=0, atol=0)
    for _ in range(design['warmup']):
        ctx.gemm_reduce_scatter(x, w, output_layout='interleaved')
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(design['repeats']):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(design['batch']):
            ctx.gemm_reduce_scatter(x, w, output_layout='interleaved')
        end.record()
        end.synchronize()
        t = torch.tensor([begin.elapsed_time(end) / design['batch']], device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(t.item())
    if rank == 0:
        result = {'target': target, 'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'tp': 2,
                  'dtype': 'float16', 'batch': design['batch'], 'warmup': design['warmup'],
                  'repeats': design['repeats'], 'gpu': torch.cuda.get_device_name(),
                  'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda_build': torch.version.cuda,
                  'checks': ['rank interleaved reference twice', 'retained output'],
                  'samples_ms': samples, 'median_ms': statistics.median(samples),
                  'warning': 'opt-in TP2 forward logical rows; not raw FlashOverlap tile packing or E2E'}
        with Path(os.environ['LAYOUT_OUT']).open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps(result), flush=True)
    ctx.close()
    finalize()


if __name__ == '__main__':
    main()
