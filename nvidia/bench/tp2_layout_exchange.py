"""Correctness and 16-call timing for explicit TP2 logical row conversion."""
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
from stowaway_nvidia.layout_exchange import convert_rs_rows
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    raw = Path(os.environ['LAYOUT_MANIFEST']).read_bytes()
    design = json.loads(raw)
    target = design['targets'][int(os.environ['LAYOUT_INDEX'])]
    m, n, k = (target[x] for x in ('M', 'N', 'K'))
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    if dist.get_world_size() != 2:
        raise ValueError('TP2 required')
    ctx = DirectExchange(initialize(), rank, 2, m, k, n, torch.float16, reuse_rs_scratch=True)
    gen = torch.Generator(device='cuda').manual_seed(5500 + rank)
    x = torch.randn((m, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    w = torch.randn((n, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    contiguous = ctx.gemm_reduce_scatter(x, w)
    parts = [torch.empty_like(contiguous) for _ in range(2)]
    dist.all_gather(parts, contiguous)
    global_rows = torch.cat(parts)
    expected = global_rows[rank::2].contiguous()
    interleaved = convert_rs_rows(contiguous, 'contiguous_to_interleaved')
    torch.testing.assert_close(interleaved, expected, rtol=0, atol=0)
    restored = convert_rs_rows(interleaved, 'interleaved_to_contiguous')
    torch.testing.assert_close(restored, contiguous, rtol=0, atol=0)
    full_product = torch.empty((m, n), device='cuda', dtype=torch.float16)
    reference = torch.empty_like(contiguous)
    torch.matmul(x, w.T, out=full_product)
    dist.reduce_scatter_tensor(reference, full_product)
    torch.testing.assert_close(contiguous, reference, rtol=.02, atol=.02)

    def time_action(fn):
        for _ in range(design['warmup']):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(design['repeats']):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(design['batch']):
                fn()
            end.record()
            end.synchronize()
            elapsed = torch.tensor([start.elapsed_time(end) / design['batch']], device='cuda')
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
        return {'samples_ms': samples, 'median_ms': statistics.median(samples)}

    results = {}
    results['contiguous_to_interleaved'] = time_action(lambda: convert_rs_rows(contiguous, 'contiguous_to_interleaved'))
    results['interleaved_to_contiguous'] = time_action(lambda: convert_rs_rows(interleaved, 'interleaved_to_contiguous'))
    results['dx_plus_contiguous_to_interleaved'] = time_action(
        lambda: convert_rs_rows(ctx.gemm_reduce_scatter(x, w), 'contiguous_to_interleaved'))
    if rank == 0:
        payload = {'target': target, 'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'tp': 2,
                   'dtype': design['dtype'], 'batch': design['batch'], 'warmup': design['warmup'],
                   'repeats': design['repeats'], 'gpu': torch.cuda.get_device_name(),
                   'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda_build': torch.version.cuda,
                   'checks': ['logical mapping exact both directions', 'DX vs GEMM+NCCL'], 'results': results,
                   'warning': 'layout conversion only; does not decode FlashOverlap tile layout or run end-to-end'}
        with Path(os.environ['LAYOUT_OUT']).open('x') as f:
            json.dump(payload, f, indent=2)
            f.write('\n')
        print(json.dumps(payload), flush=True)
    ctx.close()
    finalize()


if __name__ == '__main__':
    main()
