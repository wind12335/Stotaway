"""One arm of an exploratory same-shape Flux/DX/stock TP2 comparison."""
import datetime
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import triton


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    raw = Path(os.environ['REPLACE_MANIFEST']).read_bytes()
    design = json.loads(raw)
    target = design['targets'][int(os.environ['REPLACE_INDEX'])]
    arm = os.environ['REPLACE_ARM']
    m, n, k = (target[x] for x in ('M', 'N', 'K'))
    if arm not in ('stock', 'flux', 'dx') or dist.is_initialized():
        raise ValueError('invalid arm or distributed state')
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    if dist.get_world_size() != 2 or os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != '1':
        raise ValueError('requires TP2 and CUDA_DEVICE_MAX_CONNECTIONS=1')
    ctx = None
    if arm == 'flux':
        import flux
        flux.init_flux_shm(dist.group.WORLD)
        if target['op'] == 'ag':
            ctx = flux.AGKernel(dist.group.WORLD, 1, m, n, k, torch.bfloat16,
                                output_dtype=torch.bfloat16, use_pdl=False)
            ag_option = flux.AllGatherOption()
        else:
            ctx = flux.GemmRS(dist.group.WORLD, 1, (m + 1023) // 1024 * 1024, n,
                              torch.bfloat16, torch.bfloat16, transpose_weight=False,
                              fuse_reduction=False, ring_reduction=False)
            rs_option = flux.ReduceScatterOption()
    elif arm == 'dx':
        from stowaway_nvidia.dx import DirectExchange
        from stowaway_nvidia.runtime import initialize, finalize
        ctx = DirectExchange(initialize(), rank, 2, m, k, n, torch.bfloat16,
                             reuse_rs_scratch=(target['op'] == 'rs'))
    gen = torch.Generator(device='cuda').manual_seed(8120 + rank)
    x = torch.randn((m // 2 if target['op'] == 'ag' else m, k), device='cuda',
                    dtype=torch.bfloat16, generator=gen) * .05
    w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16, generator=gen) * .05
    partial = torch.empty((m, k if target['op'] == 'ag' else n), device='cuda', dtype=torch.bfloat16)
    output = torch.empty((m if target['op'] == 'ag' else m // 2, n), device='cuda', dtype=torch.bfloat16)

    def stock():
        if target['op'] == 'ag':
            dist.all_gather_into_tensor(partial, x)
            torch.matmul(partial, w.T, out=output)
        else:
            torch.matmul(x, w.T, out=partial)
            dist.reduce_scatter_tensor(output, partial)
        return output

    def run():
        if arm == 'stock':
            return stock()
        if arm == 'dx':
            return ctx.all_gather_gemm(x, w) if target['op'] == 'ag' else ctx.gemm_reduce_scatter(x, w)
        if target['op'] == 'ag':
            return ctx.forward(x, w, output=output, transpose_weight=False, all_gather_option=ag_option)
        return ctx.forward(x, w, fast_accum=False, reduce_scatter_option=rs_option)

    for _ in range(2):
        reference = stock().clone()
        actual = run()
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)
    for _ in range(design['warmup']):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(design['repeats']):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(design['batch']):
            run()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([start.elapsed_time(end) / design['batch']], device='cuda')
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    if rank == 0:
        result = {'arm': arm, 'target': target, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                  'tp': 2, 'dtype': 'bfloat16', 'batch': design['batch'], 'warmup': design['warmup'],
                  'repeats': design['repeats'], 'checks': 2, 'gpu': torch.cuda.get_device_name(),
                  'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda_build': torch.version.cuda,
                  'cuda_device_max_connections': os.environ['CUDA_DEVICE_MAX_CONNECTIONS'],
                  'samples_ms': samples, 'median_ms': statistics.median(samples),
                  'flux_path': flux.__file__ if arm == 'flux' else None,
                  'warning': 'exploratory; independent processes; CUDA IPC Flux with shared NCCL; RS DX scratch; no E2E'}
        with Path(os.environ['REPLACE_OUT']).open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps(result), flush=True)
    if arm == 'dx':
        ctx.close()
        finalize()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
