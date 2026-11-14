"""TP2 direct PyTorch symmetric-memory operators vs DX and serial NCCL."""
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
    if arm not in ('stock', 'async_tp', 'dx'):
        raise ValueError('invalid arm')
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    if dist.get_world_size() != 2:
        raise ValueError('TP2 required')
    ctx = None
    if arm == 'async_tp':
        import torch.distributed._symmetric_memory as symm
        symm.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        if symm._is_test_mode:
            raise RuntimeError('symmetric-memory fallback test mode is enabled')
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
    weight = w.T

    def stock():
        if target['op'] == 'ag':
            dist.all_gather_into_tensor(partial, x)
            torch.matmul(partial, weight, out=output)
        else:
            torch.matmul(x, weight, out=partial)
            dist.reduce_scatter_tensor(output, partial)
        return output

    def run():
        if arm == 'stock':
            return stock()
        if arm == 'dx':
            return ctx.all_gather_gemm(x, w) if target['op'] == 'ag' else ctx.gemm_reduce_scatter(x, w)
        if target['op'] == 'ag':
            return torch.ops.symm_mem.fused_all_gather_matmul(
                x, [weight], 0, dist.group.WORLD.group_name, return_A=False)[1][0]
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(x, weight, 'sum', 0, dist.group.WORLD.group_name)

    for _ in range(2):
        reference = stock().clone()
        torch.testing.assert_close(run(), reference, rtol=.02, atol=.02)
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
                  'cuda_device_max_connections': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset'),
                  'samples_ms': samples, 'median_ms': statistics.median(samples),
                  'weight_layout': 'N,K contiguous, K,N transpose view',
                  'warning': 'direct fused operator; test mode disabled; not Inductor model graph; DX RS scratch; no E2E'}
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
