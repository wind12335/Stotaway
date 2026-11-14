"""Compare FlashOverlap+upstream ReorderRMSNorm against DX+matching RMSNorm."""
import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl


@triton.jit
def _rms_norm(x, out, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(x + row * N + col, col < N, 0).to(tl.float32)
    square = tl.sum(value * value, 0) / N
    result = value * tl.rsqrt(square + EPS)
    tl.store(out + row * N + col, result, col < N)


def worker(rank, uid, target, config, order, shared):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT='29668')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    sys.path.insert(0, os.getcwd())
    from RMSNorm import ReorderRMSNorm
    from RowParallelLinear import OverlapRowParallelLayer
    from stowaway_nvidia.dx import DirectExchange
    from stowaway_nvidia.runtime import initialize, finalize
    m, n, k = (target[x] for x in ('M', 'N', 'K'))
    gen = torch.Generator(device='cuda').manual_seed(5500 + rank)
    x = torch.randn((m, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    w = torch.randn((n, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    layer = OverlapRowParallelLayer(rank, 2, k, n, m, config, 'reduce_scatter', uid)
    layer.weight = torch.nn.Parameter(w)
    norm = ReorderRMSNorm(n, m // 2, config['BM'] // 2, config['BN'], config['hint']).cuda()
    norm.weight = torch.nn.Parameter(torch.ones((n,), device='cuda', dtype=torch.float16))
    dx = DirectExchange(initialize(), rank, 2, m, k, n, torch.float16, reuse_rs_scratch=True)

    def flash():
        return norm(layer(x))

    def direct():
        raw = dx.gemm_reduce_scatter(x, w, output_layout='interleaved')
        out = torch.empty_like(raw)
        _rms_norm[(m // 2,)](raw, out, n, 1e-5, triton.next_power_of_2(n), num_warps=8)
        return out

    # Compare both methods to a serial reference with the same rank-interleaved rows.
    partial = torch.matmul(x, w.T)
    shard = torch.empty((m // 2, n), device='cuda', dtype=torch.float16)
    dist.reduce_scatter_tensor(shard, partial)
    shards = [torch.empty_like(shard) for _ in range(2)]
    dist.all_gather(shards, shard)
    logical = torch.cat(shards)[rank::2].contiguous()
    expected = (logical.float() * torch.rsqrt(logical.float().square().mean(-1, keepdim=True) + 1e-5)).half()
    for _ in range(2):
        torch.testing.assert_close(flash(), expected, atol=.05, rtol=.05)
        torch.testing.assert_close(direct(), expected, atol=.05, rtol=.05)
    medians = {}
    paths = {'flash_plus_reorder_norm': flash, 'dx_interleaved_plus_norm': direct}
    for name in order:
        fn = paths[name]
        for _ in range(32):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(8):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(16):
                fn()
            end.record()
            end.synchronize()
            elapsed = torch.tensor([begin.elapsed_time(end) / 16], device='cuda')
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
        medians[name] = {'samples_ms': samples, 'median_ms': statistics.median(samples)}
    if rank == 0:
        shared['result'] = medians
    dx.close()
    finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--m', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--order', choices=('flash-first', 'dx-first'), default='flash-first')
    args = ap.parse_args()
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw)
    target = {'M': args.m, 'N': 4096, 'K': 4096}
    # The upstream library registers the NCCL-ID generator in the parent;
    # spawned workers load the same library through RMSNorm/RowParallelLinear.
    torch.ops.load_library(str((Path.cwd() / '../build/lib/libst_pybinding.so').resolve()))
    uid = torch.ops.flashoverlap_op.generate_nccl_id()
    manager = mp.Manager()
    shared = manager.dict()
    order = (['flash_plus_reorder_norm', 'dx_interleaved_plus_norm'] if args.order == 'flash-first'
             else ['dx_interleaved_plus_norm', 'flash_plus_reorder_norm'])
    mp.spawn(worker, args=(uid, target, config, order, shared), nprocs=2)
    result = {'target': target, 'tp': 2, 'dtype': 'float16', 'warmup': 32, 'batch': 16, 'repeats': 8,
              'config_sha256': hashlib.sha256(config_raw).hexdigest(), 'gpu': torch.cuda.get_device_name(0),
              'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda_build': torch.version.cuda,
              'checks': 'both ranks compare both outputs with same reference twice', 'order': order,
              'results': dict(shared)['result'],
              'warning': 'different RMSNorm implementations; paired paths within a process; no model E2E'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
