"""TP8 Triton-distributed AG/RS against stock for the original MNK inventory."""
import hashlib
import json
import os
import random
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton_dist
from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
from triton_dist.kernels.nvidia.gemm_reduce_scatter import create_gemm_rs_context, gemm_rs
from triton_dist.utils import initialize_distributed, finalize_distributed


def main():
    rank = int(os.environ['LOCAL_RANK'])
    def trace(stage):
        if os.environ.get('TP8_TRACE'):
            print(f'rank={rank} stage={stage}', flush=True)
    torch.cuda.set_device(rank)
    trace('initializing')
    group = initialize_distributed()
    trace('initialized')
    if dist.get_world_size() != 8:
        raise ValueError('requires TP8')
    raw = Path(os.environ['TP8_INVENTORY']).read_bytes()
    rows = sorted(json.loads(raw), key=lambda r: (r['M'], r['model'], r['op']))
    wanted_m = {int(x) for x in os.environ.get('TP8_MATRIX_M', '8192').split(',')}
    wanted_op = os.environ.get('TP8_MATRIX_OP', 'ag')
    rows = [r for r in rows if r['M'] in wanted_m and ('ag' if r['op'].startswith('ag') else 'rs') == wanted_op]
    model_filter = os.environ.get('TP8_MATRIX_MODEL')
    if model_filter:
        rows = [r for r in rows if r['model'] == model_filter]
    exact_op = os.environ.get('TP8_EXACT_OP')
    if exact_op:
        rows = [r for r in rows if r['op'] == exact_op]
    path = Path(os.environ['TP8_MATRIX_OUT'])
    done = {tuple(json.loads(line)['identity']) for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
    for row in rows:
        identity = (row['model'], row['op'], row['M'], row['N'], row['K'])
        if identity in done:
            continue
        m, n, k = row['M'], row['N'], row['K']
        trace(f'context {identity}')
        op = wanted_op
        ctx = (create_ag_gemm_context(m, n * 8, k, torch.bfloat16, rank, 8, num_local_ranks=8)
               if op == 'ag' else create_gemm_rs_context(m, n, rank, 8, 8, torch.bfloat16, torch.cuda.Stream()))
        trace('context ready')
        gen = torch.Generator(device='cuda').manual_seed(7300 + rank)
        x = torch.randn((m // 8 if op == 'ag' else m, k), device='cuda', dtype=torch.bfloat16, generator=gen).mul_(.05)
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16, generator=gen).mul_(.05)
        full = torch.empty((m, k if op == 'ag' else n), device='cuda', dtype=torch.bfloat16)
        out = torch.empty((m if op == 'ag' else m // 8, n), device='cuda', dtype=torch.bfloat16)

        def stock():
            if op == 'ag':
                dist.all_gather_into_tensor(full, x, group=group)
                torch.matmul(full, w.T, out=out)
            else:
                torch.matmul(x, w.T, out=full)
                dist.reduce_scatter_tensor(out, full, group=group)
            return out

        def overlap():
            return (ag_gemm(x, w.T, ctx=ctx, autotune=False) if op == 'ag'
                    else gemm_rs(x, w.T, ctx=ctx, autotune=False))

        for _ in range(2):
            trace('stock check')
            expected = stock().clone()
            trace('overlap check')
            actual = overlap()
            trace('compare')
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.02)
        trace('check complete')
        measures = {}
        order = ['stock', 'triton_dist']
        random.Random(str(identity)).shuffle(order)
        for arm in order:
            fn = stock if arm == 'stock' else overlap
            for _ in range(4): fn()
            torch.cuda.synchronize(); dist.barrier(group=group)
            samples = []
            for _ in range(8):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(16): fn()
                end.record(); end.synchronize()
                value = torch.tensor([begin.elapsed_time(end) / 16], device='cuda')
                dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
                samples.append(value.item())
            measures[arm] = {'samples_ms': samples, 'median_ms': statistics.median(samples)}
        if rank == 0:
            record = {'identity': identity, 'model': row['model'], 'op': row['op'], 'M': m, 'N': n, 'K': k,
                      'tp': 8, 'dtype': 'bfloat16', 'batch': 16, 'warmup': 4, 'repeats': 8, 'checks': 2,
                      'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'triton': triton.__version__,
                      'triton_dist_path': triton_dist.__file__, 'cuda_build': torch.version.cuda,
                      'cuda_device_max_connections': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset'),
                      'inventory_sha256': hashlib.sha256(raw).hexdigest(), 'measurements': measures, 'order': order,
                      'speedup': measures['stock']['median_ms'] / measures['triton_dist']['median_ms'],
                      'warning': 'same process both arms; default config without autotune; no E2E'}
            with path.open('a') as f: f.write(json.dumps(record) + '\n'); f.flush()
            print(json.dumps({'identity': identity, 'speedup': record['speedup']}), flush=True)
        dist.barrier(group=group)
        ctx.finalize()
        del ctx, x, w, full, out, expected, actual
        torch.cuda.empty_cache()
    finalize_distributed()


if __name__ == '__main__':
    main()
