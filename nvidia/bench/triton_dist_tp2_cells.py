"""TP2 AG+GEMM measurements for installed Triton-distributed; no autotuning.

Records 16 consecutive invocations per sample, slowest rank, same benchmark
ordering as the DX collector. Each file is one M and is never overwritten.
"""
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton_dist
from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
from triton_dist.utils import finalize_distributed, initialize_distributed

from bench.tp2_cells import configured_models, measure, shapes


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    group = initialize_distributed()
    m = int(os.environ.get('CELL_M', '1024'))
    repeats = int(os.environ.get('CELL_REPEATS', '8'))
    batch = int(os.environ.get('CELL_BATCH', '16'))
    checks = int(os.environ.get('CELL_CHECKS', '2'))
    seed = int(os.environ.get('CELL_SCHEDULE_SEED', '1'))
    dtype = torch.bfloat16
    report = {'metadata': {'classification': 'exploratory TP2, actual Triton-distributed AG kernel',
                           'method': 'triton_dist', 'tp': 2, 'dtype': 'bfloat16', 'rows': m,
                           'checks': checks, 'repeats': repeats, 'batch': batch,
                           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                           'triton': triton.__version__, 'triton_dist_path': triton_dist.__file__,
                           'cuda_build': torch.version.cuda, 'python': sys.executable,
                           'autotune': False, 'schedule_seed': seed,
                           'cuda_device_max_connections': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset')},
              'results': []}
    generator = torch.Generator(device='cuda').manual_seed(917 + rank)
    for model, dims in configured_models().items():
        if os.environ.get('CELL_MODEL') and model != os.environ['CELL_MODEL']:
            continue
        for site, op, rows, k, n in shapes(*dims, m):
            if op != 'ag' or (os.environ.get('CELL_SITE') and site != os.environ['CELL_SITE']):
                continue
            ctx = create_ag_gemm_context(rows, n * 2, k, dtype, rank, 2, num_local_ranks=2)
            shard = torch.randn((rows // 2, k), device='cuda', dtype=dtype, generator=generator) * .05
            weight = torch.randn((n, k), device='cuda', dtype=dtype, generator=generator) * .05
            full = torch.empty((rows, k), device='cuda', dtype=dtype)
            output = torch.empty((rows, n), device='cuda', dtype=dtype)
            for _ in range(checks):
                dist.all_gather_into_tensor(full, shard, group=group)
                reference = full @ weight.T
                actual = ag_gemm(shard, weight.T, ctx=ctx, autotune=False)
                torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)
            def comm():
                dist.all_gather_into_tensor(full, shard, group=group)
            def gemm():
                torch.matmul(full, weight.T, out=output)
            def stock():
                comm()
                gemm()
            def overlap():
                return ag_gemm(shard, weight.T, ctx=ctx, autotune=False)
            functions = [('gemm', gemm), ('comm', comm), ('stock', stock), ('overlap', overlap)]
            random.Random(f'{seed}:{model}:{site}:{rows}').shuffle(functions)
            measurements = {name: measure(fn, repeats, batch) for name, fn in functions}
            report['results'].append({'model': model, 'site': site, 'op': op, 'M': rows, 'K': k, 'N': n,
                                      'payload_MB': rows*k*2/1e6, 'checks': checks,
                                      'actual_path': 'triton_dist_ag_gemm_no_autotune',
                                      'measurements': measurements})
            if rank == 0:
                print(json.dumps({'model': model, 'site': site, 'M': rows, 'K': k, 'N': n,
                                  **{name+'_ms': val['median_ms'] for name, val in measurements.items()}}), flush=True)
            ctx.finalize()
            del ctx, shard, weight, full, output, reference, actual
            torch.cuda.empty_cache()
    if rank == 0:
        with Path(os.environ['CELL_OUT']).open('x') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
    finalize_distributed()


if __name__ == '__main__':
    main()
