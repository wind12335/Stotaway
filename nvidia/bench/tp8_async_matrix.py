"""TP8 serial stock vs direct Async-TP ops, using original measured MNK inventory."""
import datetime
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=240))
    if dist.get_world_size() != 8 or symm._is_test_mode:
        raise ValueError('requires TP8 and real symmetric-memory path')
    symm.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    inventory = Path(os.environ['TP8_INVENTORY'])
    raw = inventory.read_bytes()
    rows = json.loads(raw)
    rows = sorted(rows, key=lambda r: (r['M'], r['model'], r['op']))
    wanted_m = {int(x) for x in os.environ.get('TP8_MATRIX_M', '8192,16384,32768').split(',')}
    rows = [r for r in rows if r['M'] in wanted_m]
    path = Path(os.environ['TP8_MATRIX_OUT'])
    if path.exists():
        done = {tuple(json.loads(line)['identity']) for line in path.read_text().splitlines() if line.strip()}
    else:
        done = set()
    dist.barrier()
    for row in rows:
        identity = (row['model'], row['op'], row['M'], row['N'], row['K'])
        if identity in done:
            continue
        m, n, k = row['M'], row['N'], row['K']
        op = 'ag' if row['op'].startswith('ag') else 'rs'
        generator = torch.Generator(device='cuda').manual_seed(7300 + rank)
        x = torch.randn((m // 8 if op == 'ag' else m, k), device='cuda',
                        dtype=torch.bfloat16, generator=generator).mul_(.05)
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16, generator=generator).mul_(.05)
        full = torch.empty((m, k if op == 'ag' else n), device='cuda', dtype=torch.bfloat16)
        result = torch.empty((m if op == 'ag' else m // 8, n), device='cuda', dtype=torch.bfloat16)

        def stock():
            if op == 'ag':
                dist.all_gather_into_tensor(full, x)
                torch.matmul(full, w.T, out=result)
            else:
                torch.matmul(x, w.T, out=full)
                dist.reduce_scatter_tensor(result, full)
            return result

        def fused():
            name = dist.group.WORLD.group_name
            if op == 'ag':
                return torch.ops.symm_mem.fused_all_gather_matmul(x, [w.T], 0, name, return_A=False)[1][0]
            return torch.ops.symm_mem.fused_matmul_reduce_scatter(x, w.T, 'sum', 0, name)

        for _ in range(2):
            expected = stock().clone()
            actual = fused()
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.02)
        measurements = {}
        order = ['stock', 'async_tp']
        random.Random(str(identity)).shuffle(order)
        for arm in order:
            fn = stock if arm == 'stock' else fused
            for _ in range(4):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            samples = []
            for _ in range(8):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(16):
                    fn()
                end.record()
                end.synchronize()
                t = torch.tensor([start.elapsed_time(end) / 16], device='cuda')
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                samples.append(t.item())
            measurements[arm] = {'samples_ms': samples, 'median_ms': statistics.median(samples)}
        if rank == 0:
            record = {'identity': identity, 'model': row['model'], 'op': row['op'], 'M': m, 'N': n, 'K': k,
                      'tp': 8, 'dtype': 'bfloat16', 'batch': 16, 'warmup': 4, 'repeats': 8, 'checks': 2,
                      'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
                      'triton': triton.__version__, 'cuda_build': torch.version.cuda,
                      'cuda_device_max_connections': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset'),
                      'inventory_sha256': hashlib.sha256(raw).hexdigest(), 'measurements': measurements,
                      'speedup': measurements['stock']['median_ms'] / measurements['async_tp']['median_ms'],
                      'order': order, 'warning': 'direct symm_mem operators; stock and fused share process; no E2E'}
            with path.open('a') as f:
                f.write(json.dumps(record) + '\n')
                f.flush()
            print(json.dumps({'identity': identity, 'speedup': record['speedup'],
                              'stock_ms': measurements['stock']['median_ms'],
                              'async_ms': measurements['async_tp']['median_ms']}), flush=True)
        dist.barrier()
        del x, w, full, result, expected, actual
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
