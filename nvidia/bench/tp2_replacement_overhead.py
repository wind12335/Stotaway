"""Measure the TP2 carrier's per-call collective validation overhead."""
import datetime
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from stowaway_nvidia.replacement_plan import ReplacementExecutor


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    exe = ReplacementExecutor(Path(os.environ['REPLACEMENT_PLAN']))
    g = torch.Generator(device='cuda').manual_seed(6120 + rank)
    x = torch.randn((512, 4096), device='cuda', dtype=torch.bfloat16, generator=g) * .05
    w = torch.randn((4096, 4096), device='cuda', dtype=torch.bfloat16, generator=g) * .05

    def direct():
        full = torch.empty((1024, 4096), device='cuda', dtype=torch.bfloat16)
        dist.all_gather_into_tensor(full, x)
        return full @ w.T

    def guarded():
        return exe.run('fc1', 'ag', x, w)

    def batch_guarded():
        return exe.run_batch([('fc1', 'ag', x, w)] * 16)

    torch.testing.assert_close(guarded(), direct(), rtol=.02, atol=.02)
    results = {'direct_stock': [], 'carrier_stock': [], 'batch_carrier_stock': []}
    paths = {'direct_stock': direct, 'carrier_stock': guarded, 'batch_carrier_stock': batch_guarded}
    for order in (('direct_stock', 'carrier_stock', 'batch_carrier_stock'),
                  ('batch_carrier_stock', 'direct_stock', 'carrier_stock'),
                  ('carrier_stock', 'batch_carrier_stock', 'direct_stock')):
        for name in order:
            fn = paths[name]
            for _ in range(4):
                guarded() if name == 'batch_carrier_stock' else fn()
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            if name == 'batch_carrier_stock':
                fn()
            else:
                for _ in range(16):
                    fn()
            torch.cuda.synchronize()
            duration = (time.perf_counter() - start) * 1000 / 16
            gathered = [None, None]
            dist.all_gather_object(gathered, duration, group=exe.group)
            results[name].append(max(gathered))
    if rank == 0:
        payload = {'classification': 'TP2 synthetic stock carrier wall-clock overhead',
                   'plan_sha256': exe.digest, 'batch': 16, 'blocks': 3,
                   'samples_ms': results,
                   'median_ms': {k: statistics.median(v) for k, v in results.items()},
                   'warning': 'same process alternating order; batch carrier performs one Gloo metadata exchange '
                              'per 16 already-materialized calls; fixed inputs; not model E2E or GPU-only time'}
        with Path(os.environ['REPLACEMENT_RESULT']).open('x') as f:
            json.dump(payload, f, indent=2)
            f.write('\n')
        print(json.dumps(payload), flush=True)
    exe.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
