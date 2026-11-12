"""Measure Qwen3.6 text-vehicle forward GEMMs only, on a single A800."""
import argparse
import datetime
import json
import statistics
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--repeats', type=int, default=8)
    args = p.parse_args()
    if min(args.batch, args.repeats) <= 0:
        p.error('batch and repeats must be positive')
    m = json.loads(args.manifest.read_text())
    if m['tp'] != 8 or m['dtype'] != 'bfloat16':
        raise ValueError('expected BF16 TP8 shape manifest')
    torch.cuda.set_device(0)
    torch.manual_seed(37)
    result = []
    cache = {}
    for site in m['sites']:
        rows, k, n = (site[x] for x in ('M', 'K', 'N'))
        key = (rows, k, n)
        if key in cache:
            row = {**cache[key], 'site': site['site'], 'reused_shape_measurement': True}
            result.append(row)
            print(json.dumps(row), flush=True)
            continue
        a = torch.randn((rows, k), device='cuda', dtype=torch.bfloat16).mul_(0.05)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16).mul_(0.05)
        out = torch.empty((rows, n), device='cuda', dtype=torch.bfloat16)
        for _ in range(5):
            torch.matmul(a, b.t(), out=out)
        torch.cuda.synchronize()
        single = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); torch.matmul(a, b.t(), out=out); end.record(); end.synchronize()
            single.append(start.elapsed_time(end))
        samples = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.batch):
                torch.matmul(a, b.t(), out=out)
            end.record(); end.synchronize()
            samples.append(start.elapsed_time(end) / args.batch)
        row = {key: site[key] for key in ('site', 'M', 'K', 'N')}
        row.update(single_samples_ms=single, single_median_ms=statistics.median(single),
                   samples_ms=samples, b2b_median_ms=statistics.median(samples),
                   reused_shape_measurement=False)
        result.append(row)
        cache[key] = row
        print(json.dumps(row), flush=True)
        del a, b, out
    report = {'classification': 'single-gpu-gemm-probe', 'model': m['model'], 'tp': m['tp'],
              'M': m['M'], 'dtype': m['dtype'], 'batch': args.batch, 'repeats': args.repeats,
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'cuda_build': torch.version.cuda, 'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'results': result}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
