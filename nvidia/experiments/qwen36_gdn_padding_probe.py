"""Single-GPU exploratory BF16 GEMM padding for Qwen GDN N=2060.

The output slice is non-contiguous when padded. No communication, backward,
or model integration is included; this is not an E2E speedup benchmark.
"""
import argparse
import datetime
import json
from pathlib import Path
import random
import statistics

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--repeats', type=int, default=8)
    args = p.parse_args()
    if min(args.batch, args.repeats) <= 0:
        p.error('positive counts required')
    torch.cuda.set_device(0)
    torch.manual_seed(7126)
    k, n = 5120, 2060
    widths = (2060, 2064, 2080, 2112, 2176)
    report = {'classification': 'single GPU forward GEMM only; non-contiguous padded slice',
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'batch': args.batch, 'repeats': args.repeats, 'results': []}
    for m in (8192, 16384):
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16).mul_(0.05)
        weight = torch.randn((n, k), device='cuda', dtype=torch.bfloat16).mul_(0.05)
        padded_weights = {}
        outs = {}
        for width in widths:
            padded_weights[width] = torch.zeros((width, k), device='cuda', dtype=torch.bfloat16)
            padded_weights[width][:n].copy_(weight)
            outs[width] = torch.empty((m, width), device='cuda', dtype=torch.bfloat16)
        reference = torch.matmul(a, weight.t())
        for width in widths:
            torch.matmul(a, padded_weights[width].t(), out=outs[width])
            torch.testing.assert_close(outs[width][:, :n], reference, rtol=0.03, atol=0.02)
            if width != n:
                checked = outs[width][:, :n].contiguous()
                torch.testing.assert_close(checked, reference, rtol=0.03, atol=0.02)
                del checked
        del reference
        for _ in range(4):
            for width in widths:
                torch.matmul(a, padded_weights[width].t(), out=outs[width])
        torch.cuda.synchronize()
        samples = {(width, mode): [] for width in widths for mode in ('gemm', 'gemm_contiguous')}
        contiguous = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        rng = random.Random(m)
        for _ in range(args.repeats):
            order = list(samples)
            rng.shuffle(order)
            for width, mode in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.batch):
                    torch.matmul(a, padded_weights[width].t(), out=outs[width])
                    if mode == 'gemm_contiguous' and width != n:
                        contiguous.copy_(outs[width][:, :n])
                end.record(); end.synchronize()
                samples[(width, mode)].append(start.elapsed_time(end) / args.batch)
        base = statistics.median(samples[(n, 'gemm')])
        for width in widths:
            for mode in ('gemm', 'gemm_contiguous'):
                median = statistics.median(samples[(width, mode)])
                item = {'M': m, 'K': k, 'N': n, 'padded_N': width, 'mode': mode, 'median_ms': median,
                        'full_vs_padded_speedup': base / median,
                        'output_view_contiguous': outs[width][:, :n].is_contiguous(),
                        'samples_ms': samples[(width, mode)]}
                report['results'].append(item)
                print(json.dumps(item), flush=True)
        del a, weight, padded_weights, outs
    with args.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
