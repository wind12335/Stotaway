"""Measure full versus DX-style 8-slice torch GEMM on one GPU, no communication.

This isolates one cost of the existing TP8 DX prototype; it does not measure
overlap, network bandwidth or the end-to-end performance of any backend.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import statistics

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['qwen3-8b', 'qwen3-32b', 'llama3-70b'])
    parser.add_argument('--rows', type=int, default=8192)
    parser.add_argument('--all-cells', action='store_true', help='measure every cell in the input profile')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    if min(args.rows, args.batch, args.repeats, args.warmup) <= 0 or args.rows % 8:
        parser.error('positive counts required; rows must be divisible by eight')
    torch.cuda.set_device(0)
    selected = [c for c in json.loads(args.cells.read_text())
                if args.all_cells or (c['model'] in args.models and c['M'] == args.rows)]
    if not selected:
        raise ValueError('no selected shapes')
    torch.manual_seed(1783)
    result = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(args.output.suffix + '.jsonl')
    progress = progress_path.open('x')
    for c in selected:
        m, k, n = c['M'], c['K'], c['N']
        # Match the amplitude of the existing TP8 operator checks.
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16).mul_(0.05)
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16).mul_(0.05)
        out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        block = m // 8
        def full():
            torch.matmul(a, w.t(), out=out)
        def sliced():
            for rank in range(8):
                s = slice(rank * block, (rank + 1) * block)
                torch.matmul(a[s], w.t(), out=out[s])
        reference = torch.matmul(a, w.t())
        sliced()
        torch.testing.assert_close(out, reference, atol=0.02, rtol=0.03)
        del reference
        for _ in range(args.warmup):
            full(); sliced()
        torch.cuda.synchronize()
        samples = {'full': [], 'sliced': []}
        methods = {'full': full, 'sliced': sliced}
        rng = random.Random(f"{c['model']}:{c['op']}")
        for _ in range(args.repeats):
            order = ['full', 'sliced']
            rng.shuffle(order)
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.batch):
                    methods[name]()
                end.record(); end.synchronize()
                samples[name].append(start.elapsed_time(end) / args.batch)
        full_ms = statistics.median(samples['full'])
        sliced_ms = statistics.median(samples['sliced'])
        item = {key: c[key] for key in ('model', 'op', 'M', 'K', 'N')}
        item.update(samples_ms=samples, full_median_ms=full_ms, sliced_median_ms=sliced_ms,
                    sliced_over_full=sliced_ms / full_ms,
                    dx8_operator_speedup_b2b=c['stock_b2b'] / c['dx_b2b'])
        result.append(item)
        progress.write(json.dumps(item) + '\n')
        progress.flush()
        print(json.dumps({k: v for k, v in item.items() if k != 'samples_ms'}), flush=True)
        del a, w, out
        torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(0)
    report = {'metadata': {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           'gpu': props.name, 'compute_capability': list(torch.cuda.get_device_capability(0)),
                           'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                           'batch': args.batch, 'repeats': args.repeats, 'warmup': args.warmup,
                           'scope': 'single GPU, full vs 8-slice GEMM only, no communication',
                           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}, 'results': result}
    progress.close()
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
