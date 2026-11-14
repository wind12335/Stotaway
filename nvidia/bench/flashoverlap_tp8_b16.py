"""FlashOverlap README-generated RS configuration, TP8 repeated 16 calls.

No third-party source modifications: imports its test helpers and invokes the
same classes. Separate reference correctness test is required for each config.
Run from the isolated FlashOverlap/test directory with this repo in PYTHONPATH.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0, os.getcwd())  # isolated FlashOverlap/test, ahead of stdlib `test`
from test import reorder_indices, generate_row_remap_array


def worker(rank, world, nccl_id, m, n, k, config, batch, repeats, seed, warmup, output):
    torch.cuda.set_device(rank)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR='127.0.0.1', MASTER_PORT='29659')
    dist.init_process_group('nccl')
    baseline = torch.classes.flashoverlap_class.BaselineImpl()
    overlap = torch.classes.flashoverlap_class.OverlapImpl()
    a = torch.randn((m, k), device='cuda', dtype=torch.float16) * .5
    b = torch.randn((n, k), device='cuda', dtype=torch.float16) * .5
    c = torch.empty((m, n), device='cuda', dtype=torch.float16)
    d = torch.empty((m // world, n), device='cuda', dtype=torch.float16)
    bm, bn, algo, seg, hint = (config[x] for x in ('BM', 'BN', 'Algo', 'cSeg', 'hint'))
    if len(seg) <= 1:
        raise ValueError('selected config does not overlap')
    tiles = (m // bm) * (n // bn)
    counter = torch.zeros((n // bn,), dtype=torch.int32, device='cuda')
    order = reorder_indices(tiles, hint).reshape((m // bm, n // bn))
    row = generate_row_remap_array(m, n, bm, bn, seg, world)
    segments_cpu = torch.tensor(seg, dtype=torch.int32)
    segments_gpu = segments_cpu.cuda(rank)
    def stock():
        baseline.gemm_reducescatter(a, b, c, d)
    def fused():
        overlap.gemm_reducescatter_overlap(a, b, c, d, counter, order, row, 1,
                                           segments_cpu, segments_gpu, algo, False)
    samples = {}
    # A process owns exactly one of the two native NCCL communicators. Reusing
    # one native NCCL id for baseline and overlap classes in the same process
    # can hang; the upstream benchmark likewise spawns each arm separately.
    name = os.environ['FLASH_ARM']
    if name == 'stock':
        baseline.nccl_init(rank, world, nccl_id)
        baseline.cublas_init()
    elif name == 'overlap':
        overlap.nccl_init(rank, world, nccl_id)
        overlap.cutlass_init()
        overlap.overlap_init()
    else:
        raise ValueError('invalid arm')
    order_names = [name]
    for name in order_names:
        fn = stock if name == 'stock' else fused
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        values = []
        for _ in range(repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(batch):
                fn()
            end.record()
            end.synchronize()
            time = torch.tensor([begin.elapsed_time(end) / batch], device='cuda')
            dist.all_reduce(time, op=dist.ReduceOp.MAX)
            values.append(time.item())
        samples[name] = {'samples_ms': values, 'median_ms': statistics.median(values)}
    if rank == 0:
        output['results'] = samples
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=4)
    ap.add_argument('--seed', type=int, default=812)
    args = ap.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    world = torch.cuda.device_count()
    if world != 8 or min(args.m, args.n, args.k, args.batch, args.repeats, args.warmup) <= 0:
        raise ValueError('requires exactly 8 GPUs and positive arguments')
    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
    manager = mp.Manager()
    output = manager.dict()
    arms = ['stock', 'overlap']
    random.Random(args.seed).shuffle(arms)
    measurements = {}
    for arm in arms:
        os.environ['FLASH_ARM'] = arm
        nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
        mp.spawn(worker, args=(world, nccl_id, args.m, args.n, args.k, config,
                               args.batch, args.repeats, args.seed, args.warmup, output), nprocs=world)
        measurements.update(dict(output)['results'])
    data = {'method': 'flashoverlap', 'direction': 'rs', 'M': args.m, 'N': args.n, 'K': args.k,
            'tp': world, 'dtype': 'float16', 'batch': args.batch, 'repeats': args.repeats, 'warmup': args.warmup,
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'config': str(args.config), 'config_sha256': hashlib.sha256(raw).hexdigest(),
            'seed': args.seed, 'correctness': 'must be checked separately using README example',
            'measurements': measurements}
    with args.output.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
