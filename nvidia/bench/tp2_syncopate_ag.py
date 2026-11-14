"""TP2 AG: Syncopate generated consumer vs DX and NCCL+GEMM."""
import datetime
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import triton


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    manifest = Path(os.environ['SYNC_MANIFEST']).read_bytes()
    design = json.loads(manifest)
    arm = os.environ['SYNC_ARM']
    m, n, k = (design['target'][x] for x in ('M', 'N', 'K'))
    if arm not in ('stock', 'syncopate', 'dx'):
        raise ValueError('invalid arm')
    if arm == 'syncopate':
        from triton_dist.utils import initialize_distributed, finalize_distributed
        from syncopate.communication.code_gen import CommGenerator
        from syncopate.communication.comm_runtime.communication_context import CommContext
        from syncopate.communication.common_descriptors import build_all_gather_plan_1d_swizzle
        from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
        group = initialize_distributed()
        work = Path(os.environ['SYNC_WORK'])
        os.chdir(work)
        spec = importlib.util.spec_from_file_location('syncopate_ag_upstream', work / 'tests/test_ag_gemm_gen.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        plans = {r: build_all_gather_plan_1d_swizzle(shape=(m, k), dtype=torch.float16, axis=0,
                     mesh_size=2, rank=r, buffer_name='a', transfer_kind='pull') for r in range(2)}
        generator = CommGenerator(plans)
        generator.plan_signals()
        schedule = lower_comm_plan_to_raw_schedules(generator)[rank]['a']
        offsets = schedule.gen_block_offset_lists()
        sizes = schedule.gen_block_shape_lists()
        for pos, shape in zip(offsets, sizes):
            pos[1] = 0
            shape[1] = n
        block_offsets = torch.tensor(offsets, device='cuda', dtype=torch.int32)
        block_shapes = torch.tensor(sizes, device='cuda', dtype=torch.int32)
        signals = torch.tensor(schedule.gen_signal_lists(), device='cuda', dtype=torch.int32)
        cum_tiles = sum(int(shape[0]) * int(shape[1]) for shape in sizes)
        comm_info = generator.generate_code_for_plan()
        comm_info.local_world_size = 2
        module.comm_info = comm_info
        ctx = CommContext(rank, comm_info)
    else:
        dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
        group = dist.group.WORLD
        ctx = None
        if arm == 'dx':
            from stowaway_nvidia.dx import DirectExchange
            from stowaway_nvidia.runtime import initialize, finalize
            ctx = DirectExchange(initialize(), rank, 2, m, k, n, torch.float16)
    if dist.get_world_size() != 2:
        raise ValueError('TP2 required')
    gen = torch.Generator(device='cuda').manual_seed(8820 + rank)
    a = torch.randn((m // 2, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    b = torch.randn((n, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    gathered = torch.empty((m, k), device='cuda', dtype=torch.float16)
    output = torch.empty((m, n), device='cuda', dtype=torch.float16)
    if arm == 'syncopate':
        buffer = ctx.comm_buffers['a'][rank]
        buffer.zero_()
        buffer[rank * (m // 2):(rank + 1) * (m // 2)].copy_(a)

    def stock():
        dist.all_gather_into_tensor(gathered, a, group=group)
        torch.matmul(gathered, b.T, out=output)
        return output

    def run():
        if arm == 'stock':
            return stock()
        if arm == 'dx':
            return ctx.all_gather_gemm(a, b)
        module.fused_all_gather_gemm(b, output, ctx, block_shapes, block_offsets, signals,
            cum_tiles, torch.cuda.get_device_properties(rank).multi_processor_count, 128, 128, 64, 3)
        return output

    for _ in range(2):
        reference = stock().clone()
        actual = run()
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)
    for _ in range(design['warmup']):
        run()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    samples = []
    for _ in range(design['repeats']):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(design['batch']):
            run()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([start.elapsed_time(end) / design['batch']], device='cuda')
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        samples.append(elapsed.item())
    if rank == 0:
        result = {'arm': arm, 'target': design['target'], 'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
            'tp': 2, 'dtype': design['dtype'], 'warmup': design['warmup'], 'batch': design['batch'],
            'repeats': design['repeats'], 'checks': 2, 'gpu': torch.cuda.get_device_name(),
            'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda_build': torch.version.cuda,
            'cuda_device_max_connections': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset'),
            'samples_ms': samples, 'median_ms': statistics.median(samples),
            'warning': 'exploratory independent process; Syncopate adapted imports and 108 SM; no E2E'}
        with Path(os.environ['SYNC_OUT']).open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps(result), flush=True)
    if arm == 'syncopate':
        del ctx
        torch.cuda.synchronize()
        dist.barrier(group=group)
        finalize_distributed()
    elif arm == 'dx':
        ctx.close()
        finalize()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
