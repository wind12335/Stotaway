"""TP8 Syncopate generated GEMM+RS vs DX scratch and NCCL+GEMM."""
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
    world = int(os.environ.get('WORLD_SIZE', '8'))
    m, n, k = (design['target'][x] for x in ('M', 'N', 'K'))
    if arm not in ('stock', 'syncopate', 'dx'):
        raise ValueError('invalid arm')
    if arm == 'syncopate':
        from triton_dist.utils import initialize_distributed, finalize_distributed, nvshmem_barrier_all_on_stream
        from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda
        from syncopate.communication.code_gen import CommGenerator
        from syncopate.communication.comm_runtime.communication_context import CommContext
        from syncopate.communication.common_descriptors import build_all_to_all_plan
        from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
        group = initialize_distributed()
        work = Path(os.environ['SYNC_WORK'])
        os.chdir(work)
        spec = importlib.util.spec_from_file_location('syncopate_rs_upstream', work / 'tests/test_gemm_rs_transform.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        producer = module._load_transformed_gemm_producer()
        plans = {r: build_all_to_all_plan(shape=(m, n), dtype=torch.float16, mesh_size=world, rank=r,
                    src_buffer='src', dst_buffer='dst', transfer_kind='push', compute_producer=True)
                 for r in range(world)}
        generator = CommGenerator(plans)
        generator.plan_signals()
        schedule = lower_comm_plan_to_raw_schedules(generator)[rank]['src']
        shapes = schedule.gen_block_shape_lists()
        block_shapes = torch.tensor(shapes, device='cuda', dtype=torch.int32)
        block_offsets = torch.tensor(schedule.gen_block_offset_lists(), device='cuda', dtype=torch.int32)
        signal_offsets = torch.tensor(schedule.gen_signal_lists(), device='cuda', dtype=torch.int32)
        cum_tiles = sum(int(shape[0]) * int(shape[1]) for shape in shapes)
        info = generator.generate_code_for_plan()
        info.local_world_size = world
        ctx = CommContext(rank, info)
        comm_stream = torch.cuda.Stream()
        signal_ptr = ctx.compute_signal_bufs[rank]
        counter_ptr = torch.zeros(signal_ptr.numel(), device='cuda', dtype=torch.int32)
    else:
        dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
        group = dist.group.WORLD
        ctx = None
        if arm == 'dx':
            from stowaway_nvidia.dx8 import DirectExchange8
            from stowaway_nvidia.runtime import initialize, finalize
            ctx = DirectExchange8(initialize(expected_world_size=world), rank, world, m, k, n, torch.float16)
    if dist.get_world_size() != world:
        raise ValueError('TP8 required')
    gen = torch.Generator(device='cuda').manual_seed(8920 + rank)
    a = torch.randn((m, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    b = torch.randn((n, k), device='cuda', dtype=torch.float16, generator=gen) * .05
    partial = torch.empty((m, n), device='cuda', dtype=torch.float16)
    output = torch.empty((m // world, n), device='cuda', dtype=torch.float16)

    def stock():
        torch.matmul(a, b.T, out=partial)
        dist.reduce_scatter_tensor(output, partial, group=group)
        return output

    def run():
        if arm == 'stock':
            return stock()
        if arm == 'dx':
            return ctx.gemm_reduce_scatter(a, b)
        ctx.reset_signals()
        counter_ptr.zero_()
        reset = torch.cuda.Event()
        reset.record(torch.cuda.current_stream())
        src = ctx.comm_buffers['src'][rank]
        producer(a, b, src, world, torch.cuda.get_device_properties(rank).multi_processor_count,
                 BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, STAGES=3,
                 cur_wave_sizes=block_shapes, wave_offsets=block_offsets, signal_offsets=signal_offsets,
                 signal_ptr=signal_ptr, counter_ptr=counter_ptr, cum_tiles=cum_tiles)
        comm_stream.wait_event(reset)
        nvshmem_barrier_all_on_stream(comm_stream)
        ctx.start_after(comm_stream)
        ctx.execute(neglect_local=True)
        ctx.end_before(comm_stream)
        for offset in range(ctx.recv_signal_bufs[rank].shape[0]):
            _wait_eq_cuda(ctx.recv_signal_bufs[rank][offset], 1, torch.cuda.current_stream())
        module.ring_reduce_tma(ctx.comm_buffers['dst'][rank], output, src, rank)
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
            'tp': world, 'dtype': design['dtype'], 'warmup': design['warmup'], 'batch': design['batch'],
            'repeats': design['repeats'], 'checks': 2, 'gpu': torch.cuda.get_device_name(),
            'torch': str(torch.__version__), 'triton': triton.__version__, 'cuda_build': torch.version.cuda,
            'cuda_device_max_connections': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset'),
            'samples_ms': samples, 'median_ms': statistics.median(samples),
            'warning': 'exploratory independent process; Syncopate adapted imports, NVSHMEM linking and 108 SM; DX scratch; no E2E'}
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
