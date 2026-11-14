"""TP8 external overlap adapters with explicit per-backend data contracts.

These adapters assume torch distributed and, where needed, NVSHMEM were
initialized once by the caller.  Context creation/freeing is collective;
every rank must create and close the same shapes in the same order.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


class TritonDistExchange:
    def __init__(self, rank, m, n, k, dtype, op):
        from triton_dist.kernels.nvidia import create_ag_gemm_context
        from triton_dist.kernels.nvidia.gemm_reduce_scatter import create_gemm_rs_context
        self.op = op
        self.ctx = (create_ag_gemm_context(m, n * 8, k, dtype, rank, 8, num_local_ranks=8)
                    if op == 'ag' else create_gemm_rs_context(m, n, rank, 8, 8, dtype, torch.cuda.Stream()))

    def all_gather_gemm(self, x, weight):
        from triton_dist.kernels.nvidia import ag_gemm
        if self.op != 'ag':
            raise ValueError('AG invoked on RS context')
        return ag_gemm(x, weight.T, ctx=self.ctx, autotune=False)

    def gemm_reduce_scatter(self, x, weight):
        from triton_dist.kernels.nvidia import gemm_reduce_scatter as td_rs
        from triton_dist import nv_utils
        if self.op != 'rs':
            raise ValueError('RS invoked on AG context')
        if not getattr(td_rs, '_stowaway_speed_fallback', False):
            original = td_rs.get_intranode_max_speed_gbps
            def speed_with_nvsmi_fallback(*args, **kwargs):
                try:
                    speed = original(*args, **kwargs)
                    if speed is not None and speed > 0:
                        return speed
                except (TypeError, ValueError):
                    pass
                gpu_index = args[0] if args else kwargs.get('gpu_index', 0)
                physical = nv_utils._get_pynvml_device_id(gpu_index)
                speed = nv_utils._get_nvlink_max_speed_gbps_nvsmi(physical)
                if speed <= 0:
                    raise RuntimeError('cannot obtain NVLink bandwidth from nvidia-smi')
                return speed * (0.8 if kwargs.get('with_scale', False) else 1.0)
            td_rs.get_intranode_max_speed_gbps = speed_with_nvsmi_fallback
            td_rs._stowaway_speed_fallback = True
        return td_rs.gemm_rs(x, weight.T, ctx=self.ctx, autotune=False)

    def close(self):
        self.ctx.finalize()


def _import_generated(work, name, filename):
    """Upstream generator uses paths relative to its checkout during import."""
    previous = os.getcwd()
    try:
        os.chdir(work)
        if str(work) not in sys.path:
            sys.path.insert(0, str(work))
        tests_path = str(work / 'tests')
        if tests_path not in sys.path:
            sys.path.insert(0, tests_path)
        spec = importlib.util.spec_from_file_location(name, work / 'tests' / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous)


class SyncopateExchange:
    def __init__(self, rank, m, n, k, dtype, op, work):
        from syncopate.communication.code_gen import CommGenerator
        from syncopate.communication.comm_runtime.communication_context import CommContext
        from syncopate.communication.common_descriptors import build_all_gather_plan_1d_swizzle, build_all_to_all_plan
        from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
        self.rank, self.m, self.n, self.k, self.dtype, self.op = rank, m, n, k, dtype, op
        # The generated AG GEMM uses tensor descriptors and full tiles. Qwen
        # has a 2060-wide local projection; pad to a complete output tile.
        self.compute_n = ((n + 127) // 128) * 128 if op == 'ag' else n
        work = Path(work)
        if op == 'ag':
            self.module = _import_generated(work, 'stowaway_sync_ag', 'test_ag_gemm_gen.py')
            plans = {r: build_all_gather_plan_1d_swizzle(shape=(m, k), dtype=dtype, axis=0, mesh_size=8,
                     rank=r, buffer_name='a', transfer_kind='pull') for r in range(8)}
            buffer = 'a'
        else:
            self.module = _import_generated(work, 'stowaway_sync_rs', 'test_gemm_rs_transform.py')
            previous = os.getcwd()
            try:
                os.chdir(work)
                self.producer = self.module._load_transformed_gemm_producer()
            finally:
                os.chdir(previous)
            plans = {r: build_all_to_all_plan(shape=(m, n), dtype=dtype, mesh_size=8, rank=r,
                     src_buffer='src', dst_buffer='dst', transfer_kind='push', compute_producer=True)
                     for r in range(8)}
            buffer = 'src'
        gen = CommGenerator(plans)
        gen.plan_signals()
        schedule = lower_comm_plan_to_raw_schedules(gen)[rank][buffer]
        offsets, shapes = schedule.gen_block_offset_lists(), schedule.gen_block_shape_lists()
        if op == 'ag':
            for offset, shape in zip(offsets, shapes):
                offset[1], shape[1] = 0, self.compute_n
        self.offsets = torch.tensor(offsets, device='cuda', dtype=torch.int32)
        self.shapes = torch.tensor(shapes, device='cuda', dtype=torch.int32)
        self.signals = torch.tensor(schedule.gen_signal_lists(), device='cuda', dtype=torch.int32)
        self.cum_tiles = sum(int(shape[0]) * int(shape[1]) for shape in shapes)
        info = gen.generate_code_for_plan()
        info.local_world_size = 8
        if op == 'ag':
            self.module.comm_info = info
        self.ctx = CommContext(rank, info)
        self.output = torch.empty((m if op == 'ag' else m // 8, self.compute_n), device='cuda', dtype=dtype)
        if op == 'rs':
            self.comm_stream = torch.cuda.Stream()
            self.signal_ptr = self.ctx.compute_signal_bufs[rank]
            self.counter = torch.zeros(self.signal_ptr.numel(), device='cuda', dtype=torch.int32)

    def all_gather_gemm(self, x, weight):
        buffer = self.ctx.comm_buffers['a'][self.rank]
        buffer[self.rank * (self.m // 8):(self.rank + 1) * (self.m // 8)].copy_(x)
        from triton_dist.utils import nvshmem_barrier_all_on_stream
        self.ctx.reset_signals()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        self.ctx.start_after(torch.cuda.current_stream())
        self.ctx.execute()
        padded_weight = (torch.nn.functional.pad(weight, (0, 0, 0, self.compute_n - self.n))
                         if self.compute_n != self.n else weight)
        self.module.generated_gemm_consumer(buffer, padded_weight, self.output, 8,
            torch.cuda.get_device_properties(self.rank).multi_processor_count,
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, STAGES=3,
            cur_wave_sizes=self.shapes, wave_offsets=self.offsets, signal_offsets=self.signals,
            signal_ptr=self.ctx.recv_signal_bufs[self.rank], cum_tiles=self.cum_tiles)
        self.ctx.end_before(torch.cuda.current_stream())
        return self.output[:, :self.n].contiguous().clone()

    def gemm_reduce_scatter(self, x, weight):
        from triton_dist.utils import nvshmem_barrier_all_on_stream
        from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda
        self.ctx.reset_signals()
        self.counter.zero_()
        reset = torch.cuda.Event()
        reset.record(torch.cuda.current_stream())
        src = self.ctx.comm_buffers['src'][self.rank]
        self.producer(x, weight, src, 8, torch.cuda.get_device_properties(self.rank).multi_processor_count,
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, STAGES=3,
            cur_wave_sizes=self.shapes, wave_offsets=self.offsets, signal_offsets=self.signals,
            signal_ptr=self.signal_ptr, counter_ptr=self.counter, cum_tiles=self.cum_tiles)
        self.comm_stream.wait_event(reset)
        nvshmem_barrier_all_on_stream(self.comm_stream)
        self.ctx.start_after(self.comm_stream)
        self.ctx.execute(neglect_local=True)
        self.ctx.end_before(self.comm_stream)
        for offset in range(self.ctx.recv_signal_bufs[self.rank].shape[0]):
            _wait_eq_cuda(self.ctx.recv_signal_bufs[self.rank][offset], 1, torch.cuda.current_stream())
        self.module.ring_reduce_tma(self.ctx.comm_buffers['dst'][self.rank], self.output, src, self.rank)
        return self.output.clone()

    def close(self):
        torch.cuda.synchronize()
        dist.barrier()
        ctx, self.ctx = self.ctx, None
        del ctx
        torch.cuda.synchronize()
        dist.barrier()


class FlashOverlapExchange:
    """RS, restoring FlashOverlap tile and cyclic-rank layouts."""
    def __init__(self, rank, m, n, k, config_path, library_path, uid, dtype=torch.float16):
        self.rank, self.m, self.n, self.k = rank, m, n, k
        config = json.loads(Path(config_path).read_text())
        self.bm, self.bn = config['BM'], config['BN']
        bm, bn = self.bm, self.bn
        if m % (8 * bm) or m % 64 or n % bn or not isinstance(config['hint'], list):
            raise ValueError('FlashOverlap layout requires aligned tiles')
        torch.ops.load_library(str(library_path))
        self.impl = torch.classes.flashoverlap_class.OverlapImpl()
        self.impl.nccl_init(rank, 8, uid)
        self.impl.cutlass_init()
        self.impl.overlap_init()
        self.algo = config['Algo']
        tm, tn = m // bm, n // bn
        hint = config['hint']
        if sorted(hint) != list(range(tm * tn)):
            raise ValueError('FlashOverlap hint must be a tile permutation')
        inverse = [0] * len(hint)
        for index, value in enumerate(hint):
            inverse[value] = index
        self.reorder = torch.tensor(inverse, dtype=torch.int32, device='cuda').reshape(tm, tn)
        seg = config['cSeg']
        if sum(seg) != tm * tn:
            raise ValueError('FlashOverlap invalid segment sizes')
        self.counter = torch.zeros((1, tn), dtype=torch.int32, device='cuda')
        self.seg_cpu = torch.tensor(seg, dtype=torch.int32)
        self.seg_gpu = self.seg_cpu.cuda(rank)
        original = torch.arange(m * n // bn, dtype=torch.int32, device='cuda')
        remap = torch.empty_like(original)
        offset = 0
        for count in seg:
            chunk = original[offset:offset + count * bm]
            order = torch.argsort(chunk % 8, stable=True)
            remap[offset:offset + count * bm] = chunk[order]
            offset += count * bm
        rowmap = torch.empty_like(original)
        rowmap[remap.long()] = torch.arange(len(remap), dtype=torch.int32, device='cuda')
        self.rowmap = rowmap
        self.tmp = torch.empty((m, n), dtype=dtype, device='cuda')
        self.raw = torch.empty((m // 8, n), dtype=dtype, device='cuda')
        self.received = torch.empty_like(self.raw)
        # Upstream reorder_rmsnorm uses this exact tile permutation. Restored
        # rows are cyclic by rank; all-to-all converts to contiguous SP shards.
        self.tile_index = self.reorder.reshape(-1).long()

    def gemm_reduce_scatter(self, x, weight):
        self.impl.gemm_reducescatter_overlap(x, weight, self.tmp, self.raw, self.counter,
            self.reorder, self.rowmap, 1, self.seg_cpu, self.seg_gpu, self.algo, False)
        rows, n = self.m // 8, self.n
        bm, bn = self.bm // 8, self.bn
        tiles = self.raw.reshape(-1, bm, bn)[self.tile_index]
        cyclic = tiles.reshape(rows // bm, n // bn, bm, bn).permute(0, 2, 1, 3).contiguous().reshape(rows, n)
        dist.all_to_all_single(self.received, cyclic.contiguous())
        return self.received.reshape(8, rows // 8, n).transpose(0, 1).contiguous().reshape(rows, n)

    def close(self):
        torch.cuda.synchronize()
        dist.barrier()
        self.impl = None
