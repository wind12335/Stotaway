"""Diagnostic Megatron distributed-optimizer bucket RS for world=DP=2.

This wraps the bucket group's coalescing manager so its wait fences both NCCL
work and every independent DX completion event.  It does not support mixed TP
and DP groups, multiple optimizer instances, or FP32-accumulating RS.
"""
from contextlib import contextmanager
import threading

import torch
import torch.distributed as dist

from .dx import ReduceScatter


class OptimizerBucketBRS:
    def __init__(self, cuda, min_bytes=0, k_slots=1):
        self.cuda, self.min_bytes, self.k_slots = cuda, min_bytes, k_slots
        self.contexts = {}
        self.streams = {}
        self.local = threading.local()
        self.originals = None
        self.calls = self.fallbacks = self.batches = self.multi_bucket_batches = 0

    def install(self):
        from megatron.core.distributed import param_and_grad_buffer as module
        original_start = module._ParamAndGradBucketGroup.start_grad_sync
        original_coalescing = module._coalescing_manager
        original_reduce = module.dist_reduce_scatter_func
        if self.originals is not None:
            raise RuntimeError("adapter already installed")
        self.originals = module, original_start, original_coalescing, original_reduce
        adapter = self

        def start(bucket_group, force_all_reduce=False):
            config = bucket_group.ddp_config
            group = getattr(bucket_group, "intra_distributed_optimizer_instance_group", None)
            eligible = (config.use_distributed_optimizer and not force_all_reduce
                        and config.num_distributed_optimizer_instances == 1
                        and not config.reduce_scatter_with_fp32_accumulation
                        and group is not None and dist.get_process_group_ranks(group) == [0, 1])
            if getattr(adapter.local, "active", False):
                raise RuntimeError("nested optimizer bucket dispatch is unsupported")
            adapter.local.active = eligible
            adapter.local.group = group if eligible else None
            try:
                return original_start(bucket_group, force_all_reduce=force_all_reduce)
            finally:
                adapter.local.active = False
                adapter.local.group = None

        @contextmanager
        def coalescing(group=None, device=None, async_ops=False):
            if not getattr(adapter.local, "active", False):
                with original_coalescing(group, device, async_ops) as cm:
                    yield cm
                return
            if getattr(adapter.local, "handles", None) is not None:
                raise RuntimeError("nested coalescing is unsupported")
            handles = []
            adapter.local.handles = handles
            try:
                with original_coalescing(group, device, async_ops) as cm:
                    # DX handles must be attached before NCCL manager exits;
                    # cm.wait() then fences custom events and any NCCL work.
                    yield cm
                    for handle in handles:
                        cm.append(handle)
                    # PyTorch 2.7 manager references an unset local `work` when
                    # no NCCL op was recorded.  A one-element all-reduce gives
                    # the manager valid work without changing bucket contents.
                    from torch.distributed.distributed_c10d import _world
                    coalesced_group = group if group is not None else dist.group.WORLD
                    if not _world.pg_coalesce_state.get(coalesced_group):
                        marker = torch.zeros(1, device="cuda", dtype=torch.int32)
                        dist.all_reduce(marker, group=group, async_op=async_ops)
                # PyTorch's synchronous coalescing manager waits only for its
                # NCCL work; it does not wait for the Work objects in cm.
                # Fence every independently launched DX/custom operation before
                # the caller can consume its output on the current CUDA stream.
                if not async_ops:
                    for handle in handles:
                        handle.wait()
                if handles:
                    adapter.batches += 1
                    adapter.multi_bucket_batches += int(len(handles) > 1)
            finally:
                adapter.local.handles = None

        def reduce(output, input, op=None, group=None, async_op=False):
            handles = getattr(adapter.local, "handles", None)
            eligible = (getattr(adapter.local, "active", False) and handles is not None
                        and group is getattr(adapter.local, "group", None)
                        and input.dtype in (torch.float16, torch.bfloat16)
                        and input.is_cuda and output.device == input.device and output.dtype == input.dtype
                        and input.is_contiguous() and output.is_contiguous()
                        and not input.requires_grad and not output.requires_grad
                        and input.numel() > 0 and input.numel() % 2 == 0
                        and output.numel() * 2 == input.numel()
                        and input.numel() * input.element_size() >= adapter.min_bytes
                        and (op is None or op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG)))
            if not eligible:
                adapter.fallbacks += 1
                kwargs = {"group": group, "async_op": async_op}
                if op is not None:
                    kwargs["op"] = op
                result = original_reduce(output, input, **kwargs)
                if handles is not None and result is not None:
                    # The fast NCCL coalescing path returns an illegal-to-wait
                    # placeholder.  Other collective implementations can return
                    # real Work, which must remain attached to this bucket group.
                    from torch.distributed.distributed_c10d import _IllegalWork
                    if not isinstance(result, _IllegalWork):
                        handles.append(result)
                return result
            key = (input.numel(), input.dtype)
            if key not in adapter.contexts:
                adapter.contexts[key] = ReduceScatter(adapter.cuda, dist.get_rank(), 2,
                                                     input.numel(), input.dtype, adapter.k_slots)
                adapter.streams[key] = torch.cuda.Stream(device=input.device)
            scale = 0.5 if op == dist.ReduceOp.AVG else 1.0
            caller = torch.cuda.current_stream(input.device)
            worker = adapter.streams[key]
            worker.wait_stream(caller)
            with torch.cuda.stream(worker):
                _, handle = adapter.contexts[key](input, output, defer_fence=True, scale=scale)
            handles.append(handle)
            adapter.calls += 1
            return handle if async_op else None

        module._ParamAndGradBucketGroup.start_grad_sync = start
        module._coalescing_manager = coalescing
        module.dist_reduce_scatter_func = reduce

    def close(self):
        if self.originals:
            module, start, coalescing, reduce = self.originals
            module._ParamAndGradBucketGroup.start_grad_sync = start
            module._coalescing_manager = coalescing
            module.dist_reduce_scatter_func = reduce
            self.originals = None
        for context in self.contexts.values():
            context.close()
        self.contexts.clear()
        self.streams.clear()
