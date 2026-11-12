"""Experimental TP-only Megatron reduce-scatter adapter for the two-rank prototype.

Do not patch distributed optimizer buckets: their coalesced work-handle contract
requires a separate fence-all implementation and DP groups can differ from TP.
Every rank must enter eligible calls with identical shapes in identical order.
"""
import torch
import torch.distributed as dist

from .dx import ReduceScatter


class TensorParallelBRS:
    def __init__(self, cuda, min_bytes=0, k_slots=1, world=2):
        self.cuda = cuda
        self.min_bytes = min_bytes
        self.k_slots = k_slots
        if world not in (2, 8): raise ValueError('requires TP2 or TP8')
        self.world = world
        self.contexts = {}
        self.streams = {}
        self.originals = []
        self.hits = {"layers": 0, "mappings": 0}
        self.fallbacks = {"layers": 0, "mappings": 0}

    def install(self):
        from megatron.core.tensor_parallel import layers, mappings
        for name, module in (("layers", layers), ("mappings", mappings)):
            original = module.dist_reduce_scatter_func
            self.originals.append((module, original))

            def wrapper(output, input, op=None, group=None, async_op=False,
                        _name=name, _original=original):
                eligible = (group is not None and dist.get_process_group_ranks(group) == list(range(self.world))
                            and input.dtype in (torch.float16, torch.bfloat16)
                            and input.is_cuda and output.device == input.device
                            and output.dtype == input.dtype and input.is_contiguous()
                            and output.is_contiguous() and not input.requires_grad
                            and not output.requires_grad and input.numel() % self.world == 0
                            and output.numel() * self.world == input.numel()
                            and input.numel() * input.element_size() >= self.min_bytes
                            and (op is None or op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG)))
                if not eligible:
                    self.fallbacks[_name] += 1
                    kwargs = {"group": group, "async_op": async_op}
                    if op is not None:
                        kwargs["op"] = op
                    return _original(output, input, **kwargs)
                key = (input.numel(), input.dtype)
                if key not in self.contexts:
                    if self.world == 8:
                        from .brs8 import ReduceScatter8
                        impl = ReduceScatter8
                    else:
                        impl = ReduceScatter
                    self.contexts[key] = impl(self.cuda, dist.get_rank(), self.world,
                                              input.numel(), input.dtype, self.k_slots)
                    self.streams[key] = torch.cuda.Stream(device=input.device)
                scale = 1.0/self.world if op == dist.ReduceOp.AVG else 1.0
                # A context requires one caller stream, but Megatron can issue
                # equal-sized collectives on different streams. Serialize their
                # dispatch on a dedicated stream, after the current input is ready.
                caller = torch.cuda.current_stream(input.device)
                worker = self.streams[key]
                worker.wait_stream(caller)
                with torch.cuda.stream(worker):
                    _, handle = self.contexts[key](input, output, defer_fence=True, scale=scale)
                if not async_op:
                    handle.wait()
                self.hits[_name] += 1
                return handle if async_op else None

            module.dist_reduce_scatter_func = wrapper

    def close(self):
        for module, original in self.originals:
            module.dist_reduce_scatter_func = original
        for context in self.contexts.values():
            context.close()
        self.contexts.clear()
        self.streams.clear()
