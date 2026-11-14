"""Exchange interface for PyTorch's Async-TP symmetric-memory collectives."""
import torch
import torch.distributed as dist

class AsyncTPExchange:
    def __init__(self, group):
        import torch.distributed._symmetric_memory as symm
        if symm._is_test_mode:
            raise RuntimeError('symmetric-memory test mode is active')
        symm.enable_symm_mem_for_group(group.group_name)
        self.group = group
        self.counts = {'ag': 0, 'rs': 0}

    def all_gather_gemm(self, shard, weight):
        self.counts['ag'] += 1
        return torch.ops.symm_mem.fused_all_gather_matmul(
            shard, [weight.T], 0, self.group.group_name, return_A=False)[1][0]

    def gemm_reduce_scatter(self, activation, weight):
        self.counts['rs'] += 1
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            activation, weight.T, 'sum', 0, self.group.group_name)
