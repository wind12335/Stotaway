"""Async-TP as external backend B: PyTorch's own fused symm-mem ops (external overlap, baseline).

Port of nvidia/stowaway_nvidia/async_tp_exchange.py. Source: PyTorch (BSD-3-Clause), the
torch.distributed._symmetric_memory kernels of the DTK torch build (2.9.0 here) -- nothing of this
repository runs on the data path. Forward only: the backward is the framework's own (autograd.py).

Availability (see results/bw1000_oprs_<date>/status.md): the symm-mem ops import and run at world=1
on this stack; the older note "symm_mem fails to initialise on DCU" is superseded for that scope.
"""
import torch

ASYNC_TP = 'async_tp'


class AsyncTpExchange:
    kind = ASYNC_TP

    def __init__(self, group, op):
        if op not in ('ag', 'rs'):
            raise ValueError(f'async_tp has no op {op!r}')
        import torch.distributed._symmetric_memory as symm
        if symm._is_test_mode:
            raise RuntimeError('torch symmetric memory is in test mode (no real backend)')
        self.group, self.op = group, op
        symm.enable_symm_mem_for_group(group.group_name)

    def all_gather_gemm(self, shard, weight):
        """shard [M/R, K] x weight [N, K] -> [M, N] (weight passed as [K, N] to the fused op)."""
        return torch.ops.symm_mem.fused_all_gather_matmul(
            shard.contiguous(), [weight.detach().t()], 0, self.group.group_name, return_A=False)[1][0]

    def gemm_reduce_scatter(self, activation, weight):
        """activation [M, K/R] x weight [N, K/R] -> [M/R, N]."""
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            activation.contiguous(), weight.detach().t(), 'sum', 0, self.group.group_name)

    backward_kind = 'stock'
    all_gather_dgrad = None

    def close(self):
        pass
