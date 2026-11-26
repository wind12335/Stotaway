"""Autograd wrappers of one overlapped call site (training): port of nvidia/stowaway_nvidia/autograd.py.

Both wrappers take an exchange object from the dispatcher (backends/): the forward is the overlap
method's fused op, the backward keeps the serial-path arithmetic of the framework:

  RowParallel     forward  exchange.gemm_reduce_scatter: [M, K/R] x [N, K/R] -> [M/R, N]
                  backward exchange.all_gather_dgrad when the method has one (DEO: fused AG of the
                          grad_out shards + per-block dgrad, dx_ag_bwd_op); otherwise the stock form:
                          AllGather of grad_out, dgrad = full @ W, wgrad = grad_out^T @ activation

  ColumnParallel  forward  exchange.all_gather_gemm: [M/R, K] x [N, K] -> [M, N]
                  backward stock form, the same order as Megatron's own (layers.py): async AllGather
                          of the input shard, dgrad = grad_out @ W, async ReduceScatter of dgrad
                          (the dgrad RS is in flight under the weight gradient), wgrad

The collectives use the group passed in (never the default group: the default group may span more
ranks than the TP group). Which backward a hit used is part of the hit log: exchange.backward_kind.
"""
import torch


class RowParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, weight, exchange, group):
        ctx.group, ctx.dgrad = group, getattr(exchange, 'all_gather_dgrad', None)
        out = exchange.gemm_reduce_scatter(activation.detach(), weight.detach())
        ctx.save_for_backward(activation.detach(), weight.detach())
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        activation, weight = ctx.saved_tensors
        if ctx.dgrad is not None:
            grad_input, grad_weight, _ = ctx.dgrad(grad_output.contiguous(), weight, activation)
            return grad_input, grad_weight, None, None
        full = torch.empty((activation.shape[0], weight.shape[0]), dtype=grad_output.dtype,
                           device=grad_output.device)
        torch.distributed.all_gather_into_tensor(full, grad_output.contiguous(), group=ctx.group)
        grad_input = torch.matmul(full, weight)
        grad_weight = full.t().matmul(activation)          # stock form: no materialised [N, M] copy
        return grad_input, grad_weight, None, None


class ColumnParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, weight, exchange, reduce_scatter_fn, group):
        ctx.group, ctx.reduce_scatter_fn = group, reduce_scatter_fn
        out = exchange.all_gather_gemm(shard.detach(), weight.detach())
        ctx.save_for_backward(shard.detach(), weight.detach())
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        shard, weight = ctx.saved_tensors
        group = ctx.group
        full = torch.empty((shard.shape[0] * group.size(), shard.shape[1]), dtype=shard.dtype,
                           device=shard.device)
        handle = torch.distributed.all_gather_into_tensor(full, shard.contiguous(), group=group,
                                                          async_op=True)
        grad_input = torch.matmul(grad_output, weight)
        handle.wait()
        # dgrad RS in flight under the wgrad (stock order); 3-D like the framework's own tensors
        grad_shard = torch.empty_like(shard)
        reduce = (ctx.reduce_scatter_fn(grad_shard, grad_input.contiguous(), group=group, async_op=True)
                  if ctx.reduce_scatter_fn is not None else
                  torch.distributed.reduce_scatter_tensor(grad_shard, grad_input.contiguous(), group=group,
                                                          async_op=True))
        grad_weight = torch.matmul(grad_output.reshape(-1, grad_output.shape[-1]).t(),
                                   full.reshape(-1, full.shape[-1]))
        reduce.wait()
        return grad_shard, grad_weight, None, None, None


def row_parallel(activation, weight, exchange, group):
    """[M, K/R] -> [M/R, N]; training (fused forward, backward as described in the module doc)."""
    return RowParallel.apply(activation, weight, exchange, group)


def column_parallel(shard, weight, exchange, group):
    """[M/R, K] -> [M, N]; training."""
    return ColumnParallel.apply(shard, weight, exchange, None, group)

