import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable


class RowParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, weight, exchange, backward_exchange):
        ctx.exchange = exchange
        ctx.backward_exchange = backward_exchange
        ctx.save_for_backward(activation, weight)
        return exchange.gemm_reduce_scatter(activation.detach(), weight.detach())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        activation, weight = ctx.saved_tensors
        if ctx.backward_exchange is not None:
            grad_input, grad_weight, _ = ctx.backward_exchange.all_gather_dgrad(
                grad_output.contiguous(), weight.detach(), activation.detach())
            return grad_input, grad_weight, None, None
        full = torch.empty((activation.shape[0], weight.shape[0]),
                           dtype=grad_output.dtype, device=grad_output.device)
        dist.all_gather_into_tensor(full, grad_output.contiguous())
        return full.matmul(weight), full.t().matmul(activation), None, None


class ColumnParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, weight, exchange, reduce_scatter_fn):
        ctx.exchange = exchange
        ctx.reduce_scatter_fn = reduce_scatter_fn
        ctx.save_for_backward(shard, weight)
        return exchange.all_gather_gemm(shard.detach(), weight.detach())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        shard, weight = ctx.saved_tensors
        full = torch.empty((grad_output.shape[0], shard.shape[1]),
                           dtype=shard.dtype, device=shard.device)
        gather = dist.all_gather_into_tensor(full, shard.contiguous(), async_op=True)
        grad_input_full = grad_output.matmul(weight)
        gather.wait()
        grad_input = torch.empty_like(shard)
        reduce_fn = ctx.reduce_scatter_fn or dist.reduce_scatter_tensor
        reduce = reduce_fn(grad_input, grad_input_full.contiguous(), async_op=True)
        grad_weight = grad_output.t().matmul(full)
        reduce.wait()
        return grad_input, grad_weight, None, None


def row_parallel(activation, weight, exchange, backward_exchange=None):
    return RowParallel.apply(activation, weight, exchange, backward_exchange)


def column_parallel(shard, weight, exchange, reduce_scatter_fn=None):
    return ColumnParallel.apply(shard, weight, exchange, reduce_scatter_fn)
