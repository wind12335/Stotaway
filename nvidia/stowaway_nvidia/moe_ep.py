"""Experimental EP2 top-k MoE autograd built on fixed-capacity exchanges."""
import torch
import torch.nn.functional as F
import os
from torch.autograd.function import once_differentiable

from .moe_grouped import expert_linear


def _expert_linear_reference(x, w, offsets):
    """Slow diagnostic: reproduce per-expert PyTorch GEMM shapes on the host."""
    ends = offsets.detach().cpu().tolist()
    start, parts = 0, []
    for expert, end in enumerate(ends):
        end = int(end)
        parts.append(x[start:end] @ w[expert])
        start = end
    if start < x.shape[0]:
        parts.append(x[start:] @ w[0] * 0)
    return torch.cat(parts)


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ids, exchange):
        rows, offsets, total, order = exchange.dispatch(x.detach(), ids)
        ctx.exchange = exchange
        ctx.save_for_backward(total, order)
        ctx.mark_non_differentiable(offsets, total, order)
        return rows, offsets, total, order

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_rows, *_):
        total, order = ctx.saved_tensors
        ex = ctx.exchange
        ones = torch.ones((ex.tokens, ex.topk), device=grad_rows.device, dtype=torch.float32)
        grad_x = ex.combine(grad_rows.contiguous(), total, order, ones)
        ex.training_pending = False
        return grad_x, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, ids, weights, total, order, exchange):
        out, received = exchange.combine(rows.detach(), total, order, weights.detach(), keep_rows=True)
        ctx.exchange = exchange
        ctx.generation = exchange.training_generation
        ctx.save_for_backward(ids, weights, received)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        ids, weights, received = ctx.saved_tensors
        ex = ctx.exchange
        if ctx.generation != ex.training_generation or getattr(ex, "backward_started", False):
            raise RuntimeError("repeated backward through an EP exchange is unsupported")
        ex.backward_started = True
        per_tag = (grad_out.float().unsqueeze(1) * weights.float().unsqueeze(-1)).to(torch.bfloat16)
        grad_weight = (grad_out.float().unsqueeze(1) * received.float()).sum(-1).to(weights.dtype)
        grad_rows, _, _, _ = ex.dispatch(per_tag.reshape(ex.p, ex.hidden).contiguous(), ids, per_tag=True)
        return grad_rows, None, grad_weight, None, None, None


def ep_moe(x, ids, weights, w1, w2, exchange, *, scale_before_fc2=False):
    """EP2 forward/backward; same call order on both ranks is mandatory.

    Both ranks must participate in each backward in the same order and all
    differentiable inputs must require gradients. The context cannot be reused
    before backward completes, so multiple outstanding forwards and recomputation
    are unsupported.
    """
    if not all(t.requires_grad for t in (x, weights, w1, w2)):
        raise ValueError("EP training prototype requires gradients for x, weights and both expert weights")
    if getattr(exchange, "training_pending", False):
        raise RuntimeError("previous EP training call must finish backward before reuse")
    if (w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != exchange.epr or
            w2.shape[0] != exchange.epr or w1.shape[1] != exchange.hidden or
            w2.shape[2] != exchange.hidden or w1.shape[2] != 2 * w2.shape[1] or
            not w1.is_contiguous() or not w2.is_contiguous()):
        raise ValueError("invalid local expert weight shapes")
    exchange.training_pending = True
    exchange.backward_started = False
    exchange.training_generation = getattr(exchange, "training_generation", 0) + 1
    rows, offsets, total, order = _Dispatch.apply(x, ids, exchange)
    linear = _expert_linear_reference if os.environ.get("STOWAWAY_MOE_REFERENCE_GEMM") == "1" else expert_linear
    up = linear(rows.contiguous(), w1, offsets)
    gate, value = up.chunk(2, dim=-1)
    activation = F.silu(gate) * value
    if scale_before_fc2:
        # Diagnostic upstream semantics: router scales the expert activation
        # before FC2. Differentiable gather is deliberately correctness-first.
        import torch.distributed.nn.functional as dist_nn
        all_weights = torch.cat(dist_nn.all_gather(weights.contiguous()), dim=0).reshape(-1)
        sorted_weights = all_weights.index_select(0, order)
        activation = (activation * sorted_weights.unsqueeze(-1)).to(activation.dtype)
        result = linear(activation.contiguous(), w2, offsets)
        return _Combine.apply(result, ids, torch.ones_like(weights), total, order, exchange)
    result = linear(activation.contiguous(), w2, offsets)
    return _Combine.apply(result, ids, weights, total, order, exchange)
