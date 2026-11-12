"""Single-GPU top-k expert MLP prototype using device-resident routing."""
import torch
import torch.nn.functional as F

from .moe_grouped import expert_linear
from .moe_routing import expert_offsets


def local_moe(x, expert_ids, route_weights, w1, w2):
    """Route [T,H] tokens through E SwiGLU experts, then combine top-k rows.

    w1 is [E,H,2F], w2 is [E,F,H], and expert_ids/route_weights are [T,K].
    All expert IDs must be in [0,E); their values are trusted on device.
    The function is a local correctness prototype, not distributed EP.
    """
    if (x.ndim != 2 or expert_ids.ndim != 2 or route_weights.shape != expert_ids.shape or
            expert_ids.shape[0] != x.shape[0] or expert_ids.shape[1] == 0 or
            w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0] or
            w1.shape[1] != x.shape[1] or w1.shape[2] != 2 * w2.shape[1] or
            w2.shape[2] != x.shape[1] or expert_ids.dtype not in (torch.int32, torch.int64) or
            route_weights.dtype not in (torch.float32, torch.float16, torch.bfloat16) or
            any(t.device != x.device or not t.is_contiguous() for t in (expert_ids, route_weights, w1, w2)) or
            not x.is_cuda or not x.is_contiguous()):
        raise ValueError("invalid local MoE input shapes, dtypes, devices, or layouts")
    t, h = x.shape
    topk = expert_ids.shape[1]
    ids = expert_ids.reshape(-1)
    order = torch.argsort(ids, stable=True)
    offsets, _ = expert_offsets(ids, w1.shape[0])
    grouped = x.index_select(0, torch.div(order, topk, rounding_mode="floor"))
    up = expert_linear(grouped.contiguous(), w1, offsets)
    gate, value = up.chunk(2, dim=-1)
    down = expert_linear((F.silu(gate) * value).contiguous(), w2, offsets)
    # Sorting the permutation back avoids an extra scatter with in-place autograd.
    restored = down.index_select(0, torch.argsort(order)).view(t, topk, h)
    return (restored.float() * route_weights.float().unsqueeze(-1)).sum(1).to(x.dtype)
