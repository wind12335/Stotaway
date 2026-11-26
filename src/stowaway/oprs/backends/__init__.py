"""Overlap-backend adapters: the uniform exchange interface the dispatcher and autograd wrappers use.

One adapter object ("exchange") per plan allocation `(method, op, M, N, K)`. All of them take and
return plain torch tensors in the sequence-parallel layout (R = TP size):

    all_gather_gemm(shard, weight)                column-parallel forward, op 'ag'
        shard [M/R, K] x weight [N, K]  ->  [M, N]   (AG of the input shards, then GEMM)
    gemm_reduce_scatter(activation, weight)       row-parallel forward, op 'rs'
        activation [M, K/R] x weight [N, K/R]  ->  [M/R, N]   (GEMM, then RS of the sum)
    all_gather_dgrad(grad_shard, weight, activation)   optional row-parallel backward:
        grad_shard [M/R, N]  ->  (grad_input [M, K/R], grad_weight [N, K/R], None);
        absent = the framework's own backward (autograd.py)
    close()                                       collective: every rank calls it at the same point

Methods (platform.METHODS): deo = Hygon's own overlap method (NVIDIA name dx); async_tp and
triton_dist = the external overlap backends B compared at operator level on BW.
"""
from ..platform import DEO, BACKENDS

METHODS = (DEO,) + BACKENDS


def dtype_name(dtype) -> str:
    return str(dtype).replace('torch.', '')


def create_exchange(method, world, op, m, n, k, dtype, *, group=None, deo_contexts=None):
    """Build the exchange object of one allocation. Context creation is collective for the
    heap-using methods: all ranks must call this for the same allocations in the same order."""
    if method == DEO:
        from .deo import DeoExchange
        if world is None or deo_contexts is None:
            raise ValueError('deo needs an EngineWorld and DeoContexts')
        return DeoExchange(deo_contexts, op, m, n, k, dtype)
    if method == 'async_tp':
        from .async_tp import AsyncTpExchange
        if group is None:
            raise ValueError('async_tp needs the torch process group of the TP ranks')
        return AsyncTpExchange(group, op)
    if method == 'triton_dist':
        from .triton_dist import TritonDistExchange
        if world is None:
            raise ValueError('triton_dist needs an EngineWorld')
        return TritonDistExchange(world, op, m, n, k, dtype)
    raise ValueError(f'unknown method {method!r}')
