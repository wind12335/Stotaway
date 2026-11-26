"""DEO (Hygon's own overlap method; NVIDIA name DX) as a candidate.

The production kernels of src/stowaway/kernels/gemm_rs_dx.py behind the uniform exchange interface:
  column-parallel forward   dx_ag_gemm_op    AG of the SP input shards fused with block GEMMs
  row-parallel forward      dx_gemm_rs_op    GEMM in row blocks, each pushed to its owner rank as soon
                                              as it is computed, one fused sum at the home rank
  row-parallel backward     dx_ag_bwd_op     AG of the grad_out shards fused with per-block dgrad
                                              (deo_bwd = 'ag', the vehicle's form; the NVIDIA runtime's
                                              row backward is the same idea). deo_bwd = 'stock' leaves
                                              the backward to autograd.py.

These are the forward-only fused operators that the operator table measures. Contexts are shared by
shape exactly as the Megatron vehicle shares them: QKV and fc1 share one AG context (same M, K),
o_proj and fc2 share one RS context and one AG+dgrad context (same M, N).
"""
import torch

from ...kernels.common_ops import dcu_state, hip
from ...kernels.gemm_rs_dx import DXAGBwdContext, DXAGGemmContext, DXGemmRSContext
from ...kernels.gemm_rs_dx import dx_ag_bwd_op, dx_ag_gemm_op, dx_gemm_rs_op
from ..platform import DEO

BWD_MODES = ('ag', 'stock')


def _free_rs_like(ctx):
    """DXGemmRSContext / DXAGBwdContext have no close() of their own (the vehicle never drops them)."""
    host, h = dcu_state()['host'], hip()
    torch.cuda.synchronize()
    host.barrier_all()
    for ptr in (ctx.slots, ctx.arr, ctx.ack):
        host.free(ptr)
    for event in ctx.ev:
        h.hipEventDestroy(event)
    ctx.slots = ctx.arr = ctx.ack = None
    ctx.tickets, ctx.ev = None, []
    host.barrier_all()


class DeoContexts:
    """Context pools of one engine world, keyed by shape and dtype (collective creation and close)."""

    def __init__(self, world, bwd='ag'):
        if bwd not in BWD_MODES:
            raise ValueError(f'deo_bwd must be one of {BWD_MODES}, got {bwd!r}')
        self.world, self.bwd = world, bwd
        self.ag, self.rs, self.agbwd = {}, {}, {}

    @staticmethod
    def _get(store, key, cls, world, dims):
        if key not in store:
            store[key] = cls(*dims, world.rank, world.size)
        return store[key]

    def ag_context(self, m, k, dtype):
        return self._get(self.ag, (m, k, dtype), DXAGGemmContext, self.world, (m, k, dtype))

    def rs_context(self, m, n, dtype):
        return self._get(self.rs, (m, n, dtype), DXGemmRSContext, self.world, (m, n, dtype))

    def agbwd_context(self, m, n, dtype):
        return self._get(self.agbwd, (m, n, dtype), DXAGBwdContext, self.world, (m, n, dtype))

    def close(self):
        """Collective: same order on every rank (the dispatcher closes what it created, sorted)."""
        if not (self.ag or self.rs or self.agbwd):
            return
        torch.cuda.synchronize()
        self.world.host.barrier_all()
        for ctx in sorted(self.ag.values(), key=lambda c: (c.M, c.K, str(c.dtype))):
            ctx.close()
        for store in (self.rs, self.agbwd):
            for ctx in sorted(store.values(), key=lambda c: (c.M, c.N, str(c.dtype))):
                _free_rs_like(ctx)
        self.ag.clear(), self.rs.clear(), self.agbwd.clear()


class DeoExchange:
    kind = DEO

    def __init__(self, contexts, op, m, n, k, dtype):
        self.contexts, self.op = contexts, op
        self._backward = None
        # instance attribute shadows the method: None = the framework's own backward (autograd.py)
        self.all_gather_dgrad = None
        if op == 'ag':
            self.ctx = contexts.ag_context(m, k, dtype)
        elif op == 'rs':
            self.ctx = contexts.rs_context(m, n, dtype)
            if contexts.bwd == 'ag':
                self._backward = contexts.agbwd_context(m, n, dtype)
                self.all_gather_dgrad = self._dgrad
        else:
            raise ValueError(f'deo has no op {op!r}')

    @property
    def backward_kind(self):
        return 'ag' if self._backward is not None else 'stock'

    def _dgrad(self, grad_shard, weight, activation):
        """(grad_input [M, K/R], grad_weight [N, K/R], None)."""
        grad_input, grad_weight, _ = dx_ag_bwd_op(grad_shard.contiguous(), weight, activation,
                                                  self._backward)
        return grad_input, grad_weight, None

    def all_gather_gemm(self, shard, weight):
        return dx_ag_gemm_op(shard.contiguous(), weight, self.ctx)[0]

    def gemm_reduce_scatter(self, activation, weight):
        out = torch.empty(activation.shape[0] // self.ctx.R, self.ctx.N, dtype=self.ctx.dtype,
                          device=activation.device)
        dx_gemm_rs_op(activation.contiguous(), weight, out, self.ctx)
        return out

    def close(self):
        pass            # contexts are pooled; DeoContexts.close() releases them
