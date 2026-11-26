"""Triton-distributed as external backend B: the DCU port of triton_dist (external overlap, baseline).

Port of nvidia/stowaway_nvidia/external_tp8.py. Source: the Hygon DCU port of Triton-distributed at
OPRS_TRITON_DIST_ROOT (default /root/private_data/tri-dist-hygon), imported read-only; its intra-node
all-gather + GEMM and GEMM + reduce-scatter contexts (triton_dist.kernels.dcu). Forward only: the
backward is the framework's own (autograd.py).

Two properties of the port shape this adapter:
  * its context creators call init_dcu() WITHOUT a host, i.e. they would build a second DushmemHost
    in the process (unsupported: one symmetric-heap world per process) -- so the adapter pre-initialises
    the port's module state with the shared EngineWorld host before any creator runs;
  * the creators offer no release; close() frees what they allocate (symmetric heap + barrier).
"""
import os
import sys

import torch

from ..platform import TRITON_DIST

DEFAULT_ROOT = '/root/private_data/tri-dist-hygon'
_MODULES = {}


def load_modules():
    """(all_gather_gemm, gemm_reduce_scatter, common_ops) modules of the DCU port at the configured root."""
    if _MODULES:
        return _MODULES['ag'], _MODULES['rs'], _MODULES['common']
    root = os.environ.get('OPRS_TRITON_DIST_ROOT') or DEFAULT_ROOT
    root = os.path.realpath(root)
    if not os.path.isdir(os.path.join(root, 'triton_dist', 'kernels', 'dcu')):
        raise RuntimeError(f'triton_dist backend: no DCU port at OPRS_TRITON_DIST_ROOT={root}')
    if root not in [os.path.realpath(p) for p in sys.path if p]:
        sys.path.append(root)                      # after stowaway/src: the port only adds names
    import triton_dist.kernels.dcu.all_gather_gemm as ag
    import triton_dist.kernels.dcu.gemm_reduce_scatter as rs
    import triton_dist.kernels.dcu.common_ops as common
    for module in (ag, rs, common):
        where = os.path.realpath(getattr(module, '__file__', '') or '')
        if not where.startswith(root + os.sep):
            raise RuntimeError(f'triton_dist was imported from {where}, not from OPRS_TRITON_DIST_ROOT={root}')
    _MODULES.update(ag=ag, rs=rs, common=common)
    return ag, rs, common


class TritonDistExchange:
    kind = TRITON_DIST
    backward_kind = 'stock'
    all_gather_dgrad = None

    def __init__(self, world, op, m, n, k, dtype):
        ag, rs, common = load_modules()
        common.init_dcu(world.mpi, world.comm, world.rank, world.size, host=world.host)   # shared host
        self.world, self.op = world, op
        if op == 'ag':
            self.ctx = ag.create_ag_gemm_intra_node_context(m, k, dtype, dtype, world.rank, world.size,
                                                            world.mpi, world.comm)
        elif op == 'rs':
            self.ctx = rs.create_gemm_rs_intra_node_context(m, n, dtype, world.rank, world.size,
                                                            world.mpi, world.comm)
        else:
            raise ValueError(f'triton_dist has no op {op!r}')

    def all_gather_gemm(self, shard, weight):
        """The port's kernel wants the weight K-contiguous [N, K] (asserts stride(1) == 1)."""
        ag, _, _ = load_modules()
        weight = weight.detach()
        if weight.stride(1) != 1:
            weight = weight.contiguous()
        return ag.ag_gemm_intra_node(shard.contiguous(), weight, False, self.ctx)

    def gemm_reduce_scatter(self, activation, weight):
        _, rs, _ = load_modules()
        return rs.gemm_rs_intra_node(activation.contiguous(), weight.detach(), self.ctx)

    def close(self):
        host, ctx = self.world.host, self.ctx
        torch.cuda.synchronize()
        host.barrier_all()
        for ptr in (getattr(self.ctx, 'ws_ptr', None), getattr(self.ctx, 'flags_ptr', None),
                    getattr(self.ctx, 'scatter_ptr', None), getattr(ctx.barrier, 'ptr', None)):
            if ptr:
                host.free(ptr)
        self.ctx = None
        host.barrier_all()
