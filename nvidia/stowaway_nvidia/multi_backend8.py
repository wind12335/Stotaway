"""Collectively prepared TP8 dispatcher for a single external backend + DX.

Contexts are preallocated in canonical order so all ranks make matching
NVSHMEM allocations. Unknown shapes use the plan's explicit default.
"""
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .auto_plan8 import AutoPlan8


class MultiBackend8:
    def __init__(self, path, workload, group=None):
        self.group = group or dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        if dist.get_world_size(self.group) != 8 or dist.get_process_group_ranks(self.group) != list(range(8)):
            raise ValueError('one world TP8 group is required')
        self.plan, error = None, None
        try:
            self.plan = AutoPlan8(path, workload)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        peers = [None] * 8
        dist.all_gather_object(peers, (self.plan.digest if self.plan else None, error), group=self.group)
        if any(e for _, e in peers) or len({digest for digest, _ in peers}) != 1:
            raise RuntimeError(f'inconsistent plan: {peers}')
        # Dynamic external contexts cannot be safely created independently on
        # unseen shapes, so unknown shapes must use stock.
        if self.plan.backend == 'async_tp' or self.plan.default != 'stock':
            raise ValueError('multi-backend plan requires stock for unknown shapes')
        self.dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[self.plan.dtype]
        self.contexts, self.hits = {}, {}
        needed = sorted({(choice, op, m, n, k) for (_, op, m, n, k), choice in self.plan.entries.items()
                         if choice != 'stock'})
        self.cuda = None
        self._nvshmem_owned = False
        if any(method == 'flashoverlap' for method, *_ in needed):
            flash_root = Path(os.environ['STOWAWAY_FLASH_ROOT'])
            if self.dtype == torch.bfloat16 and os.environ.get('STOWAWAY_FLASH_BF16') != '1':
                raise ValueError('BF16 FlashOverlap requires an explicitly built BF16 library')
            torch.ops.load_library(str(flash_root / 'build/lib/libst_pybinding.so'))
        from .external_tp8 import FlashOverlapExchange, SyncopateExchange, TritonDistExchange
        if any(method in ('dx', 'triton_dist', 'syncopate') for method, *_ in needed):
            from .runtime import initialize
            self.cuda = initialize(expected_world_size=8)
            self._nvshmem_owned = True
            if self.plan.backend in ('triton_dist', 'syncopate'):
                # Triton-distributed's JIT registers NVSHMEM device modules
                # only when its world marker is set. The shared runtime above
                # owns initialization; do not initialize NVSHMEM a second time.
                import triton_dist.utils as td_utils
                if td_utils.is_shmem_initialized():
                    raise RuntimeError('unexpected Triton-distributed NVSHMEM owner')
                td_utils._TRITON_DIST_WORLD = self.group
                td_utils._TRITON_DIST_LOCAL_WORLD_SIZE = 8
        from .dx8 import DirectExchange8
        for method, op, m, n, k in needed:
            if method == 'dx':
                ctx = DirectExchange8(self.cuda, self.rank, 8, m, k, n, self.dtype)
            elif method == 'triton_dist':
                ctx = TritonDistExchange(self.rank, m, n, k, self.dtype, op)
            elif method == 'syncopate':
                ctx = SyncopateExchange(self.rank, m, n, k, self.dtype, op, os.environ['STOWAWAY_SYNC_WORK'])
            elif method == 'flashoverlap':
                cfg = flash_root / f'configs/m{m}n{n}k{k}_a800.json'
                # Each shape owns a communicator. Reusing a NCCL unique ID
                # for a second live communicator makes initialization fail.
                uid_list = [torch.ops.flashoverlap_op.generate_nccl_id() if self.rank == 0 else None]
                dist.broadcast_object_list(uid_list, src=0, group=self.group)
                ctx = FlashOverlapExchange(self.rank, m, n, k, cfg, flash_root / 'build/lib/libst_pybinding.so', uid_list[0], self.dtype)
            else:
                raise ValueError(f'unsupported backend {method}')
            self.contexts[(method, op, m, n, k)] = ctx

    def choose(self, site, op, x, weight):
        m = x.shape[0] * 8 if op == 'ag' else x.shape[0]
        n, k = weight.shape
        dtype = str(x.dtype).removeprefix('torch.')
        choice = self.plan.choose(site, op, m, n, k, dtype)
        key = (site, op, m, n, k, choice)
        self.hits[key] = self.hits.get(key, 0) + 1
        return choice, (m, n, k)

    def run(self, site, op, x, weight):
        if op not in ('ag', 'rs') or x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError('invalid operation or tensor shapes')
        if x.dtype != weight.dtype or x.device != weight.device:
            raise ValueError('input and weight dtype/device must match')
        if x.requires_grad or weight.requires_grad:
            raise ValueError('run is forward-only; training must use exchange_for with the autograd wrapper')
        choice, (m, n, k) = self.choose(site, op, x, weight)
        if choice == 'stock':
            if op == 'ag':
                full = torch.empty((m, k), dtype=x.dtype, device=x.device)
                dist.all_gather_into_tensor(full, x.contiguous(), group=self.group)
                return full @ weight.T
            partial = x @ weight.T
            out = torch.empty((m // 8, n), dtype=x.dtype, device=x.device)
            dist.reduce_scatter_tensor(out, partial.contiguous(), group=self.group)
            return out
        ctx = self.contexts[(choice, op, m, n, k)]
        return (ctx.all_gather_gemm(x.contiguous(), weight.contiguous()) if op == 'ag'
                else ctx.gemm_reduce_scatter(x.contiguous(), weight.contiguous()))

    def exchange_for(self, site, op, m, n, k, dtype):
        """Return the context for a model's autograd wrapper, or None for stock."""
        choice = self.plan.choose(site, op, m, n, k, str(dtype).removeprefix('torch.'))
        key = (site, op, m, n, k, choice)
        self.hits[key] = self.hits.get(key, 0) + 1
        return None if choice == 'stock' else self.contexts[(choice, op, m, n, k)]

    def close(self):
        self._closed_hits = dict(self.hits)
        for key in reversed(sorted(self.contexts)):
            self.contexts[key].close()
        self.contexts.clear()
        if self._nvshmem_owned:
            import nvshmem.core
            nvshmem.core.finalize()
            self._nvshmem_owned = False
            if self.plan.backend in ('triton_dist', 'syncopate'):
                import triton_dist.utils as td_utils
                td_utils._TRITON_DIST_WORLD = None
                td_utils._TRITON_DIST_LOCAL_WORLD_SIZE = None
