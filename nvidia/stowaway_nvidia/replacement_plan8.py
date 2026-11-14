"""TP8 forward-only executable plan for one external backend, DX and stock.

All ranks must enter prepare, run and close in identical order.  Per-call
metadata is checked collectively; the cost is deliberately included. This is
an operator carrier, not an adapter to an existing model or graph capture.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import triton


def _load(path):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    if plan.get('schema') != 'stowaway-tp8-replacement-v1' or plan.get('external_backend') != 'async_tp':
        raise ValueError('unsupported plan schema or external backend')
    if plan.get('selection_source') not in ('manual_coverage', 'measured_existing'):
        raise ValueError('unknown selection source')
    if not isinstance(plan.get('entries'), list) or not plan['entries']:
        raise ValueError('nonempty entries required')
    entries = {}
    for e in plan['entries']:
        if set(e) != {'site', 'op', 'M', 'N', 'K', 'choice'}:
            raise ValueError('invalid entry fields')
        if not isinstance(e['site'], str) or not e['site'] or e['op'] not in ('ag', 'rs'):
            raise ValueError('invalid site or operation')
        if e['choice'] not in ('stock', 'async_tp', 'dx'):
            raise ValueError('invalid choice')
        if any(type(e[k]) is not int or e[k] <= 0 for k in ('M', 'N', 'K')) or e['M'] % 8:
            raise ValueError('invalid dimensions')
        key = (e['site'], e['op'], e['M'], e['N'], e['K'])
        if key in entries:
            raise ValueError('duplicate entry')
        entries[key] = e['choice']
    return plan, entries, hashlib.sha256(raw).hexdigest()


def _exchange(value, group):
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, value, group=group)
    return peers


class ReplacementExecutor:
    def __init__(self, path):
        if not dist.is_initialized() or dist.get_world_size() != 8:
            raise ValueError('initialized TP8 world required')
        self.group = dist.new_group(backend='gloo', timeout=dist.default_pg_timeout)
        try:
            plan, entries, digest = _load(path)
            expected = {'tp': 8, 'dtype': 'bfloat16', 'gpu': torch.cuda.get_device_name(),
                        'torch': str(torch.__version__), 'triton': triton.__version__,
                        'cuda_build': torch.version.cuda,
                        'cuda_device_max_connections': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset')}
            if plan.get('runtime') != expected:
                raise ValueError('plan runtime mismatch')
            state = {'digest': digest, 'error': None}
        except Exception as exc:
            plan = entries = digest = None
            state = {'digest': None, 'error': f'{type(exc).__name__}: {exc}'}
        peers = _exchange(state, self.group)
        if any(p['error'] for p in peers) or len({p['digest'] for p in peers}) != 1:
            dist.destroy_process_group(self.group)
            raise RuntimeError(f'plan mismatch: {peers}')
        self.plan, self.entries, self.digest = plan, entries, digest
        self.hits = {'stock': 0, 'async_tp': 0, 'dx': 0}
        self.fallbacks = 0
        self.contexts = {}
        self.closed = False
        if 'async_tp' in entries.values():
            import torch.distributed._symmetric_memory as symm
            if symm._is_test_mode:
                raise RuntimeError('symmetric-memory test fallback active')
            symm.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        self.has_dx = 'dx' in entries.values()
        if self.has_dx:
            from .runtime import initialize
            from .dx8 import DirectExchange8
            self.cuda = initialize(expected_world_size=8)
            # Allocate collectively in a canonical order; contexts do not
            # get created on an unknown-shape fallback.
            shapes = sorted({(m, n, k, op) for (_, op, m, n, k), choice in entries.items() if choice == 'dx'})
            for m, n, k, op in shapes:
                self.contexts[(m, n, k, op)] = DirectExchange8(
                    self.cuda, dist.get_rank(), 8, m, k, n, torch.bfloat16)

    def _metadata(self, site, op, x, weight):
        try:
            if not isinstance(site, str) or not site or op not in ('ag', 'rs'):
                raise ValueError('invalid site or operation')
            if (x.ndim != 2 or weight.ndim != 2 or not x.is_cuda or not weight.is_cuda
                    or not x.is_contiguous() or not weight.is_contiguous()
                    or x.device != weight.device or x.device.index != torch.cuda.current_device()
                    or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
                    or x.requires_grad or weight.requires_grad or x.shape[1] != weight.shape[1]):
                raise ValueError('expected same-device contiguous BF16 forward inputs')
            m = x.shape[0] * 8 if op == 'ag' else x.shape[0]
            n, k = weight.shape
            if m <= 0 or m % 8 or min(n, k) <= 0:
                raise ValueError('invalid shape')
            key = (site, op, m, n, k)
            choice = self.entries.get(key, 'stock')
            return {'key': key, 'choice': choice, 'dtype': str(x.dtype), 'error': None}
        except Exception as exc:
            return {'key': None, 'choice': None, 'dtype': None, 'error': f'{type(exc).__name__}: {exc}'}

    def run(self, site, op, x, weight):
        if self.closed:
            raise RuntimeError('executor closed')
        metadata = self._metadata(site, op, x, weight)
        peers = _exchange(metadata, self.group)
        if any(p['error'] for p in peers) or any(p != peers[0] for p in peers):
            raise RuntimeError(f'inconsistent call metadata: {peers}')
        return self._dispatch(metadata, x, weight)

    def run_batch(self, calls):
        """Validate an entire ordered call batch collectively, then dispatch.

        Each item is (site, op, input, weight). Inputs are all materialized
        before this call, so this path does not express dependent model layers.
        Do not use inside CUDA graph capture. Unknown shapes fall back to stock.
        """
        if self.closed:
            raise RuntimeError('executor closed')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('collective planning inside graph capture is unsupported')
        try:
            calls = tuple(calls)
            if not calls:
                raise ValueError('nonempty call batch required')
            metadata = [self._metadata(*call) for call in calls]
            local = {'calls': metadata, 'error': None}
        except Exception as exc:
            calls = ()
            local = {'calls': [], 'error': f'{type(exc).__name__}: {exc}'}
        peers = _exchange(local, self.group)
        if (any(p['error'] or any(m['error'] for m in p['calls']) for p in peers)
                or any(p != peers[0] for p in peers)):
            raise RuntimeError(f'inconsistent call batch: {peers}')
        return [self._dispatch(meta, x, weight) for meta, (_, _, x, weight) in zip(metadata, calls)]

    def _dispatch(self, metadata, x, weight):
        key, choice = metadata['key'], metadata['choice']
        _, op, m, n, k = key
        if key not in self.entries:
            self.fallbacks += 1
        self.hits[choice] += 1
        if choice == 'stock':
            if op == 'ag':
                full = torch.empty((m, k), device=x.device, dtype=x.dtype)
                dist.all_gather_into_tensor(full, x)
                return full @ weight.T
            partial = x @ weight.T
            out = torch.empty((m // 8, n), device=x.device, dtype=x.dtype)
            dist.reduce_scatter_tensor(out, partial)
            return out
        if choice == 'dx':
            ctx = self.contexts[(m, n, k, op)]
            return ctx.all_gather_gemm(x, weight) if op == 'ag' else ctx.gemm_reduce_scatter(x, weight)
        name = dist.group.WORLD.group_name
        if op == 'ag':
            return torch.ops.symm_mem.fused_all_gather_matmul(x, [weight.T], 0, name, return_A=False)[1][0]
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(x, weight.T, 'sum', 0, name)

    def close(self):
        if self.closed:
            return
        for ctx in self.contexts.values():
            ctx.close()
        if self.has_dx:
            import nvshmem.core
            nvshmem.core.finalize()
        dist.destroy_process_group(self.group)
        self.closed = True
