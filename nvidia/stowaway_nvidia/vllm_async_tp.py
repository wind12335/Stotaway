"""TP sequence-parallel vLLM adapter using PyTorch symmetric-memory operators."""
import os
import torch
import torch.distributed as dist
from .vllm_sp import LlamaSequenceParallel


class AsyncTPLlamaSequenceParallel(LlamaSequenceParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_ready = False
        self.backend_hits = {'ag': 0, 'rs': 0, 'dx_ag': 0, 'dx_rs': 0}
        self.dx_sites = set(filter(None, os.environ.get('VS_ASYNC_DX_SITES', '').split(',')))
        self.auto_plan = None
        self.auto_hits = {}
        plan_path = os.environ.get('VS_AUTO_PLAN8')
        if plan_path:
            from .auto_plan8 import AutoPlan8
            if self.dx_sites or self.world != 8:
                raise ValueError('automatic plan requires TP8 and no manual DX sites')
            self.auto_plan = AutoPlan8(plan_path, 'llama70_infer')
        self.auto_verified = False
        if not self.dx_sites <= {'qkv', 'proj', 'fc1', 'fc2'}:
            raise ValueError('unknown DX replacement site')

    def _ensure_async(self, group):
        if not self.async_ready:
            if self.auto_plan is not None:
                peers = [None] * self.world
                dist.all_gather_object(peers, self.auto_plan.digest, group=group)
                if len(set(peers)) != 1:
                    raise RuntimeError('automatic plan differs across ranks')
                self.auto_verified = True
            import torch.distributed._symmetric_memory as symm
            if symm._is_test_mode:
                raise RuntimeError('Async-TP symmetric memory test mode is active')
            symm.enable_symm_mem_for_group(group.group_name)
            self.async_ready = True

    def _choice(self, site, op, rows, weight, dtype):
        if self.auto_plan is None:
            return 'dx' if site in self.dx_sites else 'async_tp'
        choice = self.auto_plan.choose(site, op, rows, weight.shape[0], weight.shape[1],
                                       'bfloat16' if dtype == torch.bfloat16 else str(dtype))
        key = (site, op, rows, weight.shape[0], weight.shape[1], choice)
        self.auto_hits[str(key)] = self.auto_hits.get(str(key), 0) + 1
        return choice

    def _col(self, layer, shard, rows, group, site):
        self._ensure_async(group)
        weight = self._weight(layer)
        choice = self._choice(site, 'ag', rows, weight, shard.dtype)
        if choice == 'dx':
            self.backend_hits['dx_ag'] += 1
            return self._context(rows, shard.shape[1], weight.shape[0], shard.dtype, 'ag').all_gather_gemm(shard.contiguous(), weight)
        if choice == 'stock':
            self.backend_hits.setdefault('stock_ag', 0); self.backend_hits['stock_ag'] += 1
            full = torch.empty((rows, shard.shape[1]), dtype=shard.dtype, device=shard.device)
            dist.all_gather_into_tensor(full, shard.contiguous(), group=group)
            return layer.quant_method.apply(layer, full, None)
        self.backend_hits['ag'] += 1
        return torch.ops.symm_mem.fused_all_gather_matmul(shard.contiguous(), [weight.T], 0, group.group_name, return_A=False)[1][0]

    def _row(self, layer, value, rows, group, site):
        self._ensure_async(group)
        weight = self._weight(layer)
        choice = self._choice(site, 'rs', rows, weight, value.dtype)
        if choice == 'dx':
            self.backend_hits['dx_rs'] += 1
            return self._context(rows, value.shape[1], weight.shape[0], value.dtype, 'rs').gemm_reduce_scatter(value.contiguous(), weight)
        if choice == 'stock':
            self.backend_hits.setdefault('stock_rs', 0); self.backend_hits['stock_rs'] += 1
            partial = layer.quant_method.apply(layer, value, None)
            out = torch.empty((rows // self.world, weight.shape[0]), dtype=value.dtype, device=value.device)
            dist.reduce_scatter_tensor(out, partial.contiguous(), group=group)
            return out
        self.backend_hits['rs'] += 1
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(value.contiguous(), weight.T, 'sum', 0, group.group_name)
