"""Experimental vLLM 0.11 Llama/Qwen3 sequence-parallel inference adapter.

Derived from Stowaway's vllm_sp_patch.py (MIT); currently one TP group of two
GPUs, eager inference, non-quantized bias-free Llama or Qwen3 only. All ranks must run
the same sequence of bucket sizes and sites.  No CUDA graph or PP support.
"""
import itertools
import os

import torch
import torch.distributed as dist

from .dx import DirectExchange


class LlamaSequenceParallel:
    def __init__(self, cuda, rank, mode, buckets, architecture="llama", world=2, selected_sites=None,
                 forward_plan=None, plan_workload=None):
        if world not in (2, 8) or rank not in range(world) or mode not in ("stock_sp", "dx"):
            raise ValueError("requires world 2 or 8, a matching rank, and stock_sp or dx")
        if not buckets or any(m <= 0 or m % world for m in buckets):
            raise ValueError("positive prefill buckets divisible by world required")
        if architecture not in ("llama", "qwen3"):
            raise ValueError("architecture must be llama or qwen3")
        self.architecture = architecture
        self.selected_sites = (set(selected_sites) if selected_sites is not None else
                               {'qkv', 'proj', 'fc1', 'fc2'})
        if not self.selected_sites <= {'qkv', 'proj', 'fc1', 'fc2'}:
            raise ValueError('unknown projection site')
        self.cuda, self.rank, self.mode, self.buckets, self.world = cuda, rank, mode, set(buckets), world
        # The split column GEMM also changes BF16 numerics for some real
        # prefill sizes. Keep it opt-in until its strict output gate passes.
        self.dx_components = set(os.environ.get("VS_DX_COMPONENTS", "row").split(","))
        if not self.dx_components or not self.dx_components <= {"col", "row"}:
            raise ValueError("VS_DX_COMPONENTS must contain col and/or row")
        self.contexts, self.hits, self.fallbacks = {}, 0, 0
        self.row_diagnostics = []
        # Preserve stock GEMM shape by default: splitting the row GEMM changes
        # BF16 results and failed real-model strict greedy-token comparisons.
        # Overlap is available only by explicitly setting VS_ROW_EXACT=0.
        self.row_exact = os.environ.get("VS_ROW_EXACT", "1") == "1"
        self.reuse_rs_scratch = os.environ.get('STOWAWAY_REUSE_RS_SCRATCH', '0') == '1'
        self._scratch_config_verified = False
        if forward_plan is not None and (world != 2 or mode != 'dx' or selected_sites is not None
                                         or plan_workload is None):
            raise ValueError('forward plan requires TP2, dx, explicit workload and no site override')
        if forward_plan is not None:
            if (not isinstance(plan_workload, dict) or
                    any(not isinstance(plan_workload.get(k), int) or isinstance(plan_workload.get(k), bool)
                        or plan_workload[k] <= 0 for k in ('seq_length', 'micro_batch_size')) or
                    set(buckets) != {plan_workload['seq_length'] * plan_workload['micro_batch_size']}):
                raise ValueError('forward plan workload must match the single configured prefill bucket')
        self.forward_plan, self.plan_workload, self.lookup = forward_plan, plan_workload, None
        self.plan_decisions = {}
        self.original = None

    def _prepare_plan(self):
        if self.forward_plan is None or self.lookup is not None:
            return
        from planner.tp2_lookup import Lookup, runtime_signature
        import triton
        lookup, error = None, None
        try:
            if (torch.cuda.device_count()!=2 or not torch.cuda.can_device_access_peer(0,1) or
                    not torch.cuda.can_device_access_peer(1,0)):
                raise ValueError('forward plan requires bidirectional GPU peer access')
            lookup = Lookup(self.forward_plan, runtime_signature(torch, triton))
            if (lookup.profile.get('plan_source') != 'reduced-probe-model' or
                    lookup.profile.get('phase') != 'forward' or
                    lookup.profile.get('route') != 'dx-direct-exchange-split-gemm'):
                raise ValueError('unsupported forward plan route')
            lookup.validate_workload(self.plan_workload)
            if self.dx_components != {'col', 'row'} or self.row_exact:
                raise ValueError('forward plan requires VS_DX_COMPONENTS=col,row and VS_ROW_EXACT=0')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        states = [None] * self.world
        dist.all_gather_object(states, (lookup.digest if lookup else None, error))
        if any(e for _, e in states) or len({sha for sha, _ in states}) != 1:
            raise RuntimeError(f'forward plan invalid or inconsistent across ranks: {states}')
        self.lookup = lookup

    def _use_dx(self, site, rows, inner, columns, dtype):
        if self.forward_plan is None:
            return site in self.selected_sites
        if self.lookup is None:
            raise RuntimeError('forward plan must be collectively prepared before execution')
        decision = self.lookup.choose(site, rows, inner, columns) if dtype == torch.bfloat16 else 'stock'
        identity = (site, rows, inner, columns, decision)
        self.plan_decisions[identity] = self.plan_decisions.get(identity, 0) + 1
        return decision == 'dx'

    def _context(self, rows, inner, columns, dtype, purpose):
        if self.reuse_rs_scratch and not self._scratch_config_verified:
            raise RuntimeError('RS scratch configuration must be collectively verified before execution')
        key = (rows, inner, columns, dtype, purpose)
        if key not in self.contexts:
            if self.cuda is None:
                from .runtime import initialize
                self.cuda = initialize(expected_world_size=self.world)
            if self.world == 8:
                from .dx8 import DirectExchange8
                self.contexts[key] = DirectExchange8(self.cuda, self.rank, self.world, rows, inner, columns, dtype)
            else:
                self.contexts[key] = DirectExchange(self.cuda, self.rank, self.world, rows, inner, columns, dtype,
                                                    reuse_rs_scratch=self.reuse_rs_scratch and purpose == 'rs')
        return self.contexts[key]

    def _verify_scratch_config(self, group):
        if self._scratch_config_verified:
            return
        local = (self.reuse_rs_scratch, self.world, self.mode, self.row_exact,
                 self.forward_plan is not None)
        states = [None] * self.world
        dist.all_gather_object(states, local, group=group)
        if any(state != states[0] for state in states):
            raise RuntimeError(f'inconsistent RS scratch configuration across ranks: {states}')
        if self.reuse_rs_scratch and (self.world != 2 or self.mode != 'dx' or self.row_exact or self.forward_plan is not None):
            raise ValueError('RS scratch requires TP2 split DX and no existing forward plan')
        self._scratch_config_verified = True

    @staticmethod
    def _weight(layer):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        if (not isinstance(layer.quant_method, UnquantizedLinearMethod)
                or layer.bias is not None or layer.weight.ndim != 2
                or tuple(layer.weight.shape) != (layer.output_size_per_partition,
                                                  layer.input_size_per_partition)
                or not layer.weight.is_contiguous()):
            raise ValueError("only bias-free, unquantized upstream linear weights are supported")
        return layer.weight.detach()

    def _col(self, layer, shard, rows, group, site):
        weight = self._weight(layer)
        if self.mode == "dx" and "col" in self.dx_components and self._use_dx(site, rows, shard.shape[1], weight.shape[0], shard.dtype):
            context = self._context(rows, shard.shape[1], weight.shape[0], shard.dtype, "ag")
            return context.all_gather_gemm(shard.contiguous(), weight)
        full = torch.empty((rows, shard.shape[1]), dtype=shard.dtype, device=shard.device)
        dist.all_gather_into_tensor(full, shard.contiguous(), group=group)
        return layer.quant_method.apply(layer, full, None)

    def _row(self, layer, value, rows, group, site):
        weight = self._weight(layer)
        if self.mode == "dx" and "row" in self.dx_components and self._use_dx(site, rows, value.shape[1], weight.shape[0], value.dtype):
            if self.row_exact:
                if self.world != 2:
                    raise ValueError("exact NVSHMEM reduce-scatter is only implemented for TP2; set VS_ROW_EXACT=0 for the experimental TP8 split-GEMM path")
                # Correctness-control path: preserve the full GEMM shape used
                # by the stock projection, then communicate through NVSHMEM RS.
                # It does not overlap GEMM and communication.
                from .dx import ReduceScatter
                if self.cuda is None:
                    from .runtime import initialize
                    self.cuda = initialize()
                partial = layer.quant_method.apply(layer, value, None).contiguous()
                key = ("row_exact", partial.numel(), partial.dtype)
                if key not in self.contexts:
                    self.contexts[key] = ReduceScatter(self.cuda, self.rank, 2,
                                                       partial.numel(), partial.dtype)
                result, _ = self.contexts[key](partial)
                return result.view(rows // 2, weight.shape[0])
            context = self._context(rows, value.shape[1], weight.shape[0], value.dtype, "rs")
            result = context.gemm_reduce_scatter(value.contiguous(), weight)
            if os.environ.get("VS_ROW_DIAGNOSTIC") == "1":
                partial = layer.quant_method.apply(layer, value, None)
                reference = torch.empty_like(result)
                dist.reduce_scatter_tensor(reference, partial.contiguous(), group=group)
                delta = (result.float() - reference.float()).abs()
                split = torch.cat([part @ weight.t() for part in value.chunk(2)])
                split_ref = torch.empty_like(result)
                dist.reduce_scatter_tensor(split_ref, split, group=group)
                split_delta = (result.float() - split_ref.float()).abs()
                self.row_diagnostics.append((float(delta.max().item()),
                                             int(torch.count_nonzero(delta).item()), delta.numel(),
                                             float(split_delta.max().item()),
                                             float((delta.square().sum()).item()),
                                             float((reference.float().square().sum()).item())))
            return result
        partial = layer.quant_method.apply(layer, value, None)
        output = torch.empty((rows // self.world, weight.shape[0]), dtype=value.dtype, device=value.device)
        dist.reduce_scatter_tensor(output, partial.contiguous(), group=group)
        return output

    def forward(self, model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        from vllm.distributed import get_pp_group, get_tp_group
        source = inputs_embeds if inputs_embeds is not None else input_ids
        rows = source.shape[0]
        if rows not in self.buckets or model.aux_hidden_state_layers:
            self.fallbacks += 1
            return self.original(model, input_ids, positions, intermediate_tensors, inputs_embeds)
        if not (get_pp_group().is_first_rank and get_pp_group().is_last_rank):
            raise RuntimeError("pipeline parallelism is not supported")
        group = get_tp_group().device_group
        if dist.get_process_group_ranks(group) != list(range(self.world)):
            raise RuntimeError("requires a single world TP group")
        self._verify_scratch_config(group)
        self._prepare_plan()
        hidden = inputs_embeds if inputs_embeds is not None else model.get_input_embeddings(input_ids)
        hidden = hidden[self.rank * (rows // self.world):(self.rank + 1) * (rows // self.world)]
        residual = None
        for layer in itertools.islice(model.layers, model.start_layer, model.end_layer):
            if residual is None:
                residual, x = hidden, layer.input_layernorm(hidden)
            else:
                x, residual = layer.input_layernorm(hidden, residual)
            attention = layer.self_attn
            qkv = self._col(attention.qkv_proj, x, rows, group, 'qkv')
            q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
            if self.architecture == "qwen3":
                q = attention.q_norm(q.view(*q.shape[:-1], -1, attention.head_dim)).view(q.shape)
                k = attention.k_norm(k.view(*k.shape[:-1], -1, attention.head_dim)).view(k.shape)
            q, k = attention.rotary_emb(positions, q, k)
            # vLLM attention may write the KV cache on an auxiliary stream;
            # input lifetime is governed by vLLM's own execution contract.
            value = attention.attn(q, k, v)
            hidden = self._row(attention.o_proj, value, rows, group, 'proj')
            x, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.mlp.act_fn(self._col(layer.mlp.gate_up_proj, x, rows, group, 'fc1'))
            hidden = self._row(layer.mlp.down_proj, hidden, rows, group, 'fc2')
        hidden, _ = model.norm(hidden, residual)
        output = torch.empty((rows, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device)
        dist.all_gather_into_tensor(output, hidden.contiguous(), group=group)
        self.hits += 1
        return output

    def install(self):
        if self.architecture == "llama":
            from vllm.model_executor.models.llama import LlamaModel as Model
        else:
            from vllm.model_executor.models.qwen3 import Qwen3Model as Model
        if self.original is not None:
            raise RuntimeError("adapter already installed")
        self.model_class = Model
        self.original = Model.forward
        adapter = self

        def wrapped(model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            return adapter.forward(model, input_ids, positions, intermediate_tensors, inputs_embeds)

        Model.forward = wrapped

    def close(self):
        if self.original is not None:
            self.model_class.forward = self.original
            self.original = None
        for context in reversed(list(self.contexts.values())):
            context.close()
        self.contexts.clear()
