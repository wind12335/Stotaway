"""Opt-in, narrowly guarded Megatron MoELayer EP2 research adapter."""
import torch
import torch.distributed as dist
import os

from .megatron_moe import SequentialExpertAdapter
from .moe_ep import ep_moe
from .moe_exchange import MoEExchange


class MegatronEPAdapter:
    def __init__(self):
        self.cuda = None
        self.contexts = {}
        self.original = None
        self.hits = self.fallbacks = 0
        mode = os.environ.get("STOWAWAY_MOE_PRE_FC2_SCALE", "1")
        if mode not in ("0", "1"):
            raise ValueError("STOWAWAY_MOE_PRE_FC2_SCALE must be 0 or 1")
        self.pre_fc2_scale = mode == "1"

    def _eligible(self, module, x, intermediate, padding):
        c = module.config
        if (intermediate is not None or padding is not None or module.use_shared_expert
                or module.shared_expert_overlap or getattr(module, "moe_layer_recompute", False)
                or getattr(module, "is_mtp_layer", False) or hasattr(module, "_inference_token_dispatcher")
                or getattr(c, "moe_latent_size", None) or getattr(c, "moe_expert_capacity_factor", None)
                or c.moe_token_dispatcher_type != "alltoall"
                or getattr(c, "recompute_granularity", None) is not None
                or getattr(c, "cuda_graph_impl", None) not in (None, "none")
                or getattr(c, "moe_router_enable_expert_bias", False)
                or getattr(c, "overlap_moe_expert_parallel_comm", False)
                or getattr(c, "overlap_dispatch_backward_with_experts_wgrad", False)
                or tuple(module.fwd_execution_map) != ("route", "expert_compute", "postprocess")
                or module.tp_group.size() != 1 or module.ep_group.size() != 2
                or dist.get_world_size() != 2 or dist.get_process_group_ranks(module.ep_group) != [0, 1]
                or x.dtype != torch.bfloat16 or not x.requires_grad
                or not x.is_contiguous() or not torch.is_grad_enabled()):
            return False
        # Reuse expert guards, including fusion/quantization/bias restrictions.
        ex = module.experts
        if not hasattr(ex, "local_experts"):
            return False
        dummy_counts = torch.empty(ex.num_local_experts, dtype=torch.int32, device=x.device)
        dummy_probs = torch.empty(x.numel() // x.shape[-1], dtype=torch.float32, device=x.device)
        return SequentialExpertAdapter.eligible(ex, x.view(-1, x.shape[-1]), dummy_counts, dummy_probs)

    def install(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        if self.original is not None:
            raise RuntimeError("adapter already installed")
        self.original = MoELayer.forward
        adapter = self

        def forward(module, hidden_states, intermediate_tensors=None, padding_mask=None):
            if not adapter._eligible(module, hidden_states, intermediate_tensors, padding_mask):
                adapter.fallbacks += 1
                return adapter.original(module, hidden_states, intermediate_tensors, padding_mask)
            shape = hidden_states.shape
            x = hidden_states.reshape(-1, shape[-1])
            t, h = x.shape
            e, k = module.config.num_moe_experts, module.config.moe_router_topk
            key = (id(module), t, k, h, e)
            if key not in adapter.contexts:
                if adapter.cuda is None:
                    from .runtime import initialize
                    adapter.cuda = initialize()
                adapter.contexts[key] = MoEExchange(adapter.cuda, dist.get_rank(), t, k, h, e)
            probs, routing = module.route(hidden_states, None)
            values, ids = torch.topk(probs.reshape(t, e).float() * routing.reshape(t, e).float(), k, dim=-1)
            experts = module.experts.local_experts
            w1 = torch.stack([expert.linear_fc1.weight.t() for expert in experts]).contiguous()
            w2 = torch.stack([expert.linear_fc2.weight.t() for expert in experts]).contiguous()
            out = ep_moe(x, ids.to(torch.int32).contiguous(), values.contiguous(), w1, w2,
                         adapter.contexts[key], scale_before_fc2=adapter.pre_fc2_scale)
            adapter.hits += 1
            return out.view(shape), None

        MoELayer.forward = forward

    def close(self):
        if self.original is not None:
            from megatron.core.transformer.moe.moe_layer import MoELayer
            MoELayer.forward = self.original
            self.original = None
        for context in reversed(list(self.contexts.values())):
            context.close()
        self.contexts.clear()
