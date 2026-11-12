"""Diagnostic adapter for Megatron 571370c SequentialMLP expert GEMMs."""
import torch
import torch.nn.functional as F

from .moe_grouped import expert_linear


class SequentialExpertAdapter:
    """Replace eligible SequentialMLP calls; unsupported configurations fall back."""
    def __init__(self):
        self.original = None
        self.hits = 0
        self.fallbacks = 0

    @staticmethod
    def eligible(module, x, counts, probs):
        c = module.config
        if (module.num_local_experts < 2 or getattr(c, "fp8", None) or getattr(c, "fp4", None)
                or getattr(c, "moe_apply_probs_on_input", False)
                or getattr(c, "activation_func_clamp_value", None) is not None
                or getattr(c, "gradient_accumulation_fusion", False)
                or getattr(c, "delay_wgrad_compute", False)
                or not c.gated_linear_unit or c.activation_func is not F.silu
                or getattr(c, "glu_linear_offset", 0.0)
                or (getattr(c, "expert_tensor_parallel_size", 1) or 1) != 1
                or getattr(c, "bias_activation_fusion", False)
                or x.ndim != 2 or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16)
                or not x.is_contiguous() or not isinstance(counts, torch.Tensor)
                or counts.ndim != 1 or counts.numel() != module.num_local_experts
                or counts.device.type not in ("cpu", "cuda") or (counts.is_cuda and counts.device != x.device)
                or counts.dtype not in (torch.int32, torch.int64)
                or probs is None or probs.numel() != x.shape[0] or probs.device != x.device):
            return False
        for expert in module.local_experts:
            if (expert.linear_fc1.bias is not None or expert.linear_fc2.bias is not None
                    or expert.linear_fc1.weight.dtype != x.dtype or expert.linear_fc2.weight.dtype != x.dtype
                    or expert.linear_fc1.weight.shape[1] != x.shape[1]
                    or expert.linear_fc1.weight.shape[0] != 2 * expert.linear_fc2.weight.shape[1]
                    or expert.linear_fc2.weight.shape[0] != x.shape[1]):
                return False
        return True

    def install(self):
        from megatron.core.transformer.moe.experts import SequentialMLP
        if self.original is not None:
            raise RuntimeError("adapter already installed")
        self.original = SequentialMLP.forward
        adapter = self

        def forward(module, x, counts, probs):
            if not adapter.eligible(module, x, counts, probs):
                adapter.fallbacks += 1
                if adapter.fallbacks == 1:
                    c = module.config
                    print(f"STOWAWAY_MOE_GG_FALLBACK rank={torch.distributed.get_rank()} "
                          f"counts={type(counts).__name__}/{getattr(counts, 'device', None)} "
                          f"x={x.dtype}/{x.device} experts={module.num_local_experts} "
                          f"fusion={getattr(c, 'bias_activation_fusion', None)} "
                          f"expert_tp={getattr(c, 'expert_tensor_parallel_size', None)}", flush=True)
                return adapter.original(module, x, counts, probs)
            # Upstream dispatcher can already have copied counts to the host;
            # moving them back does not remove its existing synchronization.
            offsets = counts.to(device=x.device).cumsum(0)
            # Stack each call to preserve autograd links to Megatron parameters.
            # This copy is deliberately diagnostic, not a performance solution.
            w1 = torch.stack([e.linear_fc1.weight.t() for e in module.local_experts]).contiguous()
            w2 = torch.stack([e.linear_fc2.weight.t() for e in module.local_experts]).contiguous()
            up = expert_linear(x, w1, offsets)
            gate, value = up.chunk(2, -1)
            hidden = (F.silu(gate) * value * probs.reshape(-1, 1)).to(x.dtype).contiguous()
            output = expert_linear(hidden, w2, offsets)
            adapter.hits += 1
            return output, None

        SequentialMLP.forward = forward

    def close(self):
        if self.original is not None:
            from megatron.core.transformer.moe.experts import SequentialMLP
            SequentialMLP.forward = self.original
            self.original = None
