"""opus fork: Megatron SequentialMLP -> sync-free dropless grouped expert MLP (OPUS_MOE_GG=1, default off).

Megatron SequentialMLP.forward does `tokens_per_expert.tolist()` (host sync) and runs every local expert as
its own pair of GEMMs. This patch runs all local experts' fc1 and fc2 (forward + backward) as hipBLASLt
grouped GEMMs whose per-expert row counts are written on the device (kernels/dcu/moe_expert_mlp.py).
Weights and their gradients stay the experts' own parameters (DDP hooks unchanged).
Eligible: num_local_experts > 1, no fp8/fp4, no bias, gated SiLU (swiglu), no activation clamp,
probs applied after the activation (Megatron default). Anything else -> original forward.
OPUS_MOE_GG_B: virtual group size (rows), default 2048.
"""
import os

import torch
import torch.nn.functional as F

_ON = os.environ.get("OPUS_MOE_GG", "") == "1"
_B = int(os.environ.get("OPUS_MOE_GG_B", "2048"))
# fixed row budget for the grouped-GEMM grid (0 = power-of-two bucket of the call's rows). With EP the
# received rows change every call; a new bucket builds new hipBLASLt plans + triton fill kernels mid-run
# (11-02 vehicle: 5-11 s iterations 4-6), so the vehicle pins one budget.
_MAXROWS = int(os.environ.get("OPUS_MOE_GG_MAXROWS", "0"))
_ST = {"orig": None, "n_gg": 0, "n_orig": 0, "tabs": {}}


def _eligible(self):
    c = self.config
    if self.num_local_experts <= 1 or getattr(c, "fp8", None) or getattr(c, "fp4", None):
        return False
    if getattr(c, "moe_apply_probs_on_input", False) or getattr(c, "activation_func_clamp_value", None) is not None:
        return False
    if not c.gated_linear_unit or c.activation_func is not F.silu or getattr(c, "glu_linear_offset", 0.0):
        return False
    e0 = self.local_experts[0]
    if e0.linear_fc1.bias is not None or e0.linear_fc2.bias is not None:
        return False
    if (getattr(c, "expert_tensor_parallel_size", 1) or 1) != 1:     # expert TP needs the fc2 reduce of Linear
        return False
    return True


def _tables(self):
    from stowaway.kernels.moe_expert_mlp import _ptr_table
    key = id(self)
    t = _ST["tabs"].get(key)
    w1 = [e.linear_fc1.weight for e in self.local_experts]
    w2 = [e.linear_fc2.weight for e in self.local_experts]
    ptrs = (w1[0].data_ptr(), w2[0].data_ptr())
    if t is None or t[0] != ptrs:          # rebuild if the optimizer reallocated parameters
        t = (ptrs, w1, w2, _ptr_table(w1), _ptr_table(w2))
        _ST["tabs"][key] = t
    return t[1], t[2], t[3], t[4]


def _bucket(rows):
    b = 4096
    while b < rows:
        b *= 2
    return b


def _forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, *args, **kwargs):
    if not _eligible(self) or args or kwargs:
        _ST["n_orig"] += 1
        return _ST["orig"](self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, *args, **kwargs)
    from stowaway.kernels.moe_expert_mlp import expert_linear, get_plan
    x = permuted_local_hidden_states
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    rows = x2.shape[0]
    E = self.num_local_experts
    tpe = tokens_per_expert if torch.is_tensor(tokens_per_expert) else torch.tensor(tokens_per_expert)
    offs = torch.cumsum(tpe.to(torch.int64), 0)
    if offs.device.type != "cuda":
        offs = offs.pin_memory().to(x.device, non_blocking=True)   # no host read of the counts on the GPU path
    w1, w2, t1, t2 = _tables(self)
    H = x2.shape[1]
    N1 = w1[0].shape[0]
    Fh = w2[0].shape[1]
    mr = max(_MAXROWS, _bucket(rows)) if _MAXROWS else _bucket(rows)
    p1 = get_plan(E, H, N1, mr, _B)
    p2 = get_plan(E, Fh, w2[0].shape[0], mr, _B)
    h = expert_linear(x2, offs, w1, p1, t1)
    a, b = torch.chunk(h, 2, dim=-1)
    act = F.silu(a) * b
    if permuted_probs is not None:
        act = (act * permuted_probs.reshape(-1).unsqueeze(-1)).to(h.dtype)
    y = expert_linear(act.contiguous(), offs, w2, p2, t2)
    _ST["n_gg"] += 1
    return y.reshape(*shp[:-1], y.shape[-1]), None


def install():
    if not _ON or _ST["orig"] is not None:
        return
    from megatron.core.transformer.moe.experts import SequentialMLP
    _ST["orig"] = SequentialMLP.forward
    SequentialMLP.forward = _forward
    print(f"[opus-moe-gg] SequentialMLP.forward -> sync-free grouped expert MLP (B={_B})", flush=True)


def stats():
    return f"[opus-moe-gg] grouped={_ST['n_gg']} original={_ST['n_orig']}"
