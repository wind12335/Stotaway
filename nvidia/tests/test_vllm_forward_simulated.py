"""End-to-end SP scheduling against a synthetic Llama-like two-rank model.

Uses a stub attention and mock vLLM modules; this is not a real vLLM test.
"""
import datetime
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

from stowaway_nvidia.runtime import finalize, initialize
from stowaway_nvidia.vllm_sp import LlamaSequenceParallel


class Group:
    device_group = None
    is_first_rank = True
    is_last_rank = True


def install_stub():
    group = Group()
    group.device_group = dist.group.WORLD
    parent = types.ModuleType("vllm")
    module = types.ModuleType("vllm.distributed")
    module.get_tp_group = lambda: group
    module.get_pp_group = lambda: group
    parent.distributed = module
    sys.modules["vllm"] = parent
    sys.modules["vllm.distributed"] = module


class Method:
    def apply(self, layer, value, bias):
        return value @ layer.weight.t()


class Linear:
    def __init__(self, inner, outer, dtype, generator):
        self.weight = torch.randn(outer, inner, device="cuda", dtype=dtype, generator=generator) / inner**0.5
        self.quant_method = Method()


class Norm:
    def __call__(self, x, residual=None):
        if residual is None:
            return (x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5)).to(x.dtype)
        residual = x + residual
        return self(residual), residual


class Attention:
    def __init__(self, hidden, dtype, generator):
        self.q_size = self.kv_size = hidden // 2
        self.qkv_proj = Linear(hidden, 3 * hidden // 2, dtype, generator)
        self.o_proj = Linear(hidden // 2, hidden, dtype, generator)

    def rotary_emb(self, positions, q, k):
        return q, k

    def attn(self, q, k, v):
        return (q + k + v) / 3


class Layer:
    def __init__(self, hidden, dtype, generator):
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
        self.self_attn = Attention(hidden, dtype, generator)
        self.mlp = types.SimpleNamespace(gate_up_proj=Linear(hidden, 2 * hidden, dtype, generator),
                                         down_proj=Linear(hidden, hidden, dtype, generator),
                                         act_fn=lambda x: F.silu(x.chunk(2, dim=-1)[0]) * x.chunk(2, dim=-1)[1])


def reference(model, hidden):
    residual = None
    for layer in model.layers:
        if residual is None:
            residual, x = hidden, layer.input_layernorm(hidden)
        else:
            x, residual = layer.input_layernorm(hidden, residual)
        attention = layer.self_attn
        q, k, v = attention.qkv_proj.quant_method.apply(attention.qkv_proj, x, None).chunk(3, dim=-1)
        value = attention.attn(q, k, v)
        hidden = attention.o_proj.quant_method.apply(attention.o_proj, value, None)
        dist.all_reduce(hidden)
        x, residual = layer.post_attention_layernorm(hidden, residual)
        x = layer.mlp.act_fn(layer.mlp.gate_up_proj.quant_method.apply(layer.mlp.gate_up_proj, x, None))
        hidden = layer.mlp.down_proj.quant_method.apply(layer.mlp.down_proj, x, None)
        dist.all_reduce(hidden)
    return model.norm(hidden, residual)[0]


class TestAdapter(LlamaSequenceParallel):
    @staticmethod
    def _weight(layer):
        return layer.weight


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    install_stub()
    rounds = int(os.environ.get("TEST_ROUNDS", "64"))
    for dtype in (torch.float16, torch.bfloat16):
        generator = torch.Generator(device="cuda").manual_seed(501 + rank)
        model = types.SimpleNamespace(layers=[Layer(64, dtype, generator) for _ in range(2)],
                                      start_layer=0, end_layer=2, aux_hidden_state_layers=(), norm=Norm())
        for mode in ("stock_sp", "dx"):
            adapter = TestAdapter(cuda, rank, mode, [128])
            for turn in range(rounds):
                input_generator = torch.Generator(device="cuda").manual_seed(77 + turn)
                inputs = torch.randn(128, 64, dtype=dtype, device="cuda", generator=input_generator)
                expected = reference(model, inputs)
                actual = adapter.forward(model, None, torch.arange(128, device="cuda"), inputs_embeds=inputs)
                torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
            assert adapter.hits == rounds
            print(f"rank={rank} mode={mode} dtype={dtype} rounds={rounds} synthetic_forward PASS", flush=True)
            adapter.close()
    finalize()


if __name__ == "__main__":
    main()
