"""vLLM (0.11, das fork) Llama prefill in sequence-parallel form, so the TP AllReduce of every layer
becomes ReduceScatter + AllGather, and each of the 4 call sites can use the overlap method
(innovation 1) or the original, as chosen by the predictor (innovation 2). Eager mode only.

  VLLM_SP=0 (default)   nothing installs; stock vLLM byte-for-byte
  VLLM_SP=1             LlamaModel.forward runs the SP form when the batch's token count M is in
                        VLLM_SP_BUCKETS (default "16384") and M % TP == 0; any other M (decode,
                        short prefill tails) runs the stock forward.
  VLLM_SP_MODE          rccl : SP form, all 4 sites original (RCCL AG / RS + vendor GEMM)
                        dx   : SP form, all 4 sites overlap method
                        plan : per site, the predictor's choice (planner/planner.py, stage 1;
                               the stage-2 control-plane term is calibrated on training layer cycles
                               and is NOT applied here -- inference forward has no recompute/backward)
                        explicit "qkv=dx,o_proj=dx,fc1=rccl,fc2=dx" also accepted

Per layer, SP form (R = TP size, rank r, M tokens, H hidden):
  x_sh [M/R, H] -> input_layernorm (with residual shard) -> QKV: AG+GEMM -> [M, qkv/R]
  -> rotary + attention on all M tokens (unchanged) -> [M, H/R] -> o_proj: GEMM+RS -> [M/R, H]
  -> post_attention_layernorm -> fc1 (gate_up): AG+GEMM -> SiluAndMul -> fc2 (down): GEMM+RS
Model: embedding output (already all-reduced, full [M, H]) -> my row block; final norm on the shard,
then one RCCL AllGather back to [M, H] for the logits.
Pipeline parallel (PP > 1, e.g. 2 nodes = TP8 x PP2, one TP group per node): stage 0 starts from the
embedding, later stages from the previous stage's full hidden / residual (row block taken locally);
every stage but the last all-gathers hidden and residual back to [M, H] and returns them as
IntermediateTensors (vLLM sends them over the PP group). TP groups must be node-local (the symmetric
heap is), which holds for vLLM's rank order rank = pp_rank * TP + tp_rank with one stage per node.
Numerics: per-row ops (norm, residual, rotary, attention) are unchanged; the sum order of RS differs
from AR, so outputs match stock to bf16 rounding, not bit for bit.
Contexts (symmetric heap): one AG ctx per (M, H) shared by QKV and fc1, one RS ctx per (M, H) shared
by o_proj and fc2 -- the same sharing as the Megatron vehicle. Built at the first SP forward, which in
vLLM V1 is the memory-profiling run, so the heap is on the books before the KV cache is sized.
"""
import os
import time

import torch

_ST = {"installed": False, "rank": None, "world": None, "buckets": set(), "mode": "rccl",
       "choice": {}, "ag_ctx": {}, "rs_ctx": {}, "orig": None, "n_sp": 0, "n_stock": 0,
       "sp_s": 0.0, "printed": set()}
SITES = ("qkv", "o_proj", "fc1", "fc2")


def _group():
    from vllm.distributed import get_tp_group
    return get_tp_group().device_group


def _w_nk(layer):
    """weight as [N_out, K_in] (a view). das fork VLLM_USE_NN stores [K_in, N_out]."""
    w = layer.weight
    k_in = layer.input_size_per_partition
    n_out = layer.output_size_per_partition
    assert k_in != n_out, "square weight: orientation ambiguous"
    if w.shape[0] == k_in and w.shape[1] == n_out:
        return w.t()
    assert w.shape[0] == n_out and w.shape[1] == k_in, f"unexpected weight shape {tuple(w.shape)}"
    return w


def _parse_choice(mode, model, M):
    if "=" in mode:
        d = dict(kv.split("=") for kv in mode.split(","))
        return {s: d.get(s, "rccl") for s in SITES}
    if mode in ("rccl", "dx"):
        return {s: mode for s in SITES}
    assert mode == "plan"
    import sys
    sys.path.insert(0, os.path.join(os.environ.get("SW_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "planner"))
    import planner
    cfg = model.config
    h, f = cfg.hidden_size, cfg.intermediate_size
    nh, kv = cfg.num_attention_heads, cfg.num_key_value_heads
    res = planner.plan(h, f, nh, kv, cfg.num_hidden_layers, 1, model=os.environ.get("VLLM_SP_PLAN_MODEL"),
                       d_cycle=1e9, quiet=True, stage2=False, M=M)
    name = {"QKV": "qkv", "o_proj": "o_proj", "fc1": "fc1", "fc2": "fc2"}
    out = {name[r["name"]]: ("dx" if r["deploy"] == "dx" else "rccl") for r in res["rows"]}
    if _ST["rank"] == 0:
        print(f"[vllm_sp] predictor (stage 1) at M={M}: {out}", flush=True)
    return out


def _ag_ctx(M, K, dtype):
    key = (M, K, dtype)
    if key not in _ST["ag_ctx"]:
        from stowaway.kernels.gemm_rs_dx import DXAGGemmContext
        _ST["ag_ctx"][key] = DXAGGemmContext(M, K, dtype, _ST["rank"], _ST["world"])
    return _ST["ag_ctx"][key]


def _rs_ctx(M, N, dtype):
    key = (M, N, dtype)
    if key not in _ST["rs_ctx"]:
        from stowaway.kernels.gemm_rs_dx import DXGemmRSContext
        _ST["rs_ctx"][key] = DXGemmRSContext(M, N, dtype, _ST["rank"], _ST["world"])
    return _ST["rs_ctx"][key]


def _col(layer, x_sh, M, how):
    """column-parallel site: [M/R, K] shard -> [M, N/R]"""
    if how == "dx":
        from stowaway.kernels.gemm_rs_dx import dx_ag_gemm_op
        out, _ = dx_ag_gemm_op(x_sh.contiguous(), _w_nk(layer), _ag_ctx(M, x_sh.shape[1], x_sh.dtype))
        return out
    import torch.distributed as dist
    full = torch.empty(M, x_sh.shape[1], dtype=x_sh.dtype, device=x_sh.device)
    dist.all_gather_into_tensor(full, x_sh.contiguous(), group=_group())
    return layer.quant_method.apply(layer, full, None)


def _row(layer, a, M, how):
    """row-parallel site: [M, K/R] -> [M/R, N]"""
    R = _ST["world"]
    N = layer.output_size_per_partition
    out = torch.empty(M // R, N, dtype=a.dtype, device=a.device)
    if how == "dx":
        from stowaway.kernels.gemm_rs_dx import dx_gemm_rs_op
        dx_gemm_rs_op(a.contiguous(), _w_nk(layer), out, _rs_ctx(M, N, a.dtype))
        return out
    import torch.distributed as dist
    part = layer.quant_method.apply(layer, a, None)
    dist.reduce_scatter_tensor(out, part.contiguous(), group=_group())
    return out


def _layer_sp(layer, positions, h_sh, res, M, ch):
    attn = layer.self_attn
    if res is None:
        res = h_sh
        x = layer.input_layernorm(h_sh)
    else:
        x, res = layer.input_layernorm(h_sh, res)
    qkv = _col(attn.qkv_proj, x, M, ch["qkv"])
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q, k = attn.rotary_emb(positions, q, k)
    o = attn.attn(q, k, v)
    o_sh = _row(attn.o_proj, o, M, ch["o_proj"])
    x, res = layer.post_attention_layernorm(o_sh, res)
    gu = _col(layer.mlp.gate_up_proj, x, M, ch["fc1"])
    a = layer.mlp.act_fn(gu)
    return _row(layer.mlp.down_proj, a, M, ch["fc2"]), res


def _gather(x_sh, M):
    import torch.distributed as dist
    full = torch.empty(M, x_sh.shape[1], dtype=x_sh.dtype, device=x_sh.device)
    dist.all_gather_into_tensor(full, x_sh.contiguous(), group=_group())
    return full


def _sp_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    st = _ST
    src = inputs_embeds if inputs_embeds is not None else input_ids
    M = src.shape[0]
    R = st["world"]
    if M not in st["buckets"] or M % R or len(self.aux_hidden_state_layers) > 0:
        st["n_stock"] += 1
        return st["orig"](self, input_ids, positions, intermediate_tensors, inputs_embeds)
    from vllm.distributed import get_pp_group
    pp = get_pp_group()
    if M not in st["choice"]:
        st["choice"][M] = _parse_choice(st["mode"], self, M)
    ch = st["choice"][M]
    t0 = time.perf_counter()
    mp = M // R
    r = st["rank"]
    if pp.is_first_rank:
        hidden = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
        h_sh, res = hidden[r * mp:(r + 1) * mp], None
    else:   # pipeline stage > 0: full [M, H] tensors from the previous stage -> my row block
        h_sh = intermediate_tensors["hidden_states"][r * mp:(r + 1) * mp]
        res = intermediate_tensors["residual"][r * mp:(r + 1) * mp]
    for layer in self.layers[self.start_layer:self.end_layer]:
        h_sh, res = _layer_sp(layer, positions, h_sh, res, M, ch)
    if not pp.is_last_rank:   # hand full tensors to the next stage (the shape vLLM's PP buffers expect)
        from vllm.sequence import IntermediateTensors
        st["n_sp"] += 1
        st["sp_s"] += time.perf_counter() - t0
        return IntermediateTensors({"hidden_states": _gather(h_sh, M), "residual": _gather(res, M)})
    h_sh, _ = self.norm(h_sh, res)
    full = _gather(h_sh, M)
    st["n_sp"] += 1
    st["sp_s"] += time.perf_counter() - t0
    if r == 0 and M not in st["printed"]:
        st["printed"].add(M)
        print(f"[vllm_sp] first SP forward M={M} choice={ch}", flush=True)
    return full


def stats_line():
    return (f"[vllm_sp] sp_forwards={_ST['n_sp']} stock_forwards={_ST['n_stock']} "
            f"sp_host_s={_ST['sp_s']:.2f} mode={_ST['mode']}")


def install(rank, world):
    if _ST["installed"] or os.environ.get("VLLM_SP", "0") != "1":
        return
    from vllm.model_executor.models.llama import LlamaModel
    _ST.update(rank=rank, world=world, mode=os.environ.get("VLLM_SP_MODE", "rccl"),
               buckets={int(b) for b in os.environ.get("VLLM_SP_BUCKETS", "16384").split(",")})
    _ST["orig"] = LlamaModel.forward
    LlamaModel.forward = _sp_forward
    _ST["installed"] = True
    if rank == 0:
        import atexit
        atexit.register(lambda: print(stats_line(), flush=True))
        print(f"[vllm_sp] installed: mode={_ST['mode']} buckets={sorted(_ST['buckets'])} R={world}", flush=True)
