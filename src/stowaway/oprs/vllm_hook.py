"""vLLM (0.11 das fork) prefill in sequence-parallel form, every call site dispatched by an OPRS plan.

Hygon port of nvidia/stowaway_nvidia/vllm_multi_tp8.py (MultiBackendLlamaSequenceParallel), built on
the sequence-parallel form of src/stowaway/integrations/vllm_sp_patch.py (that file and its
predictor stay untouched: it is the old `plan`, a different system).

  choice serial   the SP form's own serial path: RCCL AllGather -> the layer's own GEMM, or the
                  layer's own GEMM -> RCCL ReduceScatter        (= the "SP-RCCL" reference path)
  choice <method> the fused forward of DEO / Async-TP / Triton-distributed (backends/), no autograd

Per layer (R = TP size, rank r, M tokens, H hidden):
  x_sh [M/R, H] -> input_layernorm -> qkv: AG+GEMM -> [M, qkv/R] -> rotary + attention -> [M, H/R]
  -> proj: GEMM+RS -> [M/R, H] -> post_attention_layernorm -> fc1: AG+GEMM -> act -> fc2: GEMM+RS
The embedding output is all-reduced [M, H]: rank r takes row block r; the final norm runs on the
shard and one RCCL AllGather restores [M, H] for the logits. Numerics: the sum order of a TP
reduction changes (AllReduce -> ReduceScatter), so outputs agree with stock to bf16 rounding.

The dispatcher is built at the FIRST SP forward (in vLLM V1 the memory-profiling run, so the
contexts are on the books before the KV cache is sized): plan digest compared across the TP ranks,
then every context of the plan in canonical order. The SP form runs only when the batch's token
count is one of the buckets; any other M (decode, short tails) runs the stock forward. The plan's
entries therefore only ever match at those M. Pipeline parallel is not supported here (one stage).

Weight layout: the das fork may store a weight as [K_in, N_out]; every method gets the [N_out,
K_in] view, except Triton-distributed's AG kernel, which needs K-contiguous [N, K] -- that one
gets a cached contiguous copy (made in the profiling run, one per weight).

  install()  entry-script prelude: after pydushmem mpi_init / init_dcu, before LLM()
  report()   hits, fallbacks, runs, counters of this rank
"""
import time

from .platform import TRITON_DIST

SITES = ("qkv", "proj", "fc1", "fc2")
_ST = {"installed": False, "rank": None, "world": None, "buckets": set(), "engine": None, "plan": None,
       "workload": None, "methods": None, "disp": None, "orig": None, "n_sp": 0, "n_stock": 0,
       "sp_s": 0.0, "printed": set(), "kc": {}}

log = lambda message: print(f"[oprs] {message}", flush=True)      # noqa: E731


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


def _k_contiguous(layer, w):
    """K-contiguous [N, K] copy of a transposed-view weight; re-made if the parameter is rewritten."""
    p = layer.weight
    key, version = (id(p), p.data_ptr()), p._version
    hit = _ST["kc"].get(key)
    if hit is None or hit[0] != version:
        hit = (version, w.detach().contiguous())
        _ST["kc"][key] = hit
    return hit[1]


def _dispatcher():
    """The dispatcher of this rank, built at the first SP forward (collective over the TP ranks)."""
    if _ST["disp"] is None:
        from .dispatcher import Dispatcher
        group = _group()
        _ST["engine"].group = group
        _ST["disp"] = Dispatcher(_ST["plan"], _ST["workload"], group=group, methods=_ST["methods"],
                                 deo_bwd="stock", engine=_ST["engine"])    # inference: no backward context
    return _ST["disp"]


def _call(layer, site, op, x):
    """One call site in the SP form: the plan's choice, or the serial path with the layer's own GEMM."""
    disp = _dispatcher()
    x, w = x.detach(), _w_nk(layer).detach()
    if disp.peek(site, op, x, w)[0] == TRITON_DIST and w.stride(1) != 1:
        w = _k_contiguous(layer, w)
    return disp.run(site, op, x, w, gemm=lambda a: layer.quant_method.apply(layer, a, None))


def _gather(x_sh, M):
    import torch
    import torch.distributed as dist
    full = torch.empty(M, x_sh.shape[1], dtype=x_sh.dtype, device=x_sh.device)
    dist.all_gather_into_tensor(full, x_sh.contiguous(), group=_group())
    return full


def _layer_sp(layer, positions, h_sh, res, M):
    attn = layer.self_attn
    if res is None:
        res = h_sh
        x = layer.input_layernorm(h_sh)
    else:
        x, res = layer.input_layernorm(h_sh, res)
    qkv = _call(attn.qkv_proj, "qkv", "ag", x)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q, k = attn.rotary_emb(positions, q, k)
    o = attn.attn(q, k, v)
    o_sh = _call(attn.o_proj, "proj", "rs", o)
    x, res = layer.post_attention_layernorm(o_sh, res)
    gu = _call(layer.mlp.gate_up_proj, "fc1", "ag", x)
    a = layer.mlp.act_fn(gu)
    return _call(layer.mlp.down_proj, "fc2", "rs", a), res


def _sp_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    st = _ST
    src = inputs_embeds if inputs_embeds is not None else input_ids
    M = src.shape[0]
    R = st["world"]
    if (M not in st["buckets"] or M % R or len(self.aux_hidden_state_layers) > 0
            or intermediate_tensors is not None):
        st["n_stock"] += 1
        return st["orig"](self, input_ids, positions, intermediate_tensors, inputs_embeds)
    from vllm.distributed import get_pp_group
    if get_pp_group().world_size != 1:
        raise NotImplementedError("the OPRS vLLM adapter runs one pipeline stage (TP only)")
    disp = _dispatcher()
    t0 = time.perf_counter()
    mp, r = M // R, st["rank"]
    hidden = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
    h_sh, res = hidden[r * mp:(r + 1) * mp], None
    for layer in self.layers[self.start_layer:self.end_layer]:
        h_sh, res = _layer_sp(layer, positions, h_sh, res, M)
    h_sh, _ = self.norm(h_sh, res)
    full = _gather(h_sh, M)
    st["n_sp"] += 1
    st["sp_s"] += time.perf_counter() - t0
    if r == 0 and M not in st["printed"]:
        st["printed"].add(M)
        plan = disp.plan
        taken = ", ".join(f"{site}:{choice}" for (site, _, m, _, _), choice in sorted(plan.entries.items())
                          if m == M and choice != "serial")
        log(f"OPRS_VLLM first SP forward M={M} plan={plan.digest[:12]} backend={plan.backend} "
            f"taken={{{taken or 'none'}}}")
    return full


def stats():
    """Figures of this rank for the entry script: forward-only runs of the dispatcher, the SP / stock
    forward counters and the dispatcher's plan load / context preparation times."""
    disp = _ST["disp"]
    out = {"disp": disp, "runs": {}, "sp_forwards": _ST["n_sp"], "stock_forwards": _ST["n_stock"]}
    if disp is not None:
        out.update(runs=dict(disp.runs), load_ms=disp.load_ms, prepare_ms=disp.prepare_ms,
                   plan=disp.plan.digest, backend=disp.plan.backend)
    return out


def stats_line():
    disp = _ST["disp"]
    ready = (f"plan={disp.plan.digest[:12]} backend={disp.plan.backend} contexts={len(disp.exchanges)} "
             f"load_ms={disp.load_ms:.1f} prepare_ms={disp.prepare_ms:.1f}" if disp is not None else
             "dispatcher not built (no SP forward)")
    return (f"[oprs] OPRS_VLLM rank={_ST['rank']} tp={_ST['world']} sp_forwards={_ST['n_sp']} "
            f"stock_forwards={_ST['n_stock']} sp_host_s={_ST['sp_s']:.2f} k_contiguous_copies={len(_ST['kc'])} "
            f"{ready}")


def report():
    """Hits, fallbacks and forward-only runs of this rank (call before the engine is torn down)."""
    if not _ST["installed"]:
        return
    if _ST["disp"] is not None:
        _ST["disp"].print_report()
    print(stats_line(), flush=True)


def install(mpi, comm, rank, world, host, plan_path, workload, *, methods=None, buckets=(16384,)):
    """Entry-script prelude: after pydushmem mpi_init + init_dcu (the heap is reserved there) and
    before LLM(). `methods` = overlap methods this process may run (default all: deo and both
    external backends); `buckets` = token counts that take the SP form."""
    from vllm.model_executor.models.llama import LlamaModel

    from .backends import METHODS
    from .engine import EngineWorld

    if _ST["installed"]:
        raise RuntimeError("the OPRS vLLM adapter is already installed")
    enabled = tuple(METHODS if methods is None else methods)
    if set(enabled) - set(METHODS):
        raise ValueError(f"unknown methods {sorted(set(enabled) - set(METHODS))}; choose from {METHODS}")
    if int(world) < 1 or int(rank) < 0 or int(rank) >= int(world):
        raise ValueError(f"bad rank/world: {rank}/{world}")
    _ST.update(installed=True, rank=int(rank), world=int(world), buckets={int(b) for b in buckets},
               plan=str(plan_path), workload=workload, methods=enabled,
               engine=EngineWorld(mpi, comm, int(rank), int(world), host))   # idempotent: the same world
    _ST["orig"] = LlamaModel.forward
    LlamaModel.forward = _sp_forward
    if _ST["rank"] == 0:
        log(f"OPRS_VLLM installed plan={plan_path} workload={workload} methods={sorted(enabled)} "
            f"buckets={sorted(_ST['buckets'])} tp={_ST['world']}")
