"""opus fork: whole MoE layer host-sync-free (OPUS_MOE_SFL=1, default off).

MoELayer.forward (route -> dispatch -> experts -> combine) is replaced, for eligible layers, by
  router (unchanged, aux loss unchanged) -> top-k ids + weights from its probs
  -> kernels/dcu/moe_sf.py: SM-store dispatch into the expert ranks' symmetric heap (device atomics,
     flags + stream waits, no count exchange, no .tolist())
  -> grouped expert MLP with device-side counts (kernels/dcu/moe_expert_mlp.py, same weights)
  -> SM-store return to the source row by tag -> weighted sum.
Backward mirrors it (same two planes, opposite direction). Routing weights are applied in the combine
instead of after the activation (fc2 is linear: identical up to bf16 rounding; gradients identical).
Eligible: training or eval, EP group == the whole world (one engine world), TP = 1, dropless
(no capacity factor), no shared experts, no padding mask, no latent projection, experts eligible for
the grouped path (opus_moe_gg_patch._eligible). Anything else -> the original forward.
Env: OPUS_MOE_SFL_MAXROWS_X (expert-rank row budget as a multiple of T*K, default 3),
     OPUS_MOE_GG_B (virtual group rows, default 2048).
"""
import os

import torch

_ON = os.environ.get("OPUS_MOE_SFL", "") == "1"
_X = float(os.environ.get("OPUS_MOE_SFL_MAXROWS_X", "3"))
_B = int(os.environ.get("OPUS_MOE_GG_B", "2048"))
_ST = {"orig": None, "boot": False, "ctx": {}, "n_sf": 0, "n_orig": 0, "why": {}}


def _boot():
    if _ST["boot"]:
        return
    import pydushmem.host as pHost
    from stowaway.kernels.common_ops import init_dcu
    mpi, world = pHost.mpi_init()
    r, n = pHost.mpi_comm_rank(mpi, world), pHost.mpi_comm_size(mpi, world)
    assert r == torch.distributed.get_rank() and n == torch.distributed.get_world_size()
    init_dcu(mpi, world, r, n)
    _ST.update(boot=True, rank=r, world=n)
    if r == 0:
        print(f"[opus-moe-sfl] engine world up: R={n} maxrows={_X}xTK B={_B}", flush=True)


def _ineligible(self, hidden_states, padding_mask):
    from stowaway.integrations.opus_moe_gg_patch import _eligible
    c = self.config
    if padding_mask is not None:
        return "padding_mask"
    if getattr(self, "use_shared_expert", False) or getattr(c, "moe_latent_size", None):
        return "shared/latent"
    if getattr(c, "moe_expert_capacity_factor", None):
        return "capacity"
    if c.moe_token_dispatcher_type != "alltoall":
        return "dispatcher"
    if hasattr(self, "_inference_token_dispatcher"):
        return "inference dispatcher"
    if self.tp_group.size() != 1 or self.ep_group.size() != torch.distributed.get_world_size():
        return "tp/ep layout"
    if hidden_states.dtype != torch.bfloat16:
        return "dtype"
    if not _eligible(self.experts):
        return "experts"
    return None


def _ctx(T, K, H, E):
    key = (T, K, H, E)
    if key not in _ST["ctx"]:
        from stowaway.kernels.moe_sf import SFMoEContext
        R = _ST["world"]
        _ST["ctx"][key] = SFMoEContext(T, K, H, E, _ST["rank"], R, int(_X * T * K))
    return _ST["ctx"][key]


def _forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
    why = None if intermediate_tensors is not None else _ineligible(self, hidden_states, padding_mask)
    if intermediate_tensors is not None:
        why = "partial cudagraph"
    if why is not None:
        _ST["n_orig"] += 1
        _ST["why"][why] = _ST["why"].get(why, 0) + 1
        return _ST["orig"](self, hidden_states, intermediate_tensors, padding_mask)
    _boot()
    from stowaway.integrations.opus_moe_gg_patch import _tables
    from stowaway.kernels.moe_expert_mlp import expert_linear, get_plan
    from stowaway.kernels.moe_sf import sf_moe
    import torch.nn.functional as F
    shp = hidden_states.shape
    H = shp[-1]
    x = hidden_states.reshape(-1, H)
    T = x.shape[0]
    K = self.config.moe_router_topk
    E = self.config.num_moe_experts
    probs, routing_map = self.route(x.view(shp))
    probs = probs.reshape(T, E)
    w, idx = torch.topk(probs.float() * routing_map.reshape(T, E).float(), K, dim=-1)
    idx = idx.to(torch.int32).reshape(-1).contiguous()
    sf = _ctx(T, K, H, E)
    ex = self.experts
    w1, w2, t1, t2 = _tables(ex)
    El = ex.num_local_experts
    p1 = get_plan(El, H, w1[0].shape[0], sf.maxrows, _B)
    p2 = get_plan(El, w2[0].shape[1], H, sf.maxrows, _B)

    def experts_fn(xp, offs):
        h = expert_linear(xp, offs, w1, p1, t1)
        a, b = torch.chunk(h, 2, dim=-1)
        return expert_linear((F.silu(a) * b).contiguous(), offs, w2, p2, t2)

    out = sf_moe(x.contiguous(), idx, w.contiguous(), sf, experts_fn)
    _ST["n_sf"] += 1
    if _ST["n_sf"] in (1, 1000) and _ST["rank"] == 0:
        print(f"[opus-moe-sfl] {stats()}", flush=True)
    return out.view(shp), None


def install():
    if not _ON or _ST["orig"] is not None:
        return
    from megatron.core.transformer.moe.moe_layer import MoELayer
    _ST["orig"] = MoELayer.forward
    MoELayer.forward = _forward
    print("[opus-moe-sfl] MoELayer.forward -> host-sync-free dispatch + grouped experts + combine", flush=True)


def stats():
    ovf = [int(c.overflow.item()) for c in _ST["ctx"].values()]
    return f"sf_calls={_ST['n_sf']} original={_ST['n_orig']} why={_ST['why']} overflow_rows={ovf}"
