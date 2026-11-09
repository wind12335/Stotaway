"""MGD t21 BRS: route the vehicle's BIG reduce-scatter class (536.9MB-in
dgrad RS, ~170 calls/step, 75% of RS bytes) through rs_relay (neighbor-
pipelined ring, no all-rank rendezvous -- the skew tax killer).

Interception = the mgd_rs_attr-proven LIVE points: rebind
`dist_reduce_scatter_func` on megatron.core.tensor_parallel.{layers,
mappings} (vocab skipped v1 -- tiny).  Eligibility: bf16, 2D contiguous,
input bytes >= MGD_BRS_MIN_MB (default 256MB -> only the dominant class;
everything else falls back to stock verbatim).

Bootstrap mirrors mgd_tp_patch: eager at train() entry via MPI_Comm_split
over the TP group (lazy split deadlocks across ranks).

Env: MGD_BRS=1 [MGD_BRS_MIN_MB=256] [MGD_BRS_DRY=1 (count only)].
"""
import ctypes
import os

import torch

_MIN_MB = float(os.environ.get("MGD_BRS_MIN_MB", "256"))
_FORM = os.environ.get("MGD_BRS_FORM", "ring")
_DEFER = os.environ.get("MGD_BRS_DEFER", "") == "1"     # ring (v3) | a2a (v4)
# t21 R12w v22: pipeline depth K -- K rings in flight per ctx (K-inflated
# heap/ticket/ack planes, slot-owned lanes+done events, ack fence = K ops
# back).  Default 1 = the v3 single-slot chain.
_SLOTS = int(os.environ.get("MGD_BRS_SLOTS", "1"))
# t21 R12ab: per-callsite K (split-K).  In-vehicle ring inflates ~9ms vs
# 4.57 rig floor; part of it is K=2 self-contention during the layer
# windows where the exposure holes live.  Layers/mappings K=1, DDO bucket
# chain (fires back-to-back, nothing to overlap but each other) K=2.
_SLOTS_DDO = int(os.environ.get("MGD_BRS_SLOTS_DDO", str(_SLOTS)))
_CELLS = int(os.environ.get("MGD_BRS_CELLS", "1"))   # v24 link-cell depth
_SUBS = int(os.environ.get("MGD_BRS_SUBS", "1"))    # v25 sub-chunk pipeline
_DUAL = int(os.environ.get("MGD_BRS_DUAL", "0"))    # v27 dual-lane ring
_DRY = os.environ.get("MGD_BRS_DRY", "") == "1"
# t21 R12n: comma list of module shortnames to relay (default all three) --
# subclass discriminator for the RCCL AG|RS fusion adjacency.
_MODS = set(filter(None, os.environ.get("MGD_BRS_MODS", "").split(",")))
# t21 R12s probe: is RCCL's adjacent-op merge broken by (ev) the relay
# fork bookkeeping (hipEventRecord on q0) or (cp) an engine memcpyAsync
# on q0 -- with NOTHING else changed (stock fallback, zero engine work)?
# Verdict via hp AG-kernel count: stock ~71 fused; BRS full relay 380.
_PROBE = os.environ.get("MGD_BRSPROBE", "")
_probe_state = {}


def _probe_bookkeeping():
    """Lazy per-rank probe resources (import-time CUDA init is unsafe)."""
    if not _probe_state:
        _probe_state["lane"] = torch.cuda.Stream()
        _probe_state["ev"] = None
        _probe_state["buf"] = torch.zeros(8, dtype=torch.int32, device="cuda")
    import ctypes as _ct
    from stowaway.kernels.common_ops import hip as _hip, \
        memcpy_async as _mc
    h = _hip()
    cur = torch.cuda.current_stream()
    cur_p = _ct.c_void_p(cur.cuda_stream)
    if _PROBE == "ev":
        ev = _ct.c_void_p()
        h.hipEventCreateWithFlags(_ct.byref(ev), 0x2)
        h.hipEventRecord(ev, cur_p)
        h.hipStreamWaitEvent(_ct.c_void_p(_probe_state["lane"].cuda_stream),
                             ev, 0)
    elif _PROBE == "cp":
        # one tiny engine DtoD ON q0 (dst != src), the shape of a relay
        # q0 footprint that is a copy instead of an event
        b = _probe_state["buf"]
        _mc(b.data_ptr() + 8, b.data_ptr(), 4, cur.cuda_stream)

_engine = {}
_ctxs = {}
_stats = {"eligible": 0, "relayed": 0, "fallback": 0, "n_ctx": 0,
          "miss_groups": [],
          "dry_shapes": __import__("collections").Counter(),
          "by_mod": __import__("collections").Counter()}
_orig = {}

# S75 per-op timing (MGD_BRS_TIMING=1): caller/compute-stream events ONLY
# (S47 law), read out at flush (events long fired -- no sync poisoning).
# Sync-mode op wall = enqueue->fence-complete, i.e. the tax-bearing span.
_BRT = os.environ.get("MGD_BRS_TIMING", "") == "1"
_brs_pending = []      # (key, host_ms, ev_s, ev_e)
_brs_oplog = {}        # key -> [host_ms...], [wall_ms...]


def _brs_flush(final=False):
    for key, host_ms, ev_s, ev_e in _brs_pending:
        try:
            w = ev_s.elapsed_time(ev_e)
        except Exception:          # not-both-completed race: skip entry
            continue
        h, wls = _brs_oplog.setdefault(key, ([], []))
        h.append(host_ms); wls.append(w)
    _brs_pending.clear()
    if not _brs_oplog:
        return
    tag = "FINAL" if final else "flush"
    for key, (hs, ws) in sorted(_brs_oplog.items(),
                                key=lambda kv: -len(kv[1][1])):
        ws2 = sorted(ws)
        print(f"[mgd-brs-t] {tag} cls={key} n={len(ws2)} "
              f"wall med {ws2[len(ws2)//2]:.2f} p90 {ws2[int(len(ws2)*0.9)]:.2f} "
              f"max {ws2[-1]:.2f} ms | host med "
              f"{sorted(hs)[len(hs)//2]:.2f} ms", flush=True)
    if final:
        _brs_oplog.clear()


def _bootstrap_all():
    """Engine world over the TP subgroup (mgd_tp_patch pattern verbatim)."""
    import pydushmem.host as pHost
    from megatron.core import parallel_state as mpu
    from stowaway.kernels.common_ops import init_dcu

    mpi, world = pHost.mpi_init()
    group = mpu.get_tensor_model_parallel_group()
    granks = torch.distributed.get_process_group_ranks(group)
    color, key = min(granks), torch.distributed.get_rank()
    mpi.MPI_Comm_split.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_void_p)]
    mpi.MPI_Comm_split.restype = ctypes.c_int
    newcomm = ctypes.c_void_p(0)
    rc = mpi.MPI_Comm_split(ctypes.c_void_p(world), color, key,
                            ctypes.byref(newcomm))
    assert rc == 0 and newcomm.value
    sub = newcomm.value
    r = pHost.mpi_comm_rank(mpi, sub)
    n = pHost.mpi_comm_size(mpi, sub)
    assert n == group.size() and r == granks.index(key)
    init_dcu(mpi, sub, r, n)
    _engine.update({"mpi": mpi, "comm": sub, "rank": r, "size": n,
                    "granks": tuple(granks)})
    if r == 0:
        print(f"[mgd-brs] engine world up: tp_size={n}", flush=True)


def _get_ctx(numel, dtype, slots=None, cells=None, subs=None, dual=None):
    # v24: C=2 doubles the heap -- DDO ctxs keep cells=1 (memory budget;
    # the K=2 bucket-chain overlap is their win, R12ac split-K law).
    # v25: S sub-slices the shard -- heap-NEUTRAL (cells hold whole
    # shards), but the ticket-value space per op grows xS; non-DDO only.
    cells = _CELLS if cells is None else cells
    subs = _SUBS if subs is None else subs
    dual = bool(_DUAL) if dual is None else bool(dual)
    key = (numel, dtype, slots if slots is not None else _SLOTS, cells,
           subs, dual)
    if _FORM == "dx":
        key = ("dx", numel, dtype, slots if slots is not None else _SLOTS)
        if key not in _ctxs:
            from stowaway.kernels.gemm_rs_dx import DXRSContext
            assert _engine, "engine not bootstrapped (train hook missing)"
            _ctxs[key] = DXRSContext(numel, dtype, _engine["rank"],
                                     _engine["size"], k_slots=key[3])
            _stats["n_ctx"] = len(_ctxs)
            if _engine["rank"] == 0:
                c = _ctxs[key]
                print(f"[mgd-brs] dx ctx numel={numel} K={c.K} P={c.P} heap "
                      f"{c.P * numel * 2 / 2**20:.0f}MB", flush=True)
        return _ctxs[key]
    if key not in _ctxs:
        import stowaway.kernels.rs_relay as rr
        assert _engine, "engine not bootstrapped (train hook missing)"
        _ctxs[key] = rr.create_rs_relay_context(
            numel, dtype, _engine["rank"], _engine["size"],
            _engine["mpi"], _engine["comm"], k_slots=key[2], cells=key[3],
            subs=key[4], dual=key[5])
        _stats["n_ctx"] = len(_ctxs)
        if _engine["rank"] == 0:
            print(f"[mgd-brs] ctx {key} K={_SLOTS} C={_CELLS} (heap "
                  f"{_engine['size'] * _SLOTS * _CELLS * (numel // _engine['size']) * 2 / 2**20:.0f}MB)",
                  flush=True)
    return _ctxs[key]


class _Skip(Exception):
    """Module filtered out (MGD_BRS_MODS): pass through to stock verbatim
    WITHOUT counting a fallback (t21 R12n subclass discriminator)."""


class _Handle:
    def wait(self):
        pass


# per-callsite deferred-fence registries (t21 R12m).  megatron keeps only
# the LAST bucket handle per group (param_and_grad_buffer.py:660 reassigns
# grad_reduce_handle in the loop; stock is safe because the coalescing
# manager batches the group into one collective) -- with independent ring
# ops, waiting the last alone leaves earlier buckets' rings in flight.
# wait() on any handle fences EVERY outstanding done-event of its class
# (over-wait is safe: a later record on the same ctx implies the earlier
# op completed -- same lane stream order).  SP and DDO stay in separate
# lists so the layers.py:454 consumer never waits behind DDO rings.
_DEFERRED = {"layers": [], "mappings": [], "param_and_grad_buffer": []}


class _HandleDeferredAll:
    """Fence-all-outstanding deferred rings of one call-site class."""

    def __init__(self, cls):
        self._cls = cls

    def wait(self):
        import ctypes as _ct
        from stowaway.kernels.common_ops import hip as _hip
        h = _hip()
        for ev in _DEFERRED[self._cls]:
            rc = h.hipStreamWaitEvent(
                _ct.c_void_p(torch.cuda.current_stream().cuda_stream), ev, 0)
            assert rc == 0
        _DEFERRED[self._cls] = []

    def is_completed(self):
        return False


def _make_wrapper(modname):
    # SIGNATURE-TRANSPARENT: the call sites run under torch.compile/dynamo
    # (t21 R12e) -- any visible deviation (extra op= kwarg, non-None sync
    # return) risks guard breaks / decomposed collective paths.  Relay
    # path returns None exactly like the stock sync call; fallback passes
    # ONLY the kwargs the caller passed.
    def _brs(output, input, op=None, group=None, async_op=False):
        try:
            if _PROBE and _eligible(input, output, op, group):
                _probe_bookkeeping()
                raise _Skip()
            if _MODS and modname.rsplit(".", 1)[-1] not in _MODS:
                raise _Skip()
            if _eligible(input, output, op, group):
                _stats["eligible"] += 1
                _stats["by_mod"][modname.rsplit(".", 1)[-1]] += 1
                if _DRY:
                    from collections import Counter
                    _stats["dry_shapes"][input.numel()] += 1
                    _stats["by_mod"][modname.rsplit(".", 1)[-1]] += 1
                if not _DRY:
                    _cls = modname.rsplit(".", 1)[-1]
                    _ddo = _cls == "param_and_grad_buffer"
                    ctx = _get_ctx(
                        input.numel(), input.dtype,
                        _SLOTS_DDO if _ddo else _SLOTS,
                        cells=1 if _ddo else None,
                        subs=1 if _ddo else None,
                        dual=False if _ddo else None)
                    import stowaway.kernels.rs_relay as rr
                    if _FORM == "dx":
                        from stowaway.kernels.gemm_rs_dx import dxv_op
                        _avg = op is torch.distributed.ReduceOp.AVG
                        _dfc = bool(async_op and _DEFER)
                        ret = dxv_op(input, output, ctx, defer_fence=_dfc,
                                     out_scale=(1.0 / _engine["size"]) if _avg
                                     else None)
                        _stats["relayed"] += 1
                        if async_op and _DEFER:
                            _DEFERRED[modname.rsplit(".", 1)[-1]].append(ret._ev)
                            return _HandleDeferredAll(modname.rsplit(".", 1)[-1])
                        return ret if async_op else None
                    if _FORM in ("ring", "graph"):
                        # MGD_BRS_DEFER=1: async callers fence at consume
                        # (handle.wait), letting megatron's fire->wgrad->wait
                        # window overlap the ring (t21 R12k)
                        # AVG == SUM then exact /R (power of two, t21 R12m)
                        _avg = op is torch.distributed.ReduceOp.AVG
                        _op = (rr.rs_relay_graph_op if _FORM == "graph"
                               else rr.rs_relay_op)
                        _dfc = bool(async_op and _DEFER)
                        if _BRT and not _dfc:      # sync span = tax-bearing
                            if len(_brs_pending) >= 64:
                                try:                # instrument never poisons
                                    _brs_flush()
                                except Exception:   # noqa: BLE001
                                    _brs_pending.clear()
                            import time as _t
                            _t0 = _t.perf_counter()
                            _ev_s = torch.cuda.Event(enable_timing=True)
                            _ev_e = torch.cuda.Event(enable_timing=True)
                            _ev_s.record()
                        ret = _op(
                            input, output, ctx,
                            defer_fence=_dfc,
                            out_scale=(1.0 / _engine["size"]) if _avg
                            else None)
                        if _BRT and not _dfc:
                            _ev_e.record()
                            _brs_pending.append(
                                ((modname.rsplit(".", 1)[-1],
                                  input.numel()),
                                 (_t.perf_counter() - _t0) * 1e3,
                                 _ev_s, _ev_e))
                        _stats["relayed"] += 1
                        if async_op and _DEFER:
                            _DEFERRED[modname.rsplit(".", 1)[-1]].append(
                                ret._ev)
                            return _HandleDeferredAll(
                                modname.rsplit(".", 1)[-1])
                        return ret if async_op else None
                    rr.rs_a2a_op(input, output, ctx)
                    _stats["relayed"] += 1
                    # torch semantics: async callers get a handle, sync None
                    return _Handle() if async_op else None
        except _Skip:
            pass
        except Exception as e:                                  # noqa: BLE001
            print(f"[mgd-brs] FALLBACK ({type(e).__name__}: {e})",
                  flush=True)
            if not _stats.get("_fb_tb"):
                import traceback
                traceback.print_exc()
                _stats["_fb_tb"] = 1
        _stats["fallback"] += 1
        kw = {}
        if op is not None:
            kw["op"] = op
        if group is not None:
            kw["group"] = group
        if async_op:
            kw["async_op"] = True
        return _orig[modname](output, input, **kw)
    return _brs


def _eligible(input, output, op, group):
    # FLAT: megatron passes 3D [S,B,H] activations; the ring only moves
    # contiguous bytes.  Threshold on INPUT bytes (the 536.9MB class).
    if input.dtype is not torch.bfloat16:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    n = input.numel()
    R = _engine.get("size", 0)
    if R < 2 or n % R or (n // R) % 2048 or n * 2 < _MIN_MB * 1e6:
        return False
    if output.numel() != n // R:
        return False
    if op is not None and op not in (torch.distributed.ReduceOp.SUM,
                                     torch.distributed.ReduceOp.AVG):
        return False
    if group is None:
        return False
    try:
        granks = torch.distributed.get_process_group_ranks(group)
    except Exception:                                           # noqa: BLE001
        return False
    if tuple(granks) != _engine.get("granks"):
        if _DRY and len(_stats["miss_groups"]) < 8:
            _stats["miss_groups"].append((tuple(granks), input.numel()))
        return False
    return True


def install():
    """Rebind the LIVE dist_reduce_scatter_func attrs (mgd_rs_attr-proven
    points).  Deferred-safe: caller may import us before vocab exists."""
    import sys
    bound = []
    for mn in ("megatron.core.tensor_parallel.layers",
               "megatron.core.tensor_parallel.mappings",
               # t21 R12m: DDO grad-bucket RS (param_and_grad_buffer.py:660,
               # async under overlap_grad_reduce, op=AVG, out ALIASES the
               # input's rank-shard view) -- the ReduceScatter_Simple_b16
               # 430x8.3ms/step whale the first two bindings never touched.
               "megatron.core.distributed.param_and_grad_buffer"):
        m = sys.modules.get(mn)
        if m is None or not hasattr(m, "dist_reduce_scatter_func"):
            continue
        f = getattr(m, "dist_reduce_scatter_func")
        if getattr(f, "__name__", "") == "_brs":
            continue
        _orig[mn] = f
        setattr(m, "dist_reduce_scatter_func", _make_wrapper(mn))
        bound.append(mn.rsplit(".", 1)[-1])
    if bound:
        print(f"[mgd-brs] bound: {bound} MIN_MB={_MIN_MB} DRY={_DRY}",
              flush=True)
        return True
    return False


def _wrap_train():
    """Eager bootstrap at train() entry + early stats after (mgd_tp law:
    rank0 teardown may die before atexit)."""
    import megatron.training.training as _mt
    _orig_train = _mt.train

    def train(*a, **kw):
        if not _engine:
            # retry: param_and_grad_buffer (DDO, R12m) imports after the
            # prelude's install() pass -- modules list re-checked here.
            install()
            _bootstrap_all()
        r = _orig_train(*a, **kw)
        if _BRT:
            _brs_flush(final=True)
        if _engine.get("rank", 0) == 0:
            print(f"[mgd-brs] early stats {_stats}", flush=True)
        return r

    _mt.train = train


if os.environ.get("MGD_BRS", "") == "1":
    try:
        ok = install()
        _wrap_train()
        if not ok:
            print("[mgd-brs] install deferred (modules not loaded yet)",
                  flush=True)
    except Exception as e:                                      # noqa: BLE001
        print(f"[mgd-brs] install failed: {e}", flush=True)
