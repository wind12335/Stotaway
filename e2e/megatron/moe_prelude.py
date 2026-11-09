"""Opus MoE vehicle prelude: per-phase timing of Megatron's alltoall token
dispatcher (OPUS_MOE_TIMING=1), then runs e2e/megatron/prelude.py unchanged.

Phases (per MoE layer call, forward and backward-recompute alike):
  pre   dispatch_preprocess   (routing metadata, count a2a, D2H sync, permute 1)
  a2a1  token_dispatch        (EP all-to-all of tokens + probs)
  post  dispatch_postprocess  (sort chunks by local expert, D2H sync)
  exp   experts               (SequentialMLP over 16 local experts)
  cpre  combine_preprocess    (unsort)
  a2a2  token_combine         (EP all-to-all back)
  cpost combine_postprocess   (unpermute)
Device time = compute-stream events around each phase (exposed time on the
critical stream; deferred readout via event.query(), no syncs added).
Host time = perf_counter around each phase call.  Backward phases are not
hooked (autograd); the dispatcher's backward all-to-alls show up as the
gap between total step time and the hooked forward phases.
Flush every 256 layer calls: per-phase mean device ms / mean host ms.
"""
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PRELUDE = os.path.join(HERE, "prelude.py")

if os.environ.get("OPUS_MOE_SF", "") == "1" or os.environ.get("OPUS_MOE_SFL", "") == "1":
    # S39 order law (as mgd_prelude2 does for its engine arms): the MPI
    # library must be loaded RTLD_GLOBAL before torch / megatron; a late
    # load + MPI_Init segfaults in PMPI_Comm_size (opus_moeveh/sfv2).
    import ctypes
    import pydushmem.host as _pHost
    ctypes.CDLL(_pHost.MPI_LIB, mode=ctypes.RTLD_GLOBAL)

if os.environ.get("OPUS_MOE_SFL", "") == "1":
    # 11-02: whole MoE layer host-sync-free (dispatch + grouped experts + combine)
    from stowaway.integrations import opus_moe_sfl_patch
    opus_moe_sfl_patch.install()

if os.environ.get("OPUS_MOE_GG", "") == "1":
    # 11-01: sync-free dropless grouped expert MLP (installed BEFORE the timing wrap so "exp" times it)
    from stowaway.integrations import opus_moe_gg_patch
    opus_moe_gg_patch.install()

if os.environ.get("OPUS_MOE_TIMING", "") == "1":
    import torch
    from megatron.core.transformer.moe import token_dispatcher as td
    from megatron.core.transformer.moe import moe_layer as ml

    PH = ["pre", "a2a1", "post", "exp", "cpre", "a2a2", "cpost"]
    pend = []                     # (phase, ev_s, ev_e)
    dev = {p: [0.0, 0] for p in PH}
    hst = {p: [0.0, 0] for p in PH}
    calls = [0]
    SKIP = int(os.environ.get("OPUS_MOE_TSKIP", "72"))     # 3 iters x 12 layers x (fwd + recompute)
    FLUSH = int(os.environ.get("OPUS_MOE_TFLUSH", "48"))   # 2 iters
    rank = int(os.environ.get("RANK", "0"))

    def drain(force=False):
        keep = []
        for ph, s, e in pend:
            if force or e.query():
                e.synchronize()
                dev[ph][0] += s.elapsed_time(e)
                dev[ph][1] += 1
            else:
                keep.append((ph, s, e))
        pend[:] = keep

    def flush():
        drain()
        if rank in (0, 7):
            parts = " ".join(f"{p}={dev[p][0] / max(1, dev[p][1]):.3f}/{hst[p][0] / max(1, hst[p][1]):.3f}"
                             for p in PH)
            tot_d = sum(dev[p][0] / max(1, dev[p][1]) for p in PH)
            tot_h = sum(hst[p][0] / max(1, hst[p][1]) for p in PH)
            print(f"[moe-t] r{rank} calls={calls[0]} per-layer-call dev/host ms: {parts} | "
                  f"sum dev {tot_d:.3f} host {tot_h:.3f}", flush=True)
        for p in PH:
            dev[p] = [0.0, 0]
            hst[p] = [0.0, 0]

    def wrap(obj, name, ph):
        orig = getattr(obj, name)

        def f(*a, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            s.record()
            r = orig(*a, **kw)
            e.record()
            hst[ph][0] += (time.perf_counter() - t0) * 1e3
            hst[ph][1] += 1
            pend.append((ph, s, e))
            if ph == "cpost":
                calls[0] += 1
                if len(pend) > 512:
                    drain()
                if calls[0] == SKIP:            # drop warmup iterations
                    drain(force=True)
                    for p_ in PH:
                        dev[p_] = [0.0, 0]
                        hst[p_] = [0.0, 0]
                elif calls[0] > SKIP and (calls[0] - SKIP) % FLUSH == 0:
                    flush()
            return r
        setattr(obj, name, f)

    D = td.MoEAlltoAllTokenDispatcher
    for nm, ph in (("dispatch_preprocess", "pre"), ("token_dispatch", "a2a1"),
                   ("dispatch_postprocess", "post"), ("combine_preprocess", "cpre"),
                   ("token_combine", "a2a2"), ("combine_postprocess", "cpost")):
        wrap(D, nm, ph)
    from megatron.core.transformer.moe import experts as ex
    wrap(ex.SequentialMLP, "forward", "exp")
    print(f"[moe-t] alltoall dispatcher phase timing installed (rank {rank})", flush=True)

if os.environ.get("OPUS_MOE_SF", "") == "1":
    from stowaway.integrations import opus_moe_a2a_patch
    opus_moe_a2a_patch.install()

sys.argv = [PRELUDE] + sys.argv[1:]
runpy.run_path(PRELUDE, run_name="__main__")
