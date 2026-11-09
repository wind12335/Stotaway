"""AG-GEMM patch for Megatron-LM(-das) on DCU: fuse ColumnParallelLinear
(SP gather + GEMM) into the tile-ready PULL allgather-GEMM engine op on
the TENSOR-PARALLEL SUBGROUP. Sibling of mgd_tp_patch (which fuses the
RS side); the two compose via a SHARED engine world (mgd_tp_patch._engine).

Engine form: ag_tr_pred.i1_ag_tr_op with I1_PULL=kernel -- the pull
KERNEL does the cross-card copy AND the ready flag in one kernel (no
engine-copy/kernel-flag race). The core gemm_ag_tr_op (engine copy +
kernel ready flag) is the known-corrupting-under-skew baseline and TEARS
at R=8: measured 14/30 rounds (t21_ag_r8probe.py), ws fresh but c stale
(t21_ag_r8probe3.py); the kernel-pull form is 0/20 clean isolated
(t21_ag_r8probe4.py).

Verify+repair (S16 design, integration-local, dtype-generic -- the bench
form kernel_i1_verify hardcodes fp16): after each op a SPOT kernel flags
suspect tiles, the host reads the flag plane (one sync per op -- the
S16-accepted cost, well under 1% of step time), and flagged tiles are
recomputed from the post-round ws (probe3: ws is always fresh) by a
repair kernel on the caller stream before the layer returns. bf16 tol
2e-2 relative (half-ulp 2^-8).

Bootstrap: reuses mgd_tp_patch's train()-entry hook convention; when
MGD_FUSED=1 the tp patch's hook bootstraps the shared engine world,
otherwise this module's hook does the one and only split.

Dispatcher: m_pr*K*itemsize >= MGD_MIN_MB (same payload law as the RS
patch; MBS=16/seq4096 -> 67MB eligible); MGD_AG_DRY=1 counts eligible
calls without engaging. MGD_AG unset = fully stock.

Usage (env): MGD_AG=1 [MGD_AG_FORM=i1|ring] [MGD_MIN_MB=64]
[MGD_AG_MIN_N=1024] [MGD_AG_DRY=1] [MGD_AG_VERIFY=1] [MGD_AG_VTOL=2e-2]
[MGD_AG_CHUNK=512 MGD_AG_LANES=2] + import after
megatron.core.tensor_parallel is importable.

T21 dispatcher: MGD_AG_MIN_N routes thin-N ops (qkv N_pr=768, measured
+18.2ms/op vs stock at M65536/R=8) back to stock; 0 restores the old
all-eligible behavior. MGD_AG_CHUNK/MGD_AG_LANES pass through to
create_i1_context (transport config, call-site knobs -- the core kernel
files are untouched).

MGD_AG_FORM selects the engine op: "i1" = the paper's tile-ready PULL
(ag_tr_pred, pull kernel + triton consumer); "ring" = v17ag
(gemm_ag_v17ring, NEW FILE): Megatron-SP AG via input row-block ring
relay + per-arrival VENDOR GEMM chunks -- isolated R=8 probe at the
vehicle shape: N768 5.2ms (stock 5.3), N3072 7.1ms (stock 8.5, i1 44.5);
bit-exact vs stock, 0 spot hits.
"""
import os
import time

# engine-form envs must precede the ag_tr_pred import (module-level reads)
os.environ.setdefault("I1_PULL", "kernel")
os.environ.setdefault("I1_VERIFY", "0")       # bench verify is fp16-only
os.environ.setdefault("I1_READY", "engine")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import stowaway.kernels.ag_tr_pred as i1  # noqa: E402
import stowaway.kernels.gemm_ag_v17ring as v17  # noqa: E402
import stowaway.kernels.gemm_ag_v18ring as v18  # noqa: E402
import stowaway.kernels.gemm_ag_v19ring as v19  # noqa: E402
import stowaway.kernels.gemm_ag_v20ring as v20  # noqa: E402
from stowaway.kernels.common_ops import hip  # noqa: E402
from stowaway.integrations import mgd_tp_patch  # noqa: E402

_TIMING2 = os.environ.get("MGD_AG_TIMING", "") == "2"   # stock baseline
_AG_BASELINE = _TIMING2
_ENABLED = os.environ.get("MGD_AG", "") == "1" or _AG_BASELINE
_DRY = os.environ.get("MGD_AG_DRY", "") == "1"
_MIN_MB = float(os.environ.get("MGD_MIN_MB", "64"))
_MIN_N = int(os.environ.get("MGD_AG_MIN_N", "1024"))   # T21 dispatcher gate
_FORM = os.environ.get("MGD_AG_FORM", "i1")  # i1 (paper PULL) | ring (v17ag)
_VERIFY = os.environ.get("MGD_AG_VERIFY", "1") == "1"
# sample-K verify: run the blocking SPOT verify+repair only every Nth
# fused op. The .item() is not just detection -- it is a CROSS-ROUND
# FENCE: every rank's op-(i+1) relay is enqueued only after its op-i
# verify released, so no peer can overwrite my ws slots before my op-i
# flag kernel read them. Sample-K keeps that fence at every verified op
# (all ranks count fused ops identically) while dividing the host-block
# run-ahead loss by K. Unverified ops read no ws -> no added race.
# K=1 (default) = per-op verify, the S16 form.
_VSAMPLE = max(1, int(os.environ.get("MGD_AG_VSAMPLE", "1")))
_vop_count = 0
# 1-op lane backpressure for the ring form (see mgd_tp_patch._BP) --
# becomes REQUIRED once VSAMPLE>1 removes the per-op .item() fence and
# the host can run ahead into a full engine queue.
_BP = os.environ.get("MGD_BP", "0") == "1"
_VTOL = float(os.environ.get("MGD_AG_VTOL", "2e-2"))   # bf16 half-ulp 2^-8
# transport config (create_i1_context call-site knobs, NOT core edits):
# defaults = t21_chunk_sweep winners at the vehicle shape (M65536 K4096 R=8,
# bench/results/t21_chunk_sweep.csv): CR=512/L=2 -> N3072 34.4ms (-27% vs
# auto 47.1), N768 15.7ms, 0 spot hits. 0 = engine auto (old default).
_CHUNK_ROWS = int(os.environ.get("MGD_AG_CHUNK", "512"))
_COMM_LANES = int(os.environ.get("MGD_AG_LANES", "2"))

_stats = {"eligible": 0, "fused": 0, "fallback": 0, "repaired_tiles": 0,
          "verify_syncs": 0, "routed_out_n": 0}
_ag_ctxs = {}          # (M, K, dtype) -> i1 ctx
_flag_buf = None       # int32 [n_tiles] verdict plane (grown on demand)
_repair_ck = None

_TIMING = os.environ.get("MGD_AG_TIMING", "") == "1"
_oplog = []            # (enqueue_ms, total_ms, dev_ms) rows; 64/flush
_ag_wraplog = []       # MGD_WRAP_TIMING=1: full fused-branch host span
_WRAPT = os.environ.get("MGD_WRAP_TIMING", "") == "1"
_ag_prev_ev = None     # (ev_start, ev_end) of the previous op


def _flush2():
    ts = sorted(t for _, t, _ in _oplog)
    es = sorted(e for e, _, _ in _oplog)
    ds = sorted(d for _, _, d in _oplog if d == d)
    print(f"[mgd-ag] stock opwall n={len(_oplog)} med {ts[len(ts)//2]:.1f} "
          f"p90 {ts[int(0.9*len(ts))]:.1f} | dev med "
          f"{ds[len(ds)//2]:.1f} p90 {ds[int(0.9*len(ds))]:.1f} | "
          f"enqueue med {es[len(es)//2]:.1f}", flush=True)
    _oplog.clear()


def _log_op(enq_s, tot_s, ev_s, ev_e):
    """Deferred device-opwall readout: the PREVIOUS op's event pair is
    complete by now (a full op + intervening vehicle work has passed), so
    elapsed_time() costs no host block. Host-clock total includes the
    verify .item() backlog drain -- the event pair does not."""
    global _ag_prev_ev
    dev_ms = float("nan")
    if _ag_prev_ev is not None:
        ps, pe = _ag_prev_ev
        pe.synchronize()                    # fired long ago
        dev_ms = ps.elapsed_time(pe)
    _ag_prev_ev = (ev_s, ev_e) if ev_e is not None else None
    _oplog.append((enq_s * 1e3, tot_s * 1e3, dev_ms))
    if len(_oplog) >= 64:
        es = sorted(e for e, _, _ in _oplog)
        ts = sorted(t for _, t, _ in _oplog)
        ds = sorted(d for _, _, d in _oplog if d == d)
        print(f"[mgd-ag] opwall n=64 med {ts[32]:.1f} p90 {ts[57]:.1f} "
              f"max {ts[-1]:.1f} | dev med "
              f"{ds[len(ds)//2]:.1f} p90 {ds[int(0.9*len(ds))]:.1f} | "
              f"enqueue med {es[32]:.1f} p90 {es[57]:.1f}", flush=True)
        if _ag_wraplog:
            ws = sorted(_ag_wraplog)
            print(f"[mgd-ag] wrap n={len(ws)} med {ws[len(ws)//2]:.1f} "
                  f"p90 {ws[int(0.9*len(ws))]:.1f}", flush=True)
            del _ag_wraplog[:]
        _oplog.clear()


@triton.jit(do_not_specialize=["ws_ptr", "num_n", "M", "N", "K"])
def _kernel_ag_flag(c, ws_ptr, b, flags, M, N, K, num_n, tol,
                    SPOT: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    """Per-tile SPOT spot-verify, dtype-generic (carried by the tensors):
    recompute SPOT elements of c in fp32 from the known-good post-round
    ws and b; store 1 to this tile's flag cell (one CTA per tile, no
    atomics; the host pre-zeroes the plane). Straight-line (no scalar
    `if` -- the hcu triton frontend's make_ttir rejects that form)."""
    pid = tl.program_id(0)
    pid_m = pid // num_n
    pid_n = pid % num_n
    dt = c.dtype.element_ty
    cp = c.to(tl.pointer_type(dt))
    ap = ws_ptr.to(tl.pointer_type(dt))
    bp = b.to(tl.pointer_type(dt))
    off_r = (17, 53, 91, 125)
    off_c = (29, 97, 61, 5)
    bad = 0.0
    for s in tl.static_range(SPOT):
        row = pid_m * BM + off_r[s]
        col = pid_n * BN + off_c[s]
        v = tl.zeros((BK,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            m = kk < K
            a_v = tl.load(ap + row * K + kk, mask=m, other=0.0)
            b_v = tl.load(bp + col * K + kk, mask=m, other=0.0)
            v += a_v.to(tl.float32) * b_v.to(tl.float32)
        acc = tl.sum(v)
        c_v = tl.load(cp + row * N + col).to(tl.float32)
        err = tl.abs(acc - c_v) / tl.maximum(tl.abs(acc), 1.0)
        bad += tl.where(err > tol, 1.0, 0.0)
    tl.store(flags + pid, bad.to(tl.int32))


@triton.jit(do_not_specialize=["ws_ptr", "num_n", "K", "N"])
def _kernel_ag_repair(c, ws_ptr, b, idx, num_n, K, N,
                      BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr):
    """Recompute the flagged output tiles from the post-round ws and b
    (ws is provably fresh -- t21_ag_r8probe3); one CTA per flagged tile,
    launched for exactly the nonzero flag indices."""
    pid = tl.program_id(0)
    tile = tl.load(idx + pid)
    pid_m = tile // num_n
    pid_n = tile % num_n
    dt = c.dtype.element_ty
    cp = c.to(tl.pointer_type(dt))
    ap = ws_ptr.to(tl.pointer_type(dt))
    bp = b.to(tl.pointer_type(dt))
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a_v = tl.load(ap + rm[:, None] * K + kk[None, :])
        b_v = tl.load(bp + rn[:, None] * K + kk[None, :])
        acc = tl.dot(a_v, tl.trans(b_v), acc)
    tl.store(cp + rm[:, None] * N + rn[None, :], acc.to(dt))


@triton.jit(do_not_specialize=["ws_ptr", "num_n", "M", "N", "K"])
def _kernel_ag_verify(c, ws_ptr, b, counter, M, N, K, num_n, tol,
                      SPOT: tl.constexpr, BM: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr):
    """Counter form of the spot-verify (kept for the t21 probe rigs)."""
    pid = tl.program_id(0)
    pid_m = pid // num_n
    pid_n = pid % num_n
    dt = c.dtype.element_ty
    cp = c.to(tl.pointer_type(dt))
    ap = ws_ptr.to(tl.pointer_type(dt))
    bp = b.to(tl.pointer_type(dt))
    off_r = (17, 53, 91, 125)
    off_c = (29, 97, 61, 5)
    bad = 0.0
    for s in tl.static_range(SPOT):
        row = pid_m * BM + off_r[s]
        col = pid_n * BN + off_c[s]
        v = tl.zeros((BK,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            m = kk < K
            a_v = tl.load(ap + row * K + kk, mask=m, other=0.0)
            b_v = tl.load(bp + col * K + kk, mask=m, other=0.0)
            v += a_v.to(tl.float32) * b_v.to(tl.float32)
        acc = tl.sum(v)
        c_v = tl.load(cp + row * N + col).to(tl.float32)
        err = tl.abs(acc - c_v) / tl.maximum(tl.abs(acc), 1.0)
        bad += tl.where(err > tol, 1.0, 0.0)
    tl.atomic_add(counter, bad.to(tl.int32))


def _verify_repair(c, ectx, w, ev_e=None):
    """SPOT-flag -> host verdict (one sync) -> repair flagged tiles on the
    caller stream. c is safe to consume when this returns. ev_e (timing
    only) records on the caller stream right after the flag kernel, i.e.
    before the host block -- the device-opwall end point."""
    global _flag_buf, _repair_ck, _vop_count
    if not _VERIFY:
        if ev_e is not None:
            ev_e.record()
        return
    _vop_count += 1
    if _VSAMPLE > 1 and _vop_count % _VSAMPLE != 0:
        _stats["verify_skips"] = _stats.get("verify_skips", 0) + 1
        if ev_e is not None:
            ev_e.record()
        return
    M, N_pr = c.shape
    num_m, num_n = M // 128, N_pr // 128
    ntile = num_m * num_n
    if _flag_buf is None or _flag_buf.numel() < ntile:
        _flag_buf = torch.zeros(ntile, dtype=torch.int32, device="cuda")
    fb = _flag_buf[:ntile]
    fb.zero_()
    _kernel_ag_flag[(ntile,)](c, ectx.ws_ptr, w, fb, M, N_pr, ectx.K,
                              num_n, _VTOL, SPOT=4, BM=128, BN=128,
                              BK=512, num_warps=4)
    if ev_e is not None:
        ev_e.record()
    ev = torch.cuda.Event()
    ev.record()
    ev.synchronize()                          # the S16-accepted per-op cost
    _stats["verify_syncs"] += 1
    idx = torch.nonzero(fb).flatten()
    if idx.numel() == 0:
        return
    num_n_t = triton.cdiv(N_pr, 128)
    ck = _repair_ck
    if ck is None:
        ck = _kernel_ag_repair.warmup(c, ectx.ws_ptr, w, idx, num_n_t,
                                      ectx.K, N_pr, BM=128, BN=128, BK=64,
                                      grid=(1,), num_warps=8)
        ck._init_handles()
        _repair_ck = ck
    ck[(int(idx.numel()), 1, 1)](c, ectx.ws_ptr, w, idx, num_n_t, ectx.K,
                                 N_pr, 128, 128, 64)
    _stats["repaired_tiles"] += int(idx.numel())
    print(f"[mgd-ag] REPAIRED {int(idx.numel())} tiles "
          f"(total {_stats['repaired_tiles']})", flush=True)


def _get_ag_ctx(M, K, dtype):
    key = (M, K, dtype, _FORM)
    if key not in _ag_ctxs:
        eng = mgd_tp_patch._engine
        assert eng, "engine not bootstrapped (train() hook missing)"
        if _FORM in ("ring", "ring18", "ring19", "ring20"):
            _mk = (v17.create_gemm_ag_v17ring_context if _FORM == "ring"
                   else v19.create_gemm_ag_v19ring_context
                   if _FORM == "ring19"
                   else v20.create_gemm_ag_v20ring_context
                   if _FORM == "ring20"
                   else v18.create_gemm_ag_v18ring_context)
            _ag_ctxs[key] = _mk(
                M, K, dtype, dtype, eng["rank"], eng["size"],
                eng["mpi"], eng["comm"])
            if eng["rank"] == 0:
                print(f"[mgd-ag] v17ring ctx up: M={M} K={K} "
                      f"sub={_ag_ctxs[key].sub} "
                      f"sub_rows={_ag_ctxs[key].sub_rows}", flush=True)
        else:
            _ag_ctxs[key] = i1.create_i1_context(
                M, K, dtype, dtype, eng["rank"], eng["size"],
                eng["mpi"], eng["comm"], chunk_rows=_CHUNK_ROWS,
                comm_lanes=_COMM_LANES)
            if eng["rank"] == 0:
                print(f"[mgd-ag] i1 ctx up: M={M} K={K} "
                      f"chunk_rows={_ag_ctxs[key].ag.chunk_rows} "
                      f"lanes={_ag_ctxs[key].ag.comm_lanes} "
                      f"pull=kernel", flush=True)
    return _ag_ctxs[key]


class _FusedColumnParallelAG(torch.autograd.Function):
    """out = gather_r(a_r) @ w  (full [M, N_pr] on every rank, SP order)."""

    @staticmethod
    def forward(fc, a, w, group, m_pr, N_pr):
        ictx = _get_ag_ctx(m_pr * group.size(), a.shape[1], a.dtype)
        c = torch.empty(m_pr * group.size(), N_pr, dtype=a.dtype,
                        device=a.device)
        # BP waits the PREVIOUS op's compute-stream completion (torch
        # Event recorded after the op call) -- NOT ctx._ev_join: the join
        # rides the lane relay tail and fires only after the whole ring
        # round has reached us, so waiting it gates the next round's
        # relay on this round's arrival GEMMs (AGW: enqueue med 58ms,
        # p90 157ms host stall). Above t0: keep the wait out of enqueue.
        if _BP and _FORM in ("ring", "ring18"):
            pev = getattr(ictx, "_ev_done", None)
            if pev is not None and not pev.query():
                pev.synchronize()
        t0 = time.perf_counter() if _TIMING else 0.0
        ev_s = torch.cuda.Event(enable_timing=True) if _TIMING else None
        ev_e = torch.cuda.Event(enable_timing=True) if _TIMING else None
        if _TIMING:
            ev_s.record()
        if _FORM in ("ring", "ring18", "ring19", "ring20"):
            (v17.gemm_ag_v17ring_op if _FORM == "ring"
             else v19.gemm_ag_v19ring_op if _FORM == "ring19"
             else v20.gemm_ag_v20ring_op if _FORM == "ring20"
             else v18.gemm_ag_v18ring_op)(a, w, c, ictx)
            dev_ev = getattr(ictx, "_ev_done", None)
            if dev_ev is None:
                dev_ev = ictx._ev_done = torch.cuda.Event()
            dev_ev.record()          # cur stream: fires when c is ready
        else:
            i1.i1_ag_tr_op(a, w, c, ictx, with_observer=False)
        t1 = time.perf_counter() if _TIMING else 0.0
        _verify_repair(c, ictx if _FORM in
                  ("ring", "ring18", "ring19", "ring20")
                  else ictx.ag, w,
                       ev_e=ev_e)
        if _TIMING:
            _log_op(t1 - t0, time.perf_counter() - t0, ev_s, ev_e)
        fc.save_for_backward(a, w)
        fc.group = group
        fc.m_pr = m_pr
        return c

    @staticmethod
    def backward(fc, grad_out):
        a, w = fc.saved_tensors
        group = fc.group
        R = group.size()
        K = a.shape[1]
        g = grad_out.contiguous()                     # [M, N_pr]
        M = g.shape[0]
        # dX = row-block of g @ W  (bwd of gather = reduce-scatter dim 0)
        dg = torch.matmul(g, w)                       # [M, K]
        dx = torch.empty(fc.m_pr, K, dtype=g.dtype, device=g.device)
        torch.distributed.reduce_scatter_tensor(dx, dg, group=group)
        # dW = g^T @ A_full (re-gather A with stock RCCL in backward)
        full = torch.empty(M, K, dtype=a.dtype, device=a.device)
        torch.distributed.all_gather_into_tensor(full, a, group=group)
        dw = torch.matmul(g.transpose(0, 1), full)    # [N_pr, K]
        return dx, dw, None, None, None


def _eligible(self, input_):
    if not _ENABLED:
        return False
    if not self.sequence_parallel or self.explicit_expert_comm:
        return False
    if self.gather_output or self.gradient_accumulation_fusion \
            or self.bias is not None:
        return False
    if input_.dim() != 3:                            # [s/R, b, h]
        return False
    if input_.dtype not in (torch.float16, torch.bfloat16):
        return False
    m_pr = input_.shape[0] * input_.shape[1]
    K = input_.shape[2]
    N_pr = self.weight.shape[0]
    R = self.tp_group.size()
    if R < 2 or m_pr % 64 or (m_pr * R) % 128 or N_pr % 128 or K % 16:
        return False
    if N_pr < _MIN_N:
        # T21 dispatcher: thin-N (qkv N_pr=768) measured +18.2ms/op vs stock
        # at the vehicle shape -- out of the engine's win region (S59
        # payload law); route to stock. Fat-N gate_up (3072) stays.
        _stats["routed_out_n"] += 1
        return False
    payload_mb = m_pr * K * input_.element_size() / 1e6
    return payload_mb >= _MIN_MB


def install():
    """Patch ColumnParallelLinear.forward (class-level, env-gated)."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    orig = ColumnParallelLinear.forward

    def patched(self, input_, weight=None, runtime_gather_output=None):
        if weight is None and runtime_gather_output is None:
            if _eligible(self, input_):
                _stats["eligible"] += 1
                if _TIMING2 and not _DRY:
                    # baseline arm: time the stock AG (RCCL gather + GEMM)
                    # on the same vehicle op set for a like-for-like
                    # device-opwall comparison with the fused forms.
                    t0 = time.perf_counter()
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    r = orig(self, input_, weight, runtime_gather_output)
                    ev_e.record()
                    global _ag_prev_ev
                    host_ms = (time.perf_counter() - t0) * 1e3
                    if _ag_prev_ev is not None:
                        ps, pe = _ag_prev_ev
                        pe.synchronize()
                        _oplog.append((host_ms, host_ms, ps.elapsed_time(pe)))
                        if len(_oplog) >= 64:
                            _flush2()
                    _ag_prev_ev = (ev_s, ev_e)
                    return r
                if not _DRY:
                    tw0 = time.perf_counter() if _WRAPT else 0.0
                    s, b, h = input_.shape
                    R = self.tp_group.size()
                    m_pr = s * b                       # (s/R)*b rows
                    a2 = input_.reshape(m_pr, h)       # copy only if torn
                    N_pr = self.weight.shape[0]
                    out = _FusedColumnParallelAG.apply(
                        a2, self.weight, self.tp_group, m_pr, N_pr)
                    if _WRAPT:
                        _ag_wraplog.append(
                            (time.perf_counter() - tw0) * 1e3)
                    _stats["fused"] += 1
                    return out.view(s * R, b, N_pr), None
        return orig(self, input_, weight, runtime_gather_output)

    ColumnParallelLinear.forward = patched

    # eager engine bootstrap at train() entry, SHARED with mgd_tp_patch:
    # when MGD_FUSED=1 its (inner, installed first) hook always splits;
    # otherwise this hook does the one and only split.
    import megatron.training.training as _mt
    _orig_train = _mt.train

    def train(*args, **kwargs):
        import sys as _s
        if os.environ.get("MGD_FUSED", "") != "1" \
                and not mgd_tp_patch._engine:
            mgd_tp_patch._bootstrap_all()
        try:
            r = _orig_train(*args, **kwargs)
        except BaseException:
            _s._mgd_exit_code = 1
            raise
        _s._mgd_exit_code = 0
        if os.environ.get("MGD_STATS_EARLY") \
                and mgd_tp_patch._engine.get("rank", 0) == 0:
            print(f"[mgd-ag] early stats {stats()}", flush=True)
        return r

    _mt.train = train
    print(f"[mgd-ag] ColumnParallelLinear.forward patched "
          f"(FORM={_FORM} MIN_MB={_MIN_MB} MIN_N={_MIN_N} DRY={_DRY} "
          f"VERIFY={_VERIFY} chunk={_CHUNK_ROWS} lanes={_COMM_LANES})",
          flush=True)


def stats():
    return dict(_stats, n_ctx=len(_ag_ctxs))


def _dump_stats_at_exit():
    if mgd_tp_patch._engine.get("rank", 0) == 0:
        print(f"[mgd-ag] exit stats {stats()}", flush=True)
    # no os._exit here: mgd_tp_patch's atexit (registered FIRST when it is
    # imported before us, hence runs LAST) owns the teardown-order exit


import atexit
atexit.register(_dump_stats_at_exit)


if os.environ.get("MGD_AG", "") == "1" or \
        os.environ.get("MGD_AG_DRY", "") == "1" or \
        os.environ.get("MGD_AG_TIMING", "") == "2":
    try:
        install()
    except Exception as e:                                  # noqa: BLE001
        print(f"[mgd-ag] install deferred: {e}", flush=True)
