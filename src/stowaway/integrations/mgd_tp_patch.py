"""TP-group comm patch for Megatron-LM(-das) on DCU: fuse RowParallelLinear
(GEMM + sequence-parallel reduce-scatter) into the v9ring engine op on the
TENSOR-PARALLEL SUBGROUP (engine world = the TP group, not COMM_WORLD).

Design:
  - bootstrap: lazily on the first eligible layer call -- torch TP group
    ranks -> MPI_Comm_split -> init_dcu(subcomm, tp_rank, tp_size). One
    engine world per process (single TP group per process).
  - ctx cache keyed (M, N, dtype); v8ring context geometry verbatim
    (heap/lane/tickets/planes), SUB=1 (frozen law).
  - forward: v9ring op body -- per-ring-block vendor GEMM of the local
    K-shard (input_is_parallel + local weight) on the caller stream, ring
    relay with a dtype-aware add kernel (bf16 integration-local variant;
    the frozen kernels are fp16-only).
  - backward: RS^-1 = all-gather of the output-block grad (stock RCCL --
    at TP=2 pure-comm AG is RCCL's win, engine loses without a GEMM to
    hide behind) + two vendor GEMMs for grad_input/grad_weight.
  - dispatcher: seg payload m_pr*N*itemsize >= MGD_MIN_MB (default 64)
    else stock path (payload law, S58b); MGD_DRY=1 counts eligible calls
    without engaging (interception probe).

Usage (env): MGD_FUSED=1 [MGD_MIN_MB=64] [MGD_DRY=1] + import this module
after megatron.core.tensor_parallel is importable.
"""
import ctypes
import os
import time

import torch
import triton
import triton.language as tl

from stowaway.kernels.common_ops import hip, init_dcu, memcpy_async
from stowaway.kernels.gemm_ar_rd import _wait_ticket
from stowaway.kernels.gemm_rs_v8ring import (
    _arr_cell,
    _slot_va,
)

_TP_TIMING = os.environ.get("MGD_TP_TIMING", "")   # 1=fused 2=stock
# timing-2 installs the patch with fusion disabled (stock baseline) --
# _eligible must behave as if enabled for the stock-timing branch to see
# the same op set the fused arm takes.
_TP_BASELINE = _TP_TIMING == "2"
_MIN_MB = float(os.environ.get("MGD_MIN_MB", "64"))
# T21 RS shape dispatcher: fuse only ops whose per-rank input width K_r
# (input dim = weight.shape[1]) is >= this -- o_proj K_r=512 vs
# down_proj K_r=1536 at TP=8. 0 = no K gating (previous behavior).
_RS_MIN_K = int(os.environ.get("MGD_RS_MIN_K", "0"))
# RSOP differential arm: "rccl" keeps the identical block-GEMM partial
# layout but reduces via plain RCCL reduce_scatter instead of the v9ring
# relay -- splits the in-vehicle penalty into compute-form vs relay cost.
_RS_RELAY = os.environ.get("MGD_RS_RELAY", "ring")
# skew-accumulation DIAGNOSTIC (E1): every Kth fused op re-syncs all
# ranks (host barrier -- poison for steady state, fine as a mechanism
# probe). If the fused-RS step time DROPS with K~8, the phantom-device-
# win law is rank-desync: RCCL collectives were the implicit sync our
# P2P relay removed.
_RESYNC = int(os.environ.get("MGD_RESYNC", "0"))
_rs_opn = 0
# 1-op lane backpressure (MGD_BP=1): before enqueueing op i, wait for
# op i-1's lane join. hipMemcpyAsync host-calls BLOCK when the lane's
# engine queue is full (5.7ms/call once the host runs ahead -- t21
# hostprof), which the enqueue median hides (p90 spikes 10-12ms in the
# vehicle). Capping run-ahead at one op keeps the queue shallow; the
# bounded wait overlaps device work that must happen anyway.
_BP = os.environ.get("MGD_BP", "0") == "1"
_DRY = os.environ.get("MGD_DRY", "") == "1"
_ENABLED = os.environ.get("MGD_FUSED", "") == "1" or _TP_BASELINE

# per-op timing (default off): compute-stream events + deferred readout --
# the previous op's elapsed_time is read at the NEXT op, when its end
# event has long fired (no sync poisoning; S47 law: events live on the
# compute stream only).
_tp_oplog = []          # (host_ms, wall_ms) pairs; flushed every 64 ops
_tp_wraplog = []        # MGD_WRAP_TIMING=1: full fused-branch host span
_WRAPT = os.environ.get("MGD_WRAP_TIMING", "") == "1"
_tp_prev = None         # (ev_start, ev_end, host_ms) of the previous op


def _tp_log_flush():
    if not _tp_oplog:
        return
    es = sorted(h for h, _ in _tp_oplog)
    ts = sorted(w for _, w in _tp_oplog)
    print(f"[mgd-tp] opwall n={len(_tp_oplog)} med {ts[len(ts)//2]:.1f} "
          f"p90 {ts[int(0.9 * len(ts))]:.1f} max {ts[-1]:.1f} | "
          f"enqueue med {es[len(es)//2]:.1f} p90 {es[int(0.9 * len(es))]:.1f}",
          flush=True)
    _tp_oplog.clear()
    if _tp_wraplog:
        ws = sorted(_tp_wraplog)
        msg = (f"[mgd-tp] wrap n={len(ws)} med {ws[len(ws)//2]:.1f} "
               f"p90 {ws[int(len(ws)*0.9)]:.1f}")
        if es:
            msg += f" | apply-enqueue-only med {es[len(es)//2]:.1f}"
        print(msg, flush=True)
        del _tp_wraplog[:]

_stats = {"bootstrap": False, "eligible": 0, "fused": 0, "fallback": 0}
_engine = {}          # bootstrap result: {"mpi","comm","rank","size"}
_ctxs = {}            # (M, N, dtype) -> ctx


@triton.jit(do_not_specialize=["dst", "src", "add", "n"])
def _add_va_bf16(dst, src, add, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask)
    y = tl.load(add.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask)
    tl.store(dst.to(tl.pointer_type(tl.bfloat16)) + offs, x + y, mask=mask)


@triton.jit(do_not_specialize=["dst", "src", "add", "n"])
def _add_va_f16(dst, src, add, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src.to(tl.pointer_type(tl.float16)) + offs, mask=mask)
    y = tl.load(add.to(tl.pointer_type(tl.float16)) + offs, mask=mask)
    tl.store(dst.to(tl.pointer_type(tl.float16)) + offs, x + y, mask=mask)


def _bootstrap_all():
    """Eager engine-world bootstrap at train() entry: ALL world ranks call
    MPI_Comm_split in lockstep (lazy bootstrap deadlocks -- PP stages reach
    the first eligible layer at different times, and Comm_split is
    collective over the full communicator)."""
    from megatron.core import parallel_state as mpu

    group = mpu.get_tensor_model_parallel_group()
    _bootstrap(group)


def _bootstrap(group):
    """One engine world over the TP subgroup."""
    import pydushmem.host as pHost

    mpi, world = pHost.mpi_init()
    granks = torch.distributed.get_process_group_ranks(group)
    color = min(granks)
    key = torch.distributed.get_rank()          # keep torch-rank order
    # OMPI: MPI_Comm is a pointer -- the OUT param is pointer-to-pointer
    # (a POINTER(c_int) truncates the handle and the next call segfaults;
    # see bench/s62_split_repro.py). pHost convention: handles flow as
    # integer addresses.
    mpi.MPI_Comm_split.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_void_p)]
    mpi.MPI_Comm_split.restype = ctypes.c_int
    newcomm = ctypes.c_void_p(0)
    rc = mpi.MPI_Comm_split(ctypes.c_void_p(world), color, key,
                            ctypes.byref(newcomm))
    assert rc == 0, f"MPI_Comm_split rc={rc}"
    assert newcomm.value, "MPI_Comm_split returned NULL"
    sub = newcomm.value
    r = pHost.mpi_comm_rank(mpi, sub)
    n = pHost.mpi_comm_size(mpi, sub)
    assert n == group.size() and r == granks.index(key), \
        f"subgroup mismatch r={r} n={n} granks={granks}"
    init_dcu(mpi, sub, r, n)
    _engine.update({"mpi": mpi, "comm": sub, "rank": r, "size": n})
    _stats["bootstrap"] = True
    if r == 0:
        print(f"[mgd] engine world up: tp_size={n} granks={granks}",
              flush=True)


def _get_ctx(M, N, dtype, group):
    key = (M, N, dtype)
    if key not in _ctxs:
        import stowaway.kernels.gemm_rs_v8ring as v8
        assert _engine, "engine not bootstrapped (train() hook missing)"
        # SUB law: SUB=1 (frozen bench path) up to ~128MB segs; larger
        # segs fall back to the engine's auto-sub (~14MB chunks; v8ring
        # reads S8R_SUB<=0 as auto) -- a single 134MB memcpy_async chunk
        # is outside the tested envelope (MBS=8 wedge suspect).
        seg_bytes = (M // _engine["size"]) * N * torch.empty(
            (), dtype=dtype).element_size()
        os.environ["S8R_SUB"] = "0" if seg_bytes > 128e6 else "1"
        c = v8.create_gemm_rs_v8ring_context(
            M, N, dtype, _engine["rank"], _engine["size"],
            _engine["mpi"], _engine["comm"])
        # v8ring context shape law: slot = m_pr*N = (M/R)*N.
        # attach the v9ring per-step event pool here (creation order law:
        # the pool must exist before the op records into it)
        h = hip()
        h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        h.hipEventCreateWithFlags.restype = ctypes.c_int
        c._blk_events = []
        for _ in range(_engine["size"]):
            ev = ctypes.c_void_p()
            assert h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2) == 0
            c._blk_events.append(ev)
        _ctxs[key] = c
    return _ctxs[key]


class _FusedRowParallelRS(torch.autograd.Function):
    """out = rows-block `tp_rank` of sum_r X_r @ W_r^T  (v9ring relay)."""

    @staticmethod
    def forward(ctx, a, w, eng, group, m_pr, N):
        ctx.eng = eng
        ctx.group = group
        ctx.m_pr = m_pr
        ctx.N = N
        ectx = _get_ctx(a.shape[0], N, a.dtype, group)   # bootstraps eng
        rank, R = eng["rank"], eng["size"]

        M = a.shape[0]
        cur = torch.cuda.current_stream()
        h = hip()
        lane = ectx.ring_lane
        nxt = (rank + 1) % R
        prv = (rank - 1 + R) % R

        ectx.round_base += ectx.round_stride
        base = ectx.round_base
        tickets_va = ectx.tickets.data_ptr()
        elemsz = ectx.elemsz
        mode = ectx.wait_mode
        partial_va = ectx.partial.data_ptr()
        addk = _add_va_f16 if a.dtype == torch.float16 else _add_va_bf16
        w_t = w.t()

        def ticket(seg, si):
            # v8ring plane law: slots/flags are laid out at seg*sub+si, and
            # round_stride counts every (seg,si) pair -- the ticket must too,
            # else values collide across segs at SUB>1.
            return base + seg * ectx.sub + si + 1

        def add(dst_va, src_va, add_va):
            n_blks = triton.cdiv(ectx.sub_elems, 8192)
            addk[(n_blks,)](dst_va, src_va, add_va, ectx.sub_elems,
                            BLOCK=8192, num_warps=8)

        def wait_and_add(seg, dst_base):
            blk = (seg - 1 + R) % R
            own_va = partial_va + blk * m_pr * N * elemsz
            for si in range(ectx.sub):
                _wait_ticket(_arr_cell(ectx.arr_ptr, prv, seg, si, ectx),
                             ticket(seg, si), lane, mode)
                add(dst_base + si * ectx.sub_elems * elemsz,
                    _slot_va(ectx.heap_ptr, seg, si, ectx),
                    own_va + si * ectx.sub_elems * elemsz)

        def forward_seg(seg, src_base):
            for si in range(ectx.sub):
                memcpy_async(_slot_va(ectx.peer_heap[nxt], seg, si, ectx),
                             src_base + si * ectx.sub_elems * elemsz,
                             ectx.sub_elems * elemsz, lane.cuda_stream)
                memcpy_async(_arr_cell(ectx.peer_arr[nxt], rank, seg, si,
                                       ectx),
                             tickets_va + 4 * ticket(seg, si), 4,
                             lane.cuda_stream)

        # compute FIRST on the caller stream, lane gated per block (S51).
        # block order = ring order (rank-1-k); the OWN block (rank) is
        # computed last and gated for the final add (S51 ordering law).
        for k in range(R):
            blk = (rank - 1 - k) % R
            torch.matmul(a[blk * m_pr:(blk + 1) * m_pr], w_t,
                         out=ectx.partial[blk * m_pr:(blk + 1) * m_pr])
            rc = h.hipEventRecord(ectx._blk_events[k],
                                  ctypes.c_void_p(cur.cuda_stream))
            assert rc == 0

        if _RS_RELAY == "rccl":
            # RSOP differential arm (see _RS_RELAY above)
            torch.distributed.reduce_scatter_tensor(_out_holder["out"],
                                                    ectx.partial,
                                                    group=ctx.group)
            ctx.save_for_backward(a, w)
            return _out_holder["out"]

        rc = h.hipEventRecord(ectx._ev_fork, ctypes.c_void_p(cur.cuda_stream))
        assert rc == 0
        rc = h.hipStreamWaitEvent(ctypes.c_void_p(lane.cuda_stream),
                                  ectx._ev_fork, 0)
        assert rc == 0

        with torch.cuda.stream(lane):
            for k in range(R - 1):
                seg = (rank - k) % R
                rc = h.hipStreamWaitEvent(ctypes.c_void_p(lane.cuda_stream),
                                          ectx._blk_events[k], 0)
                assert rc == 0
                if k == 0:
                    blk = (seg - 1 + R) % R
                    forward_seg(seg, partial_va + blk * m_pr * N * elemsz)
                else:
                    wait_and_add(seg, _slot_va(ectx.heap_ptr, seg, 0, ectx))
                    forward_seg(seg, _slot_va(ectx.heap_ptr, seg, 0, ectx))
            rc = h.hipStreamWaitEvent(ctypes.c_void_p(lane.cuda_stream),
                                      ectx._blk_events[R - 1], 0)
            assert rc == 0, "final own-gate rc"
            wait_and_add((rank + 1) % R, _out_va_holder["out"])

        rc = h.hipEventRecord(ectx._ev_join, ctypes.c_void_p(lane.cuda_stream))
        assert rc == 0
        rc = h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream),
                                  ectx._ev_join, 0)
        assert rc == 0
        ctx.save_for_backward(a, w)
        return _out_holder["out"]

    @staticmethod
    def backward(ctx, grad_out):
        a, w = ctx.saved_tensors
        M, K = a.shape
        N = ctx.N
        R = ctx.group.size()
        full = torch.empty(M, N, dtype=grad_out.dtype, device=a.device)
        torch.distributed.all_gather_into_tensor(full, grad_out.contiguous(),
                                                 group=ctx.group)
        grad_input = torch.matmul(full, w)                 # [M, K]
        grad_weight = torch.matmul(full.transpose(-2, -1).contiguous(), a)
        return grad_input, grad_weight, None, None, None, None


_out_holder = {}
_out_va_holder = {}
_dx_ctxs = {}
_DX_BWD_T = os.environ.get("OPUS_DX_BWD", "") == "t"
_DX_BWD_AG = os.environ.get("OPUS_DX_BWD", "") == "ag"
# runtime path switches (read at call time; the autotuner flips them
# between iterations, identically on every rank)
_AUTOTUNE = os.environ.get("OPUS_AUTOTUNE", "") == "1"
_RT = {"row": True, "bwd": "ag" if _DX_BWD_AG else "t", "col": True}
_dx_bwd_ctxs = {}


def _get_dx_bwd_ctx(M, N, dtype):
    key = (M, N, dtype)
    if key not in _dx_bwd_ctxs:
        from stowaway.kernels.gemm_rs_dx import DXAGBwdContext
        _dx_bwd_ctxs[key] = DXAGBwdContext(M, N, dtype, _engine["rank"],
                                           _engine["size"])
        if _engine["rank"] == 0:
            print(f"[mgd] dx bwd-ag ctx M={M} N={N}", flush=True)
    return _dx_bwd_ctxs[key]
_DX_PHASE = os.environ.get("OPUS_DX_PHASE", "")   # ""|fwd|rec


def _get_dx_ctx(M, N, dtype):
    key = (M, N, dtype)
    if key not in _dx_ctxs:
        from stowaway.kernels.gemm_rs_dx import DXGemmRSContext
        assert _engine, "engine not bootstrapped (train() hook missing)"
        _dx_ctxs[key] = DXGemmRSContext(M, N, dtype, _engine["rank"],
                                        _engine["size"])
        if _engine["rank"] == 0:
            print(f"[mgd] dx ctx M={M} N={N} heap "
                  f"{_engine['size'] * (M // _engine['size']) * N * 2 / 2**20:.0f}MB",
                  flush=True)
    return _dx_ctxs[key]


class _FusedRowParallelDX(torch.autograd.Function):
    """opus fork: row-parallel GEMM + SP reduce-scatter by direct exchange
    (block vendor GEMMs in permutation order, each pushed to its home as
    soon as computed, one fused sum at the home).  Backward = stock."""

    @staticmethod
    def forward(ctx, a, w, group, m_pr, N):
        from stowaway.kernels.gemm_rs_dx import dx_gemm_rs_op
        ctx.group = group
        ctx.N = N
        dctx = _get_dx_ctx(a.shape[0], N, a.dtype)
        out = torch.empty(m_pr, N, dtype=a.dtype, device=a.device)
        dx_gemm_rs_op(a, w, out, dctx)
        ctx.save_for_backward(a, w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, w = ctx.saved_tensors
        M = a.shape[0]
        if _RT["bwd"] == "ag" and (_DX_BWD_AG or _AUTOTUNE):
            # fused: DX permutation all-gather of grad_out, per-block dgrad
            # as each block lands, stock-form (TN view) wgrad
            from stowaway.kernels.gemm_rs_dx import dx_ag_bwd_op
            bctx = _get_dx_bwd_ctx(M, ctx.N, grad_out.dtype)
            grad_input, grad_weight, _ = dx_ag_bwd_op(grad_out, w, a, bctx)
            return grad_input, grad_weight, None, None, None
        full = torch.empty(M, ctx.N, dtype=grad_out.dtype, device=a.device)
        torch.distributed.all_gather_into_tensor(full, grad_out.contiguous(),
                                                 group=ctx.group)
        grad_input = torch.matmul(full, w)
        if _DX_BWD_T or _AUTOTUNE:
            # stock form (layers.py: grad_output.t().matmul(total_input)):
            # no materialized [N, M] transpose copy (335MB at 27B MBS8)
            grad_weight = full.t().matmul(a)
        else:
            grad_weight = torch.matmul(full.transpose(-2, -1).contiguous(), a)
        return grad_input, grad_weight, None, None, None


# --------------------------------------------------------------------------
# opus fork: column-parallel SP forward = DX all-gather fused with the GEMM
# (OPUS_CP=1). Backward = stock form (re-gather, dgrad, TN wgrad, dgrad RS
# through the live dist_reduce_scatter_func so MGD_BRS still applies).
# --------------------------------------------------------------------------
_CP = os.environ.get("OPUS_CP", "") == "1"
_CP_MAX_NC = int(os.environ.get("OPUS_CP_MAX_NC", "8192"))   # skips vocab head
# 11-01: lower width bound, default 0 = off. OPUS_CP_MIN_NC=2048 on 70B fuses fc1 (Nc 7168) but not QKV (Nc 1280)
_CP_MIN_NC = int(os.environ.get("OPUS_CP_MIN_NC", "0"))
_cp_ctxs = {}
_cp_orig = {}
_cp_stats = {"cp_fused": 0, "cp_skip": 0}
# per-op device time (compute-stream events, deferred readout; the prev
# op's end event is synchronized at the next op = diagnostic only):
# OPUS_CP_TIMING=1 times the fused op, =2 times stock on the same op set
_CP_TIMING = os.environ.get("OPUS_CP_TIMING", "")
_cp_tlog = {}
_cp_prev = None


def _cp_tick(nc, ev_s, ev_e):
    global _cp_prev
    if _cp_prev is not None:
        n0, ps, pe = _cp_prev
        pe.synchronize()
        lst = _cp_tlog.setdefault(n0, [])
        lst.append(ps.elapsed_time(pe))
        if len(lst) >= 64:
            ts = sorted(lst)
            print(f"[mgd-cp] mode={_CP_TIMING} Nc={n0} opwall n={len(ts)} "
                  f"med {ts[len(ts) // 2]:.3f} p90 {ts[int(0.9 * len(ts))]:.3f}",
                  flush=True)
            lst.clear()
    _cp_prev = (nc, ev_s, ev_e)


def _get_cp_ctx(M, K, dtype):
    key = (M, K, dtype)
    if key not in _cp_ctxs:
        from stowaway.kernels.gemm_rs_dx import DXAGGemmContext
        _cp_ctxs[key] = DXAGGemmContext(M, K, dtype, _engine["rank"],
                                        _engine["size"])
        if _engine["rank"] == 0:
            print(f"[mgd] cp ag-gemm ctx M={M} K={K}", flush=True)
    return _cp_ctxs[key]


class _FusedColParallelDX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, group):
        from stowaway.kernels.gemm_rs_dx import dx_ag_gemm_op
        S, B, K = input.shape
        R = group.size()
        M = S * R * B
        cctx = _get_cp_ctx(M, K, input.dtype)
        out, _ = dx_ag_gemm_op(input.reshape(S * B, K), weight, cctx)
        ctx.save_for_backward(input, weight)
        ctx.group = group
        return out.view(S * R, B, weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        import megatron.core.tensor_parallel.layers as _L
        input, weight = ctx.saved_tensors
        group = ctx.group
        S, B, K = input.shape
        R = group.size()
        total = torch.empty(S * R, B, K, dtype=input.dtype, device=input.device)
        handle = torch.distributed.all_gather_into_tensor(
            total, input.contiguous(), group=group, async_op=True)
        grad_input = grad_output.matmul(weight)
        handle.wait()
        # stock order (layers.py backward): dgrad RS in flight under wgrad
        sub = torch.empty(S, B, K, dtype=input.dtype, device=input.device)
        h2 = _L.dist_reduce_scatter_func(sub, grad_input, group=group,
                                         async_op=True)
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        grad_weight = g2.t().matmul(total.view(-1, K))
        h2.wait()
        return sub, grad_weight, None


def _install_cp():
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    orig = ColumnParallelLinear._forward_impl

    def patched(self, input, weight, *args, **kwargs):
        ok = (_engine and kwargs.get("sequence_parallel")
              and kwargs.get("bias") is None
              and not kwargs.get("gradient_accumulation_fusion")
              and not kwargs.get("allreduce_dgrad")
              and kwargs.get("grad_output_buffer") is None
              and weight.requires_grad and input.dim() == 3
              and input.dtype in (torch.bfloat16, torch.float16)
              and _CP_MIN_NC <= weight.shape[0] <= _CP_MAX_NC and not args)
        if not ok or not _RT["col"]:
            _cp_stats["cp_skip"] += 1
            return orig(self, input, weight, *args, **kwargs)
        if _CP_TIMING in ("h", "hs"):
            # host enqueue span only: no events, no sync (does not perturb
            # the stream); h = fused, hs = stock on the same op set
            t0 = time.perf_counter()
            if _CP_TIMING == "hs":
                r = orig(self, input, weight, *args, **kwargs)
            else:
                _cp_stats["cp_fused"] += 1
                r = _FusedColParallelDX.apply(input, weight, kwargs["tp_group"])
            nc = weight.shape[0]
            lst = _cp_tlog.setdefault(nc, [])
            lst.append((time.perf_counter() - t0) * 1e3)
            if len(lst) >= 256:
                ts = sorted(lst)
                print(f"[mgd-cp] mode={_CP_TIMING} Nc={nc} host n={len(ts)} "
                      f"med {ts[len(ts) // 2]:.3f} p90 {ts[int(0.9 * len(ts))]:.3f} "
                      f"mean {sum(ts) / len(ts):.3f}", flush=True)
                lst.clear()
            return r
        if _CP_TIMING and _CP_TIMING != "hb":
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            if _CP_TIMING == "2":
                r = orig(self, input, weight, *args, **kwargs)
            else:
                _cp_stats["cp_fused"] += 1
                r = _FusedColParallelDX.apply(input, weight, kwargs["tp_group"])
            ev_e.record()
            _cp_tick(weight.shape[0], ev_s, ev_e)
            return r
        _cp_stats["cp_fused"] += 1
        return _FusedColParallelDX.apply(input, weight, kwargs["tp_group"])

    ColumnParallelLinear._forward_impl = patched
    _cp_orig["cls"], _cp_orig["fn"] = ColumnParallelLinear, orig
    print(f"[mgd] ColumnParallelLinear._forward_impl patched (OPUS_CP, "
          f"MIN_NC={_CP_MIN_NC} MAX_NC={_CP_MAX_NC})", flush=True)


# --------------------------------------------------------------------------
# opus fork: online greedy autotuner (OPUS_AUTOTUNE=1).  Path classes:
#   row = fused row-parallel GEMM+RS forward (DX)
#   bwd = fused row-parallel backward (DX-AG + per-block dgrad) vs stock-form
#   col = fused column-parallel AG+GEMM forward (DX)
# Warm W iters with every class on (contexts + kernels built), then greedy:
# measure all-off, then switch on row, bwd, col one at a time and keep each
# only if the synced iteration wall (max over ranks) drops by > margin.
# Every probe = 2 iterations, the second one is measured.  The decision is
# made from all-reduced (MAX) timings, so every rank flips identically.
# --------------------------------------------------------------------------
_AT_W = int(os.environ.get("OPUS_AT_WARM", "2"))
_AT_MARGIN = float(os.environ.get("OPUS_AT_MARGIN", "0.003"))
_at = {"it": 0, "best": None, "cand": None, "plan": None, "done": False, "log": []}


def _at_set(cfg):
    _RT.update(cfg)


def _at_next_plan():
    base = {"row": False, "bwd": "t", "col": False}
    return [("base", dict(base)),
            ("+row", {"row": True}),
            ("+bwd", {"bwd": "ag"}),
            ("+col", {"col": True})]


_AT_L = int(os.environ.get("OPUS_AT_PROBE", "3"))   # iters per probe: 1 switch + (L-1) timed
# diagnostics (default off): OPUS_AT_WARM_COL=0 keeps col off during warm-up
# (no column ctx built); OPUS_AT_FIXED="row,bwd" pins that config after
# warm-up with no probing (row/bwd/col names; bwd -> "ag", absent -> off/"t")
_AT_WARM_COL = os.environ.get("OPUS_AT_WARM_COL", "0") != "0"   # 10-31: lazy col is the validated default (F32/F33)
_fx = os.environ.get("OPUS_AT_FIXED")
_AT_FIXED = None if _fx is None else {
    "row": "row" in _fx.split(","), "col": "col" in _fx.split(","),
    "bwd": "ag" if "bwd" in _fx.split(",") else "t"}


def _at_before(it):
    """returns 'start' / 'stop' / None for timing at the END of iteration it.
    Probe = L iterations: switch at phase 0, sync+start at the end of phase
    0, sync+stop at the end of phase L-1 -> (L-1) pipelined steady-state
    iterations are timed (a single synced iteration over-credits paths
    whose cost is host enqueue: opus_e2e_70b/a2 probe +col 17345.5 vs
    separate-process all-on 17702.0)."""
    if _at["done"]:
        return None
    if _at["plan"] is None:
        _at["plan"] = _at_next_plan()
    if it < _AT_W:
        _at_set({"row": True, "bwd": "ag", "col": _AT_WARM_COL})
        return None
    if _AT_FIXED is not None:
        # diagnostic: skip probing, pin a config right after warm-up
        _at["done"] = True
        _at_set(_AT_FIXED)
        if not _AT_FIXED["col"] and _AT_RELEASE:
            _at_release_col()
        if _engine.get("rank", 0) == 0:
            print(f"[autotune] FIXED {_AT_FIXED} at it={it} "
                  f"(warm col={_AT_WARM_COL})", flush=True)
        return None
    k, ph = divmod(it - _AT_W, _AT_L)
    if k >= len(_at["plan"]):
        return None
    if ph == 0:
        name, delta = _at["plan"][k]
        cfg = dict(_at["best"][1]) if _at["best"] else {}
        cfg.update(delta)
        if name == "+bwd" and not cfg.get("row"):
            cfg["bwd"] = "t"
        _at["cand"] = (name, cfg)
        _at_set(cfg)
        return "start"
    if ph == _AT_L - 1:
        return "stop"
    return None


# OPUS_AT_RELEASE=1 (default): when col is dropped, free its contexts and
# restore the stock ColumnParallelLinear (opus_e2e_27b_atres: col ctx left
# resident after warm-up = +2.0%/iter, wrapper + rest +0.6%)
_AT_RELEASE = os.environ.get("OPUS_AT_RELEASE", "1") != "0"


def _at_release_col():
    n = len(_cp_ctxs)
    for c in list(_cp_ctxs.values()):
        c.close()
    _cp_ctxs.clear()
    if _cp_orig:
        _cp_orig["cls"]._forward_impl = _cp_orig["fn"]
    torch.cuda.empty_cache()
    if _engine.get("rank", 0) == 0:
        print(f"[autotune] released col: {n} ctx closed, stock "
              f"ColumnParallelLinear restored", flush=True)


def _at_after(it, wall_ms):
    import torch.distributed as dist
    t = torch.tensor([wall_ms / (_AT_L - 1)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    w = t.item()
    name, cfg = _at["cand"]
    keep = _at["best"] is None or w < _at["best"][0] * (1.0 - _AT_MARGIN)
    _at["log"].append((name, dict(cfg), round(w, 1), keep))
    if keep:
        _at["best"] = (w, dict(cfg))
    if _engine.get("rank", 0) == 0:
        print(f"[autotune] it={it} probe {name} cfg={cfg} wall/iter={w:.1f}ms "
              f"{'KEEP' if keep else 'drop'} best={_at['best'][1]}", flush=True)
    if (it - _AT_W) // _AT_L == len(_at["plan"]) - 1:
        _at["done"] = True
        _at_set(_at["best"][1])
        if not _at["best"][1].get("col") and _AT_RELEASE:
            _at_release_col()
        if _engine.get("rank", 0) == 0:
            print(f"[autotune] CHOSEN {_at['best'][1]} after {it + 1} iters | {_at['log']}", flush=True)


def _install_autotune():
    """hook train_step in EVERY loaded training module: the das adaptor
    swaps megatron.training.training.train for hcu_megatron's own train(),
    which calls hcu_megatron.training.training.train_step (opus_e2e_70b/a1:
    a megatron-only hook never fired)."""
    import sys as _sys
    mods = [m for n, m in list(_sys.modules.items())
            if n.endswith("training.training") and hasattr(m, "train_step")
            and not getattr(m.train_step, "_opus_at", False)]
    for m in mods:
        _hook_train_step(m)
    print(f"[autotune] installed (greedy over row/bwd/col) on {[m.__name__ for m in mods]}", flush=True)


def _hook_train_step(_mt):
    orig = _mt.train_step

    def train_step(*a, **kw):
        it = _at["it"]
        act = _at_before(it)
        r = orig(*a, **kw)
        if act == "start":
            torch.cuda.synchronize()
            _at["t0"] = time.perf_counter()
        elif act == "stop":
            torch.cuda.synchronize()
            _at_after(it, (time.perf_counter() - _at["t0"]) * 1e3)
        _at["it"] = it + 1
        return r

    train_step._opus_at = True
    _mt.train_step = train_step


def _eligible(self, input_):
    if not _ENABLED or not _RT["row"]:
        return False
    if not self.sequence_parallel or self.explicit_expert_comm:
        return False
    if self.gradient_accumulation_fusion or self.bias is not None:
        return False
    if input_.dim() != 3:                                  # [S, B, H]
        return False
    if input_.dtype not in (torch.float16, torch.bfloat16):
        return False
    M = input_.shape[0] * input_.shape[1]
    N = self.weight.shape[0]
    R = self.tp_group.size()
    if R < 2 or M % R or N % 128 or (M // R) % 64:
        return False
    if _RS_MIN_K and self.weight.shape[1] < _RS_MIN_K:
        return False
    seg_mb = (M // R) * N * input_.element_size() / 1e6
    if _DX_PHASE:
        # full recompute: the first forward runs under no_grad
        # (CheckpointFunction.forward), the recompute under enable_grad
        g = torch.is_grad_enabled()
        if (_DX_PHASE == "fwd" and g) or (_DX_PHASE == "rec" and not g):
            return False
    return seg_mb >= _MIN_MB


def install():
    """Patch RowParallelLinear.forward (class-level, env-gated)."""
    from megatron.core.tensor_parallel.layers import RowParallelLinear

    orig = RowParallelLinear.forward

    def patched(self, input_):
        if _eligible(self, input_):
            _stats["eligible"] += 1
            if _TP_TIMING == "2" and not _DRY:
                # baseline arm: patch installed but fusion disabled --
                # time the stock GEMM+reduce_scatter on the same vehicle
                # op set for a like-for-like device-opwall comparison.
                t0 = time.perf_counter()
                ev_s = torch.cuda.Event(enable_timing=True)
                ev_e = torch.cuda.Event(enable_timing=True)
                ev_s.record()
                r = orig(self, input_)
                ev_e.record()
                global _tp_prev
                host_ms = (time.perf_counter() - t0) * 1e3
                if _tp_prev is not None:
                    ps, pe, _ = _tp_prev
                    pe.synchronize()
                    _tp_oplog.append((host_ms, ps.elapsed_time(pe)))
                    if len(_tp_oplog) >= 64:
                        _tp_log_flush()
                _tp_prev = (ev_s, ev_e, host_ms)
                return r
            M = input_.shape[0] * input_.shape[1]
            N = self.weight.shape[0]
            if not _DRY:
                global _rs_opn
                _rs_opn += 1
                if _RESYNC and _rs_opn % _RESYNC == 0:
                    torch.distributed.barrier(group=self.tp_group)
                tw0 = time.perf_counter() if _WRAPT else 0.0
                a2 = input_.view(M, input_.shape[-1])
                out = torch.empty(M // self.tp_group.size(), N,
                                  dtype=input_.dtype, device=input_.device)
                _out_holder["out"] = out
                _out_va_holder["out"] = out.data_ptr()
                if _RS_RELAY == "dx":
                    _out_holder.pop("out", None)
                if _BP:
                    c = _ctxs.get((M, N, input_.dtype))
                    if c is not None and hip().hipEventQuery(
                            c._ev_join) != 0:
                        hip().hipEventSynchronize(c._ev_join)
                _tim = _TP_TIMING == "1"
                t0 = time.perf_counter() if _tim else 0.0
                ev_s = torch.cuda.Event(enable_timing=True) if _tim \
                    else None
                ev_e = torch.cuda.Event(enable_timing=True) if _tim \
                    else None
                if _tim:
                    ev_s.record()
                if _RS_RELAY == "dx":
                    out = _FusedRowParallelDX.apply(
                        a2, self.weight, self.tp_group,
                        M // self.tp_group.size(), N)
                else:
                    out = _FusedRowParallelRS.apply(
                        a2, self.weight, _engine, self.tp_group,
                        M // self.tp_group.size(), N)
                if _WRAPT:
                    _tp_wraplog.append((time.perf_counter() - tw0) * 1e3)
                if _tim:
                    ev_e.record()
                    # (_tp_prev already declared global above)
                    host_ms = (time.perf_counter() - t0) * 1e3
                    if _tp_prev is not None:
                        ps, pe, _ = _tp_prev
                        pe.synchronize()      # prev end fired long ago
                        _tp_oplog.append((host_ms, ps.elapsed_time(pe)))
                        if len(_tp_oplog) >= 64:
                            _tp_log_flush()
                    _tp_prev = (ev_s, ev_e, host_ms)
                _stats["fused"] += 1
                # SP contract: return [s/R, b, h] like stock (the RS row
                # block of the flat [s*b, h] maps 1:1 onto the seq chunk)
                R = self.tp_group.size()
                out = out.view(input_.shape[0] // R, input_.shape[1], N)
                output_bias = self.bias if self.skip_bias_add else None
                return out, output_bias
        return orig(self, input_)

    RowParallelLinear.forward = patched
    if _CP or _AUTOTUNE:
        _install_cp()
    if _AUTOTUNE:
        _install_autotune()

    # eager engine bootstrap: every world rank enters train() in lockstep
    import megatron.training.training as _mt
    _orig_train = _mt.train

    def train(*args, **kwargs):
        import sys as _s
        # guard: when the ag patch's outer hook already bootstrapped
        # (MGD_AG_TIMING=2 baseline arm runs BOTH hooks), a second
        # _bootstrap_all would re-enter MPI_Init -> MPI_ERR_OTHER abort.
        if not _engine:
            _bootstrap_all()
        if _AUTOTUNE:
            _install_autotune()
        try:
            r = _orig_train(*args, **kwargs)
        except BaseException:
            _s._mgd_exit_code = 1          # propagate failure through the
            raise                          # atexit os._exit (see below)
        _s._mgd_exit_code = 0
        if os.environ.get("MGD_STATS_EARLY") and \
                _engine.get("rank", 0) == 0:
            # R=8: rank0's teardown crashes BEFORE the atexit dump;
            # emit stats right after train() returns instead.
            print(f"[mgd] early stats {stats()}", flush=True)
        return r

    _mt.train = train
    print(f"[mgd] RowParallelLinear.forward patched (MIN_MB={_MIN_MB} "
          f"DRY={_DRY})", flush=True)


def stats():
    return dict(_stats, n_ctx=len(_ctxs), **_cp_stats)


def _dump_stats_at_exit():
    if _engine.get("rank", 0) == 0:
        print(f"[mgd] exit stats {stats()}", flush=True)
    # teardown-order law: letting the interpreter proceed to NCCL
    # destructors after the engine's MPI world teardown segfaults (139)
    # on this stack -- everything meaningful is flushed above, so leave
    # NOW with the training exit code (sys.exit already in flight).
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    import os as _os_local
    _os_local._exit(getattr(_sys, "_mgd_exit_code", 0) or 0)


import atexit
atexit.register(_dump_stats_at_exit)


if os.environ.get("MGD_FUSED", "") == "1" or os.environ.get("MGD_DRY", "") == "1" \
        or os.environ.get("MGD_TP_TIMING", "") == "2":
    try:
        install()
    except Exception as e:                                  # noqa: BLE001
        print(f"[mgd] install deferred: {e}", flush=True)
        import atexit
        atexit.register(lambda: None)
