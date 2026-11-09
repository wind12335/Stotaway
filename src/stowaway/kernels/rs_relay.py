"""rs_relay: pure ring reduce-scatter relay for ALREADY-COMPUTED tensors
(t21 Round 12 -- the 536.9MB dgrad class).

WHY: the vehicle's dominant RS class (mappings._reduce_scatter_along_first_
dim + layers.py:349 backward, ~170 calls/step, 536.9MB in / 67.1MB out each)
runs at 29GB/s effective under RCCL -- 18.5ms/call vs a 645ms/step ring wire
floor.  PMC evidence: under the slowed host the same kernels finish in
5.05ms (106GB/s) -- the gap is RCCL's all-rank rendezvous spin (skew wait),
not wire.  This engine is the textbook neighbor-pipelined ring RS: every
wait is on the PREVIOUS NEIGHBOR's round only (no global rendezvous), so
per-op skew is absorbed once into the pipeline instead of 170 times.

Algorithm (verified R=3 by hand, send-before-wait per round):
  pre-loop : engine-send own partial of chunk (r-1)%R (+ticket 1)
  round k=1..R-1:
      wait ticket k on OWN cell (prev's round-k chunk landed in my slot)
      add   input[(r-k-1)%R] += recv_slot          (SM kernel, in place)
      if k < R-1:
          engine-send input[(r-k-1)%R] -> next rank's recv slot (+ticket k+1)
  final   : engine-copy input[r] -> caller's out (+done event)
  chunk c circulates ranks (c+1)..(c+R-1) accumulating, ends at owner c.

Slot reuse (single slot per writer: the writer's R-1 sends of one op AND
its next op's pre-send all target the same address) is closed by the ACK
plane: the reader fires a 1-CTA sys-release xchg kernel (ack=base+k)
into the WRITER's cell, stream-ordered after its add kernel; the writer
waits ack >= base+k
before the send that would overwrite (and ack >= prev_base+R-1 before the
next op's pre-send).  Acks were engine 4B memcpys in the probe-clean
draft -- the VEHICLE wedged op-1 with the host parked inside that
memcpy's enqueue (v5 ack-wedge law); acks carry VALUE only (no payload
ordering) so the SM xchg is safe where it would NOT be for tickets.
First probe draft used parity double-buffering
-- WRONG for a different reason: the R-1 sends of ONE op all hit the
SAME slot, so round k+1's send overwrote the reader's round-k chunk
in-flight (600/600 corrupt).  Tickets/acks: monotonic, single-use,
position-keyed (S15); all remote writes land in the WAITER's plane via
engine memcpy (S16 law).

Primitives (v3, the no-4B-engine-write law): DATA moves as big engine
memcpys only (never wedge); EVERY plane write (ticket + ack) is a 1-CTA
sys-release xchg kernel (gemm_ar_rd family -- a 4B hipMemcpyAsync
enqueued into a queue parked at an unresolved wait wedges the host in
the DTK runtime; both the ack site and the ticket site wedged the
vehicle op-1 while 860 probe ops stayed clean); EVERY waiter is the
128-mask hybrid spin kernel (POLL=128 law: the mask covers the peer-L2
propagation lag of engine data vs the faster SM ticket -- v20).  Peer
writes land in the WAITER's plane via host.ptr (S16 law).

Port law: NEW file; no existing kernel touched.
(RECONSTRUCTED 10-25: the on-disk copy truncated to 0 bytes at ~16:03
after the R12m edits -- recovered from the session Read + documented
edits; engine files now keep a .bak twin.)
"""
import ctypes

import torch
import triton
import triton.language as tl

from stowaway.kernels.common_ops import hip, init_dcu, memcpy_async, dcu_state
from stowaway.kernels.gemm_ar_rd import (
    MAX_TICKETS, _launch_flag_xchg, _wait_ticket)

# MGD_RR_HPROF=1 (S75B A1): host sub-span profile of rs_relay_op -- pure
# perf_counter accumulation, NO stream events (S47 instrument law).
# Phases: pre (ctx math + ctypes structs) / fork (ev pair) / body (the
# with-stream enqueue cluster) / tail (final caller-stream wait enqueue).
import os
import time as _time

# ---- t21 F2d: ticket-ceiling law (values NEVER recycle) ----
# The old `assert base + R < MAX_TICKETS` (1<<20) bounded only the arange
# staging table `tickets_va` -- a gemm_ar_rd legacy the ring path never
# reads (_launch_flag_xchg(dst, val) takes the host-computed value).  The
# REAL bound is the int32 flag word: base < 2^31 -> ~238M ops per ctx
# (vehicle ~1500 ops/iter -> ~160k iters).  The old assert killed the
# vehicle at ~116.5k ops (~iter 78; F2d R1 one-shot traceback; the SOAK
# "iter 2" was log-granularity illusion -- megatron prints memory only at
# iter 2/100, the trip sat between; CHAMP 25-iter sat below it).  Not a
# race: pure op-count ceiling.
# Recycling (base->0 + plane memset) is REJECTED: device sync drains my
# kernels but NOT the fabric -- an in-flight store from an already-retired
# peer xchg can land AFTER my memset and leave a stale LARGE value in the
# word; post-recycle waits use small targets and false-pass on it (t23w
# rig law: sync ckpt md=2.9e12, defer md=5.9e15).  Monotone-forever values
# make the false-pass class unrepresentable.  MGD_RR_WRAP_MIN instead
# places optional full-drain sync points (sync + barrier; no memset, no
# reset) -- correctness-neutral, lets rigs exercise the drain machinery.
_WRAP_LIMIT = int(os.environ.get("MGD_RR_WRAP_MIN", "0"))   # 0 = off
_I32_CEIL = (1 << 31) - (1 << 16)


def _advance_base(ctx):
    """Monotonic ticket base (never recycled).  Ring + a2a host paths."""
    if _WRAP_LIMIT and \
            (ctx.round_base + ctx.round_stride) // _WRAP_LIMIT \
            > ctx.round_base // _WRAP_LIMIT:
        _drain_point(ctx)                 # one sync point per window
    ctx.round_base += ctx.round_stride
    assert ctx.round_base + 2 * ctx.round_stride + ctx.num_ranks \
        < _I32_CEIL, (
            "ticket values would leave int32 (~238M ops/ctx): the engine "
            "needs a value-recycle protocol (see law above)")
    return ctx.round_base


def _drain_point(ctx):
    mpi = getattr(ctx, "mpi", None)
    assert mpi is not None, "ctx built without mpi handle (drain needs it)"
    torch.cuda.synchronize()              # my device drained
    rc = mpi.MPI_Barrier(ctypes.c_void_p(ctx.comm))
    assert rc == 0
    if ctx.rank == 0:
        print(f"[rs_relay] drain point op={ctx._n_ops} base="
              f"{ctx.round_base} (values stay monotone)", flush=True)

_HPROF = os.environ.get("MGD_RR_HPROF", "") == "1"
_RRH = {}


def _rrh(k, dt):
    a = _RRH.setdefault(k, [0.0, 0.0, 0])
    a[0] += dt
    if dt > a[1]:
        a[1] = dt
    a[2] += 1


def _tw(cat, fn, *a, **kw):
    # timed call wrapper (body sub-categories); fast path when off
    if not _HPROF:
        return fn(*a, **kw)
    t = _time.perf_counter()
    r = fn(*a, **kw)
    _rrh(cat, _time.perf_counter() - t)
    return r


def _rrh_flush(rank, force=False):
    n = _RRH.get("n", 0)
    if rank != 0 or (n % 1024 and not force):
        return
    parts = []
    for k in ("pre", "fork", "body", "c_wait", "c_add", "c_xchg",
              "c_memcpy", "tail"):
        if k in _RRH:
            sm, mx, cnt = _RRH[k]
            denom = cnt if k.startswith("c_") else max(n, 1)
            parts.append("%s mean=%.2fms max=%.2fms n=%d"
                         % (k, sm * 1e3 / max(denom, 1), mx * 1e3, cnt))
    print("[rr-hprof] n=%d %s" % (n, " ".join(parts)), flush=True)


@triton.jit(do_not_specialize=["dst", "add"])
def kernel_add_bf16(dst, add, n, BLOCK: tl.constexpr):
    """dst[i] += add[i] over raw VAs (the recv slot is symmetric-heap
    memory with no torch tensor). bf16 in/out, fp32 accumulate."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(dst.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask)
    y = tl.load(add.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask)
    tl.store(dst.to(tl.pointer_type(tl.bfloat16)) + offs, x + y, mask=mask)


@triton.jit(do_not_specialize=["dst", "src"])
def kernel_copy_bf16(dst, src, n, BLOCK: tl.constexpr):
    """dst[i] = src[i] over raw VAs -- SM replacement for engine memcpy
    (S75B A1: hipMemcpyAsync blocks the host 2.7ms/call in-vehicle vs
    0.02ms for a cached triton launch; device cost decided by A/B).
    dst may be a PEER symmetric-heap VA (fine-grained pcie)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask)
    tl.store(dst.to(tl.pointer_type(tl.bfloat16)) + offs, x, mask=mask)


@triton.jit(do_not_specialize=["dst"])
def kernel_scale_bf16(dst, n, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    """dst[i] *= ALPHA (bf16 in/out, fp32 math) -- AVG post-scale, exact for
    ALPHA = 1/2^k (t21 R12m).  ALPHA is constexpr (no float-scalar runtime
    arg in this das-triton launcher ABI; constant-folded)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(dst.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask)
    tl.store(dst.to(tl.pointer_type(tl.bfloat16)) + offs, x * ALPHA,
             mask=mask)


class RSRelayContext:
    pass


def create_rs_relay_context(numel, dtype, rank, num_ranks, mpi, comm,
                            k_slots=1, cells=1, subs=1, dual=None):
    """FLAT semantics: one ctx per total-element count (megatron passes
    3D [S,B,H] activations -- the ring only cares about contiguous bytes).
    Recv slots [R x shard] symmetric + ticket/ack planes + arange staging
    + lane.  dtype must be bf16 (v1).

    k_slots (v22, t21 R12w): pipeline depth -- K independent rings in
    flight.  Heap/ticket/ack planes inflate xK (every addressable unit
    becomes rank*K + slot), each slot owns a lane + done event, and an
    op only fences its slot's ack chain (waits the ack of the op K rings
    back, not the immediately previous one).  THE reason planes must be
    K-inflated: with K>1 consecutive ops run on DIFFERENT lanes, so a
    shared ticket cell would receive op n+1's value while op n's later
    rounds are still pending (different streams, no order) -- the
    reader's >=base+k test would false-fire on stale slot data.  Per-slot
    cells restore per-cell stream order because ops sharing a slot share
    a lane."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert dtype == torch.bfloat16, "rs_relay v1: bf16 only"
    assert numel % num_ranks == 0
    shard_elems = numel // num_ranks
    assert shard_elems % 2048 == 0
    shard_byt = shard_elems * 2
    R = num_ranks

    host = dcu_state()["host"]
    cells = max(1, int(cells))
    subs = max(1, int(subs))
    assert shard_elems % (2048 * subs) == 0, "sub-shard must stay 2048-multiple"
    heap_ptr = host.malloc(R * k_slots * cells * shard_byt)  # C cells per (writer, K)
    tk_ptr = host.malloc(R * k_slots * 4)
    ak_ptr = host.malloc(R * k_slots * 4)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(tk_ptr), 0, R * k_slots * 4) == 0
    assert h.hipMemset(ctypes.c_void_p(ak_ptr), 0, R * k_slots * 4) == 0
    peer_heap = [host.ptr(heap_ptr, d) for d in range(R)]
    peer_tk = [host.ptr(tk_ptr, d) for d in range(R)]
    peer_ak = [host.ptr(ak_ptr, d) for d in range(R)]

    tickets = torch.arange(MAX_TICKETS, dtype=torch.int32, device="cuda")
    h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    h.hipEventCreateWithFlags.restype = ctypes.c_int
    h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipEventRecord.restype = ctypes.c_int
    h.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_uint64]
    h.hipStreamWaitEvent.restype = ctypes.c_int

    def _ev():
        ev = ctypes.c_void_p()
        assert h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2) == 0
        return ev

    ctx = RSRelayContext()
    ctx.rank, ctx.num_ranks = rank, R
    ctx.heap_ptr, ctx.tk_ptr, ctx.ak_ptr = heap_ptr, tk_ptr, ak_ptr
    ctx.numel, ctx.dtype = numel, dtype
    ctx.shard_elems, ctx.shard_byt = shard_elems, shard_byt
    ctx.peer_heap, ctx.peer_tk, ctx.peer_ak = peer_heap, peer_tk, peer_ak
    ctx.tickets_va = tickets.data_ptr()
    ctx.k_slots = k_slots
    ctx.cells = cells
    ctx.subs = subs
    ctx.lanes = [torch.cuda.Stream() for _ in range(k_slots)]
    # v27 dual-lane: dma-lane (ctx.lanes) sends; add-lane adds/acks.  The
    # single-lane serial rate max(DMA, add) was the rig floor's binding
    # constraint (0.71x7); dual-lane drops MY emission period to the DMA
    # rate (t21 R12af).
    if dual is None:
        dual = int(os.environ.get("T21_BRS_DUAL", "0"))
    ctx.dual = bool(int(dual))
    if ctx.dual:
        ctx.lanes_add = [torch.cuda.Stream() for _ in range(k_slots)]
    ctx.lane = ctx.lanes[0]            # a2a (v4) + graph (v21) compat alias
    ctx._ev_dones = [_ev() for _ in range(k_slots)]
    ctx._ev_tail = [_ev() for _ in range(k_slots)]
    ctx._ev_done = ctx._ev_dones[0]
    ctx._ev_fork = _ev()
    ctx.round_base = 0
    # v25: ticket values enumerate sub-units t=(k-1)*S+j linearly; stride
    # covers (R-1)*S values + slack.  S=1 recovers the v3 stride R+2.
    ctx.round_stride = (R - 1) * subs + 2
    ctx._add_ck = None
    ctx._n_ops = 0
    create_rs_graph_ctx_fields(ctx, host)
    ctx.mpi, ctx.comm = mpi, comm      # F2d wrap: barrier handle on ctx
    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ctx


def finalize_rs_relay_context(ctx):
    host = dcu_state()["host"]
    finalize_rs_graph_fields(ctx, host)
    if getattr(ctx, "heap_ptr", 0):
        host.free(ctx.heap_ptr)
    if getattr(ctx, "tk_ptr", 0):
        host.free(ctx.tk_ptr)
    if getattr(ctx, "ak_ptr", 0):
        host.free(ctx.ak_ptr)
    ctx.heap_ptr = ctx.tk_ptr = 0


class _RSHandle:
    """Mimics the torch async-op handle: the caller stream already waits
    the done event (enforced in the op), so .wait() is a no-op."""

    def wait(self):
        pass

    def is_completed(self):
        return True


class _RSHandleDeferred:
    """defer_fence=True: the op did NOT fence the caller stream; .wait()
    enqueues the done-event wait on the stream current AT CONSUME TIME
    (t21 R12k -- megatron layers.py:349 fires the SP dgrad RS async, runs
    the wgrad GEMM, then handle.wait() at :454 before consuming; fencing
    at fire time serialized the ring and turned its 5.7ms into a caller
    stream hole).  Event-reuse order is safe: python-serial callers call
    wait() before the next op re-records ev_done (S13 family)."""

    def __init__(self, ev):
        self._ev = ev

    def wait(self):
        h = hip()
        sp = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = h.hipStreamWaitEvent(sp, self._ev, 0)
        assert rc == 0

    def is_completed(self):
        return False


def _launch_add(ctx, dst_va, add_va, elems=None):
    """Normal JIT launch -- this das-triton counts tl.constexpr params in
    the cached-launch C signature (12-vs-13 trap, t21v20), and BLOCK is a
    constexpr.  ~40us host per launch, 7/op: fine for v1.  elems: sub-shard
    size for v25 sub-pipelining (None = whole shard)."""
    n = elems if elems is not None else ctx.shard_elems
    assert n % 2048 == 0
    kernel_add_bf16[(n // 2048,)](dst_va, add_va, n,
                                  BLOCK=2048, num_warps=8)


_copy_ck_cache = {}


def _launch_copy(dst_va, src_va, elems):
    ck = _copy_ck_cache.get("k")
    if ck is None:
        ck = kernel_copy_bf16.warmup(1 << 40, 1 << 40, 1,
                                     grid=(1,), num_warps=8, BLOCK=2048)
        ck._init_handles()
        _copy_ck_cache["k"] = ck
    ck[(elems // 2048, 1, 1)](dst_va, src_va, elems, 2048)


_SMCOPY = os.environ.get("MGD_RR_SMCOPY", "") == "1"


def _cp(dst, src, nbytes, stream_handle=None):
    """memcpy_async-compatible front.  MGD_RR_SMCOPY=1 swaps engine
    memcpy for an SM copy kernel -- REJECTED at the S75d rig gate
    (md=8 on +-1 payload: SM stores into PEER symmetric-heap VA do not
    land; every cross-card WRITE in this codebase is engine memcpy,
    SM is cross-card read-only).  Kept gated-off for the record."""
    if _SMCOPY:
        _launch_copy(dst, src, nbytes // 2)
    else:
        memcpy_async(dst, src, nbytes, stream_handle)


def _launch_scale(dst_va, alpha, n):
    kernel_scale_bf16[(n // 2048,)](dst_va, n, ALPHA=alpha,
                                    BLOCK=2048, num_warps=8)


def _rs_relay_dual(input, out, ctx, defer_fence, out_scale, chunk_ev):
    """v27 (t21 R12af): dual-lane ring -- dma-lane (ctx.lanes[slot]) runs
    the engine memcpys + tickets + cell fences; add-lane (ctx.lanes_add)
    runs arrival waits + adds + acks + the out tail.  Cross-lane chaining
    is a rank-LOCAL gate cell xchg'd by the add-lane after each round's
    add (value base+k, tail = base+round_stride) and GTE-spun by the
    dma-lane -- NEVER a hipEvent: a wait binds the event's LATEST record,
    so a pre-armed (already-complete) event gates nothing and every send
    fires early (run-2 law: 16/16 rounds corrupt, err ~0.67).
    Single-writer-per-plane-cell preserved: tickets dma-lane, acks
    add-lane, local gate add-lane (S15).  The cell fence (C=2) is
    round-keyed k-C so the emission period is the DMA rate, not
    add+DMA.  rs_relay_op's preamble already advanced round_base/_n_ops
    -- no second increment (run-3 law: doubled base -> lag fence waits a
    ticket nobody writes -> op-1 spin)."""
    R, rank = ctx.num_ranks, ctx.rank
    h = hip()
    cur = torch.cuda.current_stream()
    cur_sp = ctypes.c_void_p(cur.cuda_stream)
    # rs_relay_op's preamble (:412/:415) ALREADY advanced round_base and
    # _n_ops for this op -- dispatch reaches us after it.  Do NOT
    # increment again: a doubled base makes the lag fence wait a ticket
    # value nobody ever writes (t21 R12af run-3 wedge, op-1 spin).
    base = ctx.round_base
    K = ctx.k_slots
    slot = (ctx._n_ops - 1) % K
    dma = ctx.lanes[slot]
    add = ctx.lanes_add[slot]
    dma_sp = ctypes.c_void_p(dma.cuda_stream)
    add_sp = ctypes.c_void_p(add.cuda_stream)
    ev_done = ctx._ev_dones[slot]
    # rank-local gate cell (see ctx comment above): add-lane is the single
    # writer, dma-lane the single GTE reader -- a REAL data gate, unlike a
    # pre-armed event whose latest record is already complete.
    gate = ctx.peer_ak[rank] + (rank * K + slot) * 4

    prev = (rank - 1) % R
    nxt = (rank + 1) % R
    lag_base = base - K * ctx.round_stride

    in_va = input.data_ptr()
    out_va = out.data_ptr()
    sb = ctx.shard_byt
    C = getattr(ctx, "cells", 1)
    nxt_cell = ctx.peer_heap[nxt] + (rank * K + slot) * C * sb
    own_cell = ctx.peer_heap[rank] + (prev * K + slot) * C * sb
    own_tk = ctx.peer_tk[rank] + (rank * K + slot) * 4
    nxt_tk = ctx.peer_tk[nxt] + (nxt * K + slot) * 4
    own_ak_from_next = ctx.peer_ak[rank] + (nxt * K + slot) * 4
    prev_ak = ctx.peer_ak[prev] + (rank * K + slot) * 4

    # fork: both lanes must not touch input before its producer completes
    if chunk_ev is None:
        rc = h.hipEventRecord(ctx._ev_fork, cur_sp)
        assert rc == 0
        rc = h.hipStreamWaitEvent(dma_sp, ctx._ev_fork, 0)
        assert rc == 0
        rc = h.hipStreamWaitEvent(add_sp, ctx._ev_fork, 0)
        assert rc == 0

    def _gate(sp, c):
        if chunk_ev is None:
            return
        rc = h.hipStreamWaitEvent(sp, chunk_ev[c], 0)
        assert rc == 0

    # ---- dma-lane: lag fence, pre-send, per-round sends ----
    with torch.cuda.stream(dma):
        if lag_base > 0:
            _wait_ticket(own_ak_from_next, lag_base + R - 1, dma, "spin")
        c0 = (rank - 1) % R
        _gate(dma_sp, c0)
        memcpy_async(nxt_cell, in_va + c0 * sb, sb, dma.cuda_stream)
        _launch_flag_xchg(nxt_tk, base + 1)
        for k in range(1, R):
            if k < R - 1:
                c = (rank - k - 1) % R
                # send k carries the ACCUMULATED chunk: gated on the
                # add-lane's round-k local gate value
                _wait_ticket(gate, base + k, dma, "spin")
                _gate(dma_sp, c)
                if k >= C:
                    _wait_ticket(own_ak_from_next, base + k - C + 1,
                                 dma, "spin")
                memcpy_async(nxt_cell + (k % C) * sb, in_va + c * sb, sb,
                             dma.cuda_stream)
                _launch_flag_xchg(nxt_tk, base + k + 1)

    # ---- add-lane: per-round arrival waits + adds + acks + tail ----
    with torch.cuda.stream(add):
        for k in range(1, R):
            c = (rank - k - 1) % R
            _gate(add_sp, c)
            _wait_ticket(own_tk, base + k, add, "spin")
            _launch_add(ctx, in_va + c * sb,
                        own_cell + ((k - 1) % C) * sb)
            _launch_flag_xchg(prev_ak, base + k)
            _launch_flag_xchg(gate, base + k)
        if out_va != in_va + rank * sb:
            _gate(add_sp, rank)
            memcpy_async(out_va, in_va + rank * sb, sb, add.cuda_stream)
        if out_scale is not None:
            _launch_scale(out_va, out_scale, ctx.shard_elems)
        # tail marker: dma-lane's ev_done must cover BOTH lanes (input
        # reuse safety -- the caller may overwrite the dgrad buffer)
        _launch_flag_xchg(gate, base + ctx.round_stride)

    with torch.cuda.stream(dma):
        _wait_ticket(gate, base + ctx.round_stride, dma, "spin")
        rc = h.hipEventRecord(ev_done, dma_sp)
        assert rc == 0

    if defer_fence:
        return _RSHandleDeferred(ev_done)
    rc = h.hipStreamWaitEvent(cur_sp, ev_done, 0)
    assert rc == 0
    return _RSHandle()


def rs_relay_op(input, out, ctx, defer_fence=False, out_scale=None,
                chunk_ev=None):
    """input [M, K] bf16 contiguous (mutated in place on non-own chunks);
    out [M/R, K] receives rank's shard of the sum.  Enqueues everything on
    the lane; the CALLER stream waits the done event before return
    (defer_fence=False).  defer_fence=True (async callers only): no fence
    here; the returned handle fences at .wait() so independent caller
    work enqueued between fire and consume (megatron's wgrad GEMM)
    executes DURING the ring instead of after it.  out_scale (DDO AVG
    class, R12m): scale the out shard by alpha on the lane before the done
    event (AVG == SUM + exact 1/2^k); when out ALIASES the input's own-shard
    view the final copy is skipped (in-place semantics already hold) and the
    scale lands in place.

    Single-slot-per-writer discipline: the reader ACKs (1-CTA sys-release
    xchg kernel, stream-ordered after its add kernel) into the writer's
    ack cell every round; the writer waits that ack before overwriting
    the slot with the next round's chunk (and before the next op's
    pre-loop send -- the ack value base+R-1 covers the whole op).
    Tickets stay ENGINE 4B writes (data-ordering law: engine data needs
    the engine's peer-L2 propagation; an SM ticket reopens the v20 tear
    window).  Acks are value-only -- kernel xchg, never engine memcpy
    (v5 ack-wedge law: tiny engine flag-write after a wait-stalled queue
    parks the host inside the enqueue).  All waits are on OWN cells
    (S16 law); tickets/acks monotonic, position-keyed, single-use per
    value (S15 law)."""
    R, rank = ctx.num_ranks, ctx.rank
    assert input.dtype == torch.bfloat16
    assert input.is_contiguous() and out.is_contiguous()
    assert input.numel() == ctx.numel
    assert out.numel() * R == ctx.numel
    _t0 = _time.perf_counter() if _HPROF else 0.0
    _tF = _tB = _t0
    h = hip()
    cur = torch.cuda.current_stream()
    cur_sp = ctypes.c_void_p(cur.cuda_stream)
    base = _advance_base(ctx)
    ctx._n_ops += 1
    K = ctx.k_slots
    slot = (ctx._n_ops - 1) % K            # same on every rank (op order is
    lane = ctx.lanes[slot]                 # the collective call sequence)
    lane_sp = ctypes.c_void_p(lane.cuda_stream)
    lane_q = lane.cuda_stream
    ev_done = ctx._ev_dones[slot]

    prev = (rank + 1) % R if getattr(ctx, "reversed", False) \
        else (rank - 1) % R
    nxt = (rank - 1) % R if getattr(ctx, "reversed", False) \
        else (rank + 1) % R
    # the pre-send may overwrite the slot the op K rings back used: it
    # fences THAT op's completion ack, not the previous op's (K=1 is the
    # v3 chain).
    lag_base = base - K * ctx.round_stride

    in_va = input.data_ptr()
    out_va = out.data_ptr()
    sb = ctx.shard_byt
    # addresses: reader's slot indexed by ITS writer (x K slots);
    # ticket/ack cells K-inflated the same way (see ctx docstring: ops
    # sharing a slot share a lane -- per-cell write order = lane order).
    C = getattr(ctx, "cells", 1)
    nxt_slot = ctx.peer_heap[nxt] + (rank * K + slot) * C * sb
    own_slot = ctx.peer_heap[rank] + (prev * K + slot) * C * sb
    own_tk = ctx.peer_tk[rank] + (rank * K + slot) * 4
    nxt_tk = ctx.peer_tk[nxt] + (nxt * K + slot) * 4
    own_ak_from_next = ctx.peer_ak[rank] + (nxt * K + slot) * 4
    prev_ak = ctx.peer_ak[prev] + (rank * K + slot) * 4

    if _HPROF:
        _rrh("pre", _time.perf_counter() - _t0)
        _tF = _time.perf_counter()
    # fork: lane must not touch input before its producer (caller-stream
    # matmul) completes.  GATED mode (v23 tile-gate probe): producer
    # publishes per-chunk events instead -- the lane waits ev[c] right
    # before touching chunk c (pre-send / add / final copy), letting early
    # ring rounds run under the producer's later chunks (laws doc #7).
    if chunk_ev is None:
        rc = h.hipEventRecord(ctx._ev_fork, cur_sp)
        assert rc == 0
        rc = h.hipStreamWaitEvent(lane_sp, ctx._ev_fork, 0)
        assert rc == 0
        if _HPROF:
            _rrh("fork", _time.perf_counter() - _tF)
            _tB = _time.perf_counter()

    def _gate(c):
        if chunk_ev is None:
            return
        rc = h.hipStreamWaitEvent(lane_sp, chunk_ev[c], 0)
        assert rc == 0

    S = getattr(ctx, "subs", 1)
    dual = bool(getattr(ctx, "dual", False)) and S == 1
    if dual:
        return _rs_relay_dual(input, out, ctx, defer_fence, out_scale,
                              chunk_ev)
    ssb = sb // S
    with torch.cuda.stream(lane):
        # pre-loop: own partial of chunk (r-1)%R -- only after next has
        # consumed the whole op that last used this slot (ack covers the
        # last sub-unit (R-1)*S; lag_base = prev op at K=1 = the v3 chain)
        if lag_base > 0:
            _tw("c_wait", _wait_ticket, own_ak_from_next,
                lag_base + (R - 1) * S, lane, "spin")
        c = (rank - 1) % R
        _gate(c)
        for j in range(S):
            _tw("c_memcpy", _cp, nxt_slot + j * ssb,
                in_va + c * sb + j * ssb, ssb, lane_q)
            _launch_flag_xchg(nxt_tk, base + j + 1)
        for k in range(1, R):
            c = (rank - k - 1) % R
            _gate(c)
            for j in range(S):
                # v25: sub-unit t=(k-1)*S+j -- per-sub tickets/acks let
                # the DMA of sub j fly under the add of sub j+1.  The
                # accumulate chain arrive->add->forward is the ring's real
                # serializer (7 x (DMA 0.49 + add 0.22) in-vehicle); C=2
                # alone changed nothing in the isolated rig.  Cell index
                # is ROUND-keyed (k%C): a cell holds a whole shard, subs
                # are offsets in it; the reuse fence is the ack of round
                # k-C's SAME sub.  S=1 recovers v24; C=1,S=1 = v3.
                t_prev = (k - 1) * S + j
                _tw("c_wait", _wait_ticket, own_tk, base + t_prev + 1,
                    lane, "spin")
                _tw("c_add", _launch_add, ctx, in_va + c * sb + j * ssb,
                    own_slot + ((k - 1) % C) * sb + j * ssb,
                    elems=ctx.shard_elems // S)
                # ack prev: stream-ordered after the add kernel; plane
                # writes are 1-CTA sys-release xchg kernels (v5 ack-wedge
                # + v20 tear laws; POLL=128 hybrid spin waiters).
                _tw("c_xchg", _launch_flag_xchg, prev_ak,
                    base + t_prev + 1)
                if k < R - 1:
                    if k >= C:
                        _wait_ticket(own_ak_from_next,
                                     base + (k - C) * S + j + 1,
                                     lane, "spin")
                    _tw("c_memcpy", _cp,
                        nxt_slot + (k % C) * sb + j * ssb,
                        in_va + c * sb + j * ssb, ssb, lane_q)
                    _tw("c_xchg", _launch_flag_xchg, nxt_tk,
                        base + k * S + j + 1)
        if out_va != in_va + rank * sb:
            _gate(rank)
            _tw("c_memcpy", _cp, out_va, in_va + rank * sb, sb,
                lane_q)
        if out_scale is not None:
            _launch_scale(out_va, out_scale, ctx.shard_elems)
        rc = h.hipEventRecord(ev_done, lane_sp)
        assert rc == 0
    if _HPROF:
        _rrh("body", _time.perf_counter() - _tB)
    if defer_fence:
        # out is lane-written; the consumer fences via the handle.  Input
        # reuse stays safe: any later write to the scratch is enqueued on
        # the caller stream after the consume-time wait.  ev_done is
        # SLOT-keyed: with K>1 the next op re-records a DIFFERENT event,
        # so the deferred wait stays bound to this op's lane (a later
        # op's done on another lane does NOT imply this one).
        return _RSHandleDeferred(ev_done)
    if _HPROF:
        _tW = _time.perf_counter()
    rc = h.hipStreamWaitEvent(cur_sp, ev_done, 0)
    assert rc == 0
    if _HPROF:
        _rrh("tail", _time.perf_counter() - _tW)
        _RRH["n"] = _RRH.get("n", 0) + 1
        _rrh_flush(rank)
    return _RSHandle()


def rs_a2a_op(input, out, ctx):
    """v4 a2a-direct RS (t21 run6 postmortem): same bytes, but every rank
    sends its chunk-j slice DIRECTLY to rank j -- latency = max(arrivals)
    instead of the ring's sum(hops) (7 sequential 67MB relays), and the
    pipeline needs NO warm-up, so the cold-start in-context op the vehicle
    actually runs hits the ~5-6ms wire floor instead of a drained ring.

    Sends and consumes are FULLY DECOUPLED (ring coupled them: round k+1's
    payload was round k's consumed data).  Send lane: [ack-waits for the
    PREVIOUS op's slots] + per-destination (data engine-copy, ticket xchg)
    on PRISTINE input slices.  Consume lane: per-sender (ticket-wait, add
    into input[rank] slice, ack xchg) + final out copy.  Same ctx/heap as
    the ring (slot indexed by SENDER; tk cell[sender] at the RECEIVER;
    ack cell[receiver] at the SENDER -- waiter's plane law throughout).

    Plane discipline identical (no 4B engine writes anywhere): all plane
    writes are sys-release xchg kernels, all waits are 128-mask spins,
    data moves only as big engine memcpys.  Tickets/acks position-keyed
    by SENDER rank (value = base + sender), monotonic per (i,j) pair."""
    R, rank = ctx.num_ranks, ctx.rank
    assert input.dtype == torch.bfloat16
    assert input.is_contiguous() and out.is_contiguous()
    assert input.numel() == ctx.numel
    assert out.numel() * R == ctx.numel
    assert ctx.k_slots == 1
    assert getattr(ctx, "cells", 1) == 1, "v24 cells: ring form only"
    assert getattr(ctx, "subs", 1) == 1, "v25 subs: ring form only"
    assert ctx.k_slots == 1, "a2a form: single-slot addressing only (v22)"
    if not hasattr(ctx, "lane_rx"):
        ctx.lane_rx = torch.cuda.Stream()
    h = hip()
    cur = torch.cuda.current_stream()
    cur_sp = ctypes.c_void_p(cur.cuda_stream)
    lane_s, lane_rx = ctx.lane, ctx.lane_rx
    lane_s_sp = ctypes.c_void_p(lane_s.cuda_stream)
    lane_rx_sp = ctypes.c_void_p(lane_rx.cuda_stream)
    lane_s_q, lane_rx_q = lane_s.cuda_stream, lane_rx.cuda_stream

    base = _advance_base(ctx)
    prev_base = base - ctx.round_stride
    ctx._n_ops += 1

    in_va = input.data_ptr()
    out_va = out.data_ptr()
    sb = ctx.shard_byt
    own_slice = in_va + rank * sb            # my chunk-rank slice (add dst)

    # fork: BOTH lanes read input (slices j != rank pristine; slice rank
    # is the add dst) -- they must not run ahead of the caller producer
    rc = h.hipEventRecord(ctx._ev_fork, cur_sp)
    assert rc == 0
    rc = h.hipStreamWaitEvent(lane_s_sp, ctx._ev_fork, 0)
    assert rc == 0
    rc = h.hipStreamWaitEvent(lane_rx_sp, ctx._ev_fork, 0)
    assert rc == 0

    with torch.cuda.stream(lane_s):
        # previous op's slots must be consumed before any peer re-sends
        # into MY slot cells -- one ack-wait per destination cell (own
        # ak plane, cell[receiver j], value = prev_base + rank (me))
        if prev_base > 0:
            for j in range(R):
                if j == rank:
                    continue
                _wait_ticket(ctx.peer_ak[rank] + j * 4, prev_base + rank,
                             lane_s, "spin")
        for j in range(R):
            if j == rank:
                continue
            memcpy_async(ctx.peer_heap[j] + rank * sb, in_va + j * sb, sb,
                         lane_s_q)
            _launch_flag_xchg(ctx.peer_tk[j] + rank * 4, base + rank)

    with torch.cuda.stream(lane_rx):
        for i in range(R):
            if i == rank:
                continue
            _wait_ticket(ctx.peer_tk[rank] + i * 4, base + i, lane_rx,
                         "spin")
            _launch_add(ctx, own_slice, ctx.peer_heap[rank] + i * sb)
            _launch_flag_xchg(ctx.peer_ak[i] + rank * 4, base + i)
        memcpy_async(out_va, own_slice, sb, lane_rx_q)
        rc = h.hipEventRecord(ctx._ev_done, lane_rx_sp)
        assert rc == 0
    rc = h.hipStreamWaitEvent(cur_sp, ctx._ev_done, 0)
    assert rc == 0
    return _RSHandle()


# ============================================================================
# v21 (t21 R12u): GRAPH-CAPTURED ring -- the host-gap law (docs
# T21_ENGINE_LAWS_20261025) says ~1ms of python between RCCL enqueues
# defuses their merge AND costs wall directly (PRBEV 0.961x). The v3 body
# costs ~35 host enqueues/op (~1.2ms). v21 replays the IDENTICAL op
# sequence from one hipGraphLaunch: 6 host calls/op (~200us), bit-equal
# plane protocol (device-counter waits keep per-round ticket values out
# of the graph).
# ============================================================================

_dev_wait_ck = {}
_dev_xchg_ck = {}


@triton.jit(do_not_specialize=["flag_ptr", "base_ptr", "off"])
def kernel_wait_flag_ge_dev(flag_ptr, base_ptr, off):
    """1-CTA spin: wait flag >= load_volatile(base_ptr) + off. The round
    base lives in DEVICE memory so a captured graph need no per-round
    scalar updates (T-series v18 'device-base wait kernel' design; POLL=128
    hybrid pattern + sys-acquire scope per the family laws)."""
    target = tl.load(base_ptr.to(tl.pointer_type(tl.int32)),
                     volatile=True) + off
    fp = flag_ptr.to(tl.pointer_type(tl.int32))
    v = tl.atomic_add(fp, 0, scope="sys")
    while v < target:
        for _ in range(128):
            v = tl.load(fp, volatile=True)
        if v < target:
            v = tl.atomic_add(fp, 0, scope="sys")


@triton.jit(do_not_specialize=["dst", "val_ptr", "off"])
def kernel_flag_xchg_dev(dst, val_ptr, off):
    """1-CTA: release-xchg (load_volatile(val_ptr) + off) into dst. Plane
    writes whose VALUE depends on the round base, graph-safe (sem/scope
    identical to kernel_flag_xchg)."""
    v = tl.load(val_ptr.to(tl.pointer_type(tl.int32)),
                volatile=True) + off
    tl.atomic_xchg(dst.to(tl.pointer_type(tl.int32)), v,
                   sem="release", scope="sys")


def _launch_wait_dev(flag_ptr, base_ptr, off):
    ck = _dev_wait_ck.get("k")
    if ck is None:
        ck = kernel_wait_flag_ge_dev.warmup(1 << 40, 1 << 40, 1,
                                            grid=(1,), num_warps=1)
        ck._init_handles()
        _dev_wait_ck["k"] = ck
    ck[(1, 1, 1)](flag_ptr, base_ptr, off)


def _launch_xchg_dev(dst, val_ptr, off):
    ck = _dev_xchg_ck.get("k")
    if ck is None:
        ck = kernel_flag_xchg_dev.warmup(1 << 40, 1 << 40, 1,
                                         grid=(1,), num_warps=1)
        ck._init_handles()
        _dev_xchg_ck["k"] = ck
    ck[(1, 1, 1)](dst, val_ptr, off)


_GRAPH_PROTOS_BOUND = False


def _bind_graph_protos(h):
    global _GRAPH_PROTOS_BOUND
    if _GRAPH_PROTOS_BOUND:
        return
    h.hipStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
    h.hipStreamBeginCapture.restype = ctypes.c_int
    h.hipStreamEndCapture.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_void_p)]
    h.hipStreamEndCapture.restype = ctypes.c_int
    h.hipGraphInstantiateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint64]
    h.hipGraphInstantiateWithFlags.restype = ctypes.c_int
    h.hipGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipGraphLaunch.restype = ctypes.c_int
    h.hipGraphDestroy.argtypes = [ctypes.c_void_p]
    h.hipGraphDestroy.restype = ctypes.c_int
    h.hipGraphExecDestroy.argtypes = [ctypes.c_void_p]
    h.hipGraphExecDestroy.restype = ctypes.c_int
    _GRAPH_PROTOS_BOUND = True


def create_rs_graph_ctx_fields(ctx, host):
    """v21 additions to a v3 ctx: local round-base counter cell (device
    int32, host.malloc'd symmetric for convenience) + per-(in_va,out_va)
    graph cache. Call from create_rs_relay_context."""
    ctx.counter_ptr = host.malloc(4)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(ctx.counter_ptr), 0, 4) == 0
    ctx.counter_va = ctx.counter_ptr          # local cell, no peer xlat
    ctx._graphs = {}
    _bind_graph_protos(h)


def _ring_body_dev(ctx, in_va, out_va, out_scale, record_done):
    """The v3 ring sequence in device-counter form, enqueued on the
    CALLER's current stream (the lane under `with torch.cuda.stream`).
    Used TWICE per new key: once for real (eager op), once under stream
    capture (recorded as graph nodes, never executed). Protocol/values
    are IDENTICAL to the v3 scalar form at the same round base."""
    R, rank = ctx.num_ranks, ctx.rank
    sb = ctx.shard_byt
    base = ctx.round_base                     # host mirror (asserts only)
    assert base + R < MAX_TICKETS
    cva = ctx.counter_va
    prev = (rank - 1) % R
    nxt = (rank + 1) % R
    nxt_slot = ctx.peer_heap[nxt] + rank * sb
    own_slot = ctx.peer_heap[rank] + prev * sb
    own_tk = ctx.peer_tk[rank] + rank * 4
    nxt_tk = ctx.peer_tk[nxt] + nxt * 4
    own_ak_from_next = ctx.peer_ak[rank] + nxt * 4
    prev_ak = ctx.peer_ak[prev] + rank * 4
    lane_q = torch.cuda.current_stream().cuda_stream

    if ctx.round_base - ctx.round_stride > 0:
        # ack of the PREVIOUS op: target = prev_base + R - 1 = B - 3
        # (stride = R + 2); constant offset vs the device base.
        _launch_wait_dev(own_ak_from_next, cva, -(ctx.round_stride - R + 1))
    c = (rank - 1) % R
    memcpy_async(nxt_slot, in_va + c * sb, sb, lane_q)
    _launch_xchg_dev(nxt_tk, cva, 1)
    for k in range(1, R):
        c = (rank - k - 1) % R
        _launch_wait_dev(own_tk, cva, k)
        _launch_add(ctx, in_va + c * sb, own_slot)
        _launch_xchg_dev(prev_ak, cva, k)
        if k < R - 1:
            _launch_wait_dev(own_ak_from_next, cva, k)
            memcpy_async(nxt_slot, in_va + c * sb, sb, lane_q)
            _launch_xchg_dev(nxt_tk, cva, k + 1)
    if out_va != in_va + rank * sb:
        memcpy_async(out_va, in_va + rank * sb, sb, lane_q)
    if out_scale is not None:
        _launch_scale(out_va, out_scale, ctx.shard_elems)
    if record_done:
        rc = hip().hipEventRecord(ctx._ev_done,
                                  ctypes.c_void_p(lane_q))
        assert rc == 0


def rs_relay_graph_op(input, out, ctx, defer_fence=False, out_scale=None):
    """v21: same semantics/protocol as rs_relay_op; the ring runs from a
    per-(in_va, out_va) captured graph (6 host calls/op vs ~35). First
    occurrence of a key runs the eager device-counter body, then records
    a capture pass (host-only); later occurrences graphLaunch. On any
    capture-API failure the key falls back to eager permanently."""
    R, rank = ctx.num_ranks, ctx.rank
    assert input.dtype == torch.bfloat16
    assert input.is_contiguous() and out.is_contiguous()
    assert input.numel() == ctx.numel
    assert out.numel() * R == ctx.numel
    assert ctx.k_slots == 1
    assert getattr(ctx, "cells", 1) == 1, "v24 cells: ring form only"
    assert getattr(ctx, "subs", 1) == 1, "v25 subs: ring form only"
    assert ctx.k_slots == 1, "graph form: single-slot addressing (v22 next)"
    h = hip()
    cur = torch.cuda.current_stream()
    cur_sp = ctypes.c_void_p(cur.cuda_stream)
    lane = ctx.lane
    lane_sp = ctypes.c_void_p(lane.cuda_stream)
    assert ctx.round_base + 2 * ctx.round_stride < _I32_CEIL, (
        "graph form: ticket values would leave int32 (~238M ops/ctx)")
    ctx.round_base += ctx.round_stride
    ctx._n_ops += 1
    in_va, out_va = input.data_ptr(), out.data_ptr()
    key = (in_va, out_va)

    rc = h.hipEventRecord(ctx._ev_fork, cur_sp)
    assert rc == 0
    rc = h.hipStreamWaitEvent(lane_sp, ctx._ev_fork, 0)
    assert rc == 0

    with torch.cuda.stream(lane):
        # round base -> device cell (plain scalar xchg; NOT in the graph)
        _launch_flag_xchg(ctx.counter_va, ctx.round_base)
        entry = ctx._graphs.get(key, "new")
        _dbg = rank == 0 and ctx._n_ops <= 6
        if _dbg:
            print(f"[rs_relay] g-op#{ctx._n_ops} entry={entry if isinstance(entry, str) else 'graph'}", flush=True)
        if entry == "eager":
            _ring_body_dev(ctx, in_va, out_va, out_scale, False)
        elif entry == "new":
            _ring_body_dev(ctx, in_va, out_va, out_scale, False)
            if _dbg:
                print(f"[rs_relay] g-op#{ctx._n_ops} eager enqueued", flush=True)
            try:
                g = ctypes.c_void_p()
                rc = h.hipStreamBeginCapture(lane_sp, 1)  # mode global
                assert rc == 0, f"BeginCapture rc={rc}"
                if _dbg:
                    print(f"[rs_relay] g-op#{ctx._n_ops} capture begun", flush=True)
                _ring_body_dev(ctx, in_va, out_va, out_scale, False)
                rc = h.hipStreamEndCapture(lane_sp, ctypes.byref(g))
                assert rc == 0 and g.value, f"EndCapture rc={rc}"
                if _dbg:
                    print(f"[rs_relay] g-op#{ctx._n_ops} capture ended", flush=True)
                ex = ctypes.c_void_p()
                rc = h.hipGraphInstantiateWithFlags(
                    ctypes.byref(ex), g, 0)
                assert rc == 0 and ex.value, f"Instantiate rc={rc}"
                ctx._graphs[key] = (g, ex)
                if _dbg:
                    print(f"[rs_relay] g-op#{ctx._n_ops} instantiated", flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"[rs_relay] graph capture failed ({e}); "
                      f"key stays eager", flush=True)
                ctx._graphs[key] = "eager"
        else:
            rc = h.hipGraphLaunch(entry[1], lane_sp)
            assert rc == 0, f"hipGraphLaunch rc={rc}"
        rc = h.hipEventRecord(ctx._ev_done, lane_sp)
        assert rc == 0
    if defer_fence:
        return _RSHandleDeferred(ctx._ev_done)
    rc = h.hipStreamWaitEvent(cur_sp, ctx._ev_done, 0)
    assert rc == 0
    return _RSHandle()


def finalize_rs_graph_fields(ctx, host):
    for v in getattr(ctx, "_graphs", {}).values():
        if isinstance(v, tuple):
            hip().hipGraphExecDestroy(v[1])
            hip().hipGraphDestroy(v[0])
    ctx._graphs = {}
    if getattr(ctx, "counter_ptr", 0):
        host.free(ctx.counter_ptr)
    ctx.counter_ptr = 0
