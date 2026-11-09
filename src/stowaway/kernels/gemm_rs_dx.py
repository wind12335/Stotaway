"""DX = direct-exchange ReduceScatter (opus fork, 2026-10-29).

Platform law behind it (bench/opus/opus_perm.py): with all 8 ranks pushing
at once, an engine push to ANY peer in permutation order (step j -> rank
r+j) runs ~141 GB/s/rank; there is no unicast fan-out wall for engine
pushes, so the ring's 7-hop accumulate-forward chain is unnecessary.
Each rank pushes every non-home block straight to its home in permutation
order; the home sums all R pieces ONCE (fp32 accumulate).

Two ops:
  dxv_op(input, out, ctx)            pure RS (vehicle BRS drop-in, the
                                     rs_relay_op call contract)
  dx_gemm_rs_op(a, w, out, ctx)      fused row-parallel GEMM + RS: block
                                     vendor GEMMs in permutation order on
                                     the caller stream, each block pushed
                                     to its home as soon as it is computed

Plane laws (all inherited, all kept): every wait is on an OWN cell
(hipStreamWaitValue32 GTE); tickets are engine 4B writes ordered after the
data push on the same stream; acks are SM sys-release xchg kernels launched
after the consuming kernel (launch boundary retires its reads); values are
monotone per cell; a slot is rewritten only after its reader acked.
"""
import ctypes
import time

import torch
import triton
import triton.language as tl

from stowaway.kernels.common_ops import dcu_state, hip, memcpy_async
from stowaway.kernels.gemm_ar_rd import _launch_flag_xchg, _wait_value_gte

_MAX_OPS = 1 << 22          # ticket staging table (monotone op counter)


@triton.jit(do_not_specialize=["out", "own", "slots", "slot_elems", "n"])
def k_dx_sum(out, own, slots, slot_elems, n, RANK: tl.constexpr,
             NR: tl.constexpr, ALPHA: tl.constexpr, IS_BF16: tl.constexpr,
             BLOCK: tl.constexpr):
    """out[i] = ALPHA * (own[i] + sum_{s != RANK} slots[s*slot_elems + i]),
    fp32 accumulate, one pass over all R pieces."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if IS_BF16:
        ty = tl.bfloat16
    else:
        ty = tl.float16
    acc = tl.load(own.to(tl.pointer_type(ty)) + offs, mask=m).to(tl.float32)
    sp = slots.to(tl.pointer_type(ty))
    for s in tl.static_range(NR):
        if s != RANK:
            acc += tl.load(sp + s * slot_elems + offs, mask=m).to(tl.float32)
    if ALPHA != 1.0:
        acc = acc * ALPHA
    tl.store(out.to(tl.pointer_type(ty)) + offs, acc.to(ty), mask=m)


def _launch_sum(out_va, own_va, slots_va, n, rank, R, dtype, alpha=1.0):
    k_dx_sum[(triton.cdiv(n, 4096),)](
        out_va, own_va, slots_va, n, n, RANK=rank, NR=R, ALPHA=float(alpha),
        IS_BF16=(dtype == torch.bfloat16), BLOCK=4096, num_warps=8)


def _new_event(h):
    e = ctypes.c_void_p()
    assert h.hipEventCreateWithFlags(ctypes.byref(e), 0x2) == 0
    return e


def _bind(h):
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    h.hipEventCreateWithFlags.restype = ctypes.c_int


import os as _os
_AG_STAGE_SM = _os.environ.get("OPUS_AG_STAGE", "") == "sm"


class _CAI:
    def __init__(self, ptr, nelem):
        self.__cuda_array_interface__ = {"shape": (nelem,), "typestr": "<i2",
                                         "data": (ptr, False), "version": 2}


def _heap_i16(ptr, nbytes):
    """int16 view of a heap range (SM staging: kernel copy heap->plain
    instead of an engine D2D copy that queues behind the RS pushes)."""
    return torch.as_tensor(_CAI(ptr, nbytes // 2), device="cuda")


# --------------------------------------------------------------------------
# opus fork: push-issue thread (OPUS_ISSUE=thread).  opus_vehf10/chb1: a
# 42 MB engine push into a peer heap holds the host ~600 us (~ its own copy
# time), 7 per fused call = 4.0-4.5 ms of host time per call, which starves
# the compute stream.  The thread owns the lane: it issues the lane's waits,
# pushes and tickets; ctypes releases the GIL, so the main thread keeps
# enqueueing compute.  FIFO keeps lane order; every event the thread waits
# on is recorded by the main thread BEFORE the task is submitted; per-op
# event pools of depth _IP + a host-side lag bound keep re-records safe;
# tensors read by the lane are held by the task closure until issued and
# record_stream'ed on the lane (caching-allocator reuse fence).
# --------------------------------------------------------------------------
import queue as _queue
import threading as _threading

_ISSUE_THREAD = _os.environ.get("OPUS_ISSUE", "") == "thread"
_IP = 4                     # event-pool depth = max ops the thread may lag


class _Issuer:
    def __init__(self, device):
        self.q = _queue.SimpleQueue()
        self.cv = _threading.Condition()
        self.submitted = 0
        self.done = 0
        self.err = None
        self.device = device
        self.t = _threading.Thread(target=self._run, daemon=True, name="opus-issue")
        self.t.start()

    def _run(self):
        torch.cuda.set_device(self.device)
        while True:
            fn = self.q.get()
            if fn is None:
                return
            try:
                fn()
            except BaseException as e:      # surfaced at the next submit/wait
                self.err = e
            with self.cv:
                self.done += 1
                self.cv.notify_all()

    def submit(self, fn):
        if self.err is not None:
            raise RuntimeError(f"opus issue thread failed: {self.err!r}")
        self.submitted += 1
        self.q.put(fn)
        return self.submitted

    def wait(self, ticket):
        with self.cv:
            while self.done < ticket:
                self.cv.wait()
        if self.err is not None:
            raise RuntimeError(f"opus issue thread failed: {self.err!r}")


_issuer = [None]


def _get_issuer():
    if _issuer[0] is None:
        _issuer[0] = _Issuer(torch.cuda.current_device())
    return _issuer[0]


def _issue_pool(ctx, h):
    """per-ctx event pool for the threaded path + per-slot task tickets"""
    if not hasattr(ctx, "_ip_ev"):
        ctx._ip_ev = [_new_event(h) for _ in range(_IP)]
        ctx._ip_tk = [0] * _IP


def _issue_pushes(ctx, h, cur_h, src_t, op):
    """record the fork on the caller stream, then hand the lane's ack waits +
    data pushes + tickets for `op` to the issue thread."""
    iss = _get_issuer()
    _issue_pool(ctx, h)
    slot = op % _IP
    iss.wait(ctx._ip_tk[slot])          # the task that last waited on this event
    ev = ctx._ip_ev[slot]
    assert h.hipEventRecord(ev, cur_h) == 0
    src_t.record_stream(ctx.lane)
    R, rank, bb = ctx.R, ctx.rank, ctx.blk_byt
    lane, lq = ctx.lane, ctx.lane.cuda_stream
    lane_h = ctypes.c_void_p(lq)
    tk = ctx.tickets.data_ptr() + 4 * op
    sva = src_t.data_ptr()
    p_slots, p_arr, ack = ctx.p_slots, ctx.p_arr, ctx.ack

    def task(src_keep=src_t):
        assert h.hipStreamWaitEvent(lane_h, ev, 0) == 0
        for j in range(1, R):
            d = (rank + j) % R
            if op > 1:
                _wait_value_gte(ack + 4 * d, op - 1, lane)
            memcpy_async(p_slots[d] + rank * bb, sva, bb, lq)
            memcpy_async(p_arr[d] + 4 * rank, tk, 4, lq)
    ctx._ip_tk[slot] = iss.submit(task)
    return ev


class _Handle:
    """async-op handle: wait() fences the stream current at consume time."""

    def __init__(self, ev):
        self._ev = ev

    def wait(self):
        rc = hip().hipStreamWaitEvent(
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), self._ev, 0)
        assert rc == 0

    def is_completed(self):
        return False


# --------------------------------------------------------------------------
# pure RS (vehicle BRS form): K lanes in flight, P = max(2, K) slot sets
# --------------------------------------------------------------------------
class DXRSContext:
    def __init__(self, numel, dtype, rank, R, k_slots=1):
        assert numel % R == 0 and (numel // R) % 4096 == 0
        assert dtype in (torch.bfloat16, torch.float16)
        host = dcu_state()["host"]
        h = hip()
        _bind(h)
        self.rank, self.R, self.numel, self.dtype = rank, R, numel, dtype
        self.K = max(1, int(k_slots))
        self.P = max(2, self.K)
        if self.P % self.K:
            self.P += self.K - self.P % self.K
        self.sh = numel // R
        self.sb = self.sh * torch.empty((), dtype=dtype).element_size()
        P = self.P
        self.slots = host.malloc(P * R * self.sb)
        self.arr = host.malloc(P * R * 4)
        self.ack = host.malloc(P * R * 4)
        assert self.slots and self.arr and self.ack, "dushmem_malloc returned NULL"
        assert h.hipMemset(ctypes.c_void_p(self.arr), 0, P * R * 4) == 0
        assert h.hipMemset(ctypes.c_void_p(self.ack), 0, P * R * 4) == 0
        self.p_slots = [host.ptr(self.slots, d) for d in range(R)]
        self.p_arr = [host.ptr(self.arr, d) for d in range(R)]
        self.p_ack = [host.ptr(self.ack, d) for d in range(R)]
        self.tickets = torch.arange(_MAX_OPS, dtype=torch.int32, device="cuda")
        self.lanes_s = [torch.cuda.Stream() for _ in range(self.K)]
        self.lanes_rx = [torch.cuda.Stream() for _ in range(self.K)]
        self.ev_fork = _new_event(h)
        self.ev_s = [_new_event(h) for _ in range(self.K)]
        self.ev_done = [_new_event(h) for _ in range(16)]
        self.op = 0
        torch.cuda.synchronize()
        host.barrier_all()


def dxv_op(input, out, ctx, defer_fence=False, out_scale=None, chunk_ev=None):
    """input [numel] contiguous (NOT mutated); out [numel/R] (may alias the
    input's own shard).  defer_fence=True: caller fences via handle.wait()."""
    assert input.dtype == ctx.dtype and input.is_contiguous()
    assert input.numel() == ctx.numel and out.numel() == ctx.sh
    R, rank, sb, P, K = ctx.R, ctx.rank, ctx.sb, ctx.P, ctx.K
    ctx.op += 1
    op = ctx.op
    assert op < _MAX_OPS
    p = op % P
    h = hip()
    cur = torch.cuda.current_stream()
    cur_h = ctypes.c_void_p(cur.cuda_stream)
    ln = op % K
    ls, lr = ctx.lanes_s[ln], ctx.lanes_rx[ln]
    ls_h, lr_h = ctypes.c_void_p(ls.cuda_stream), ctypes.c_void_p(lr.cuda_stream)
    iva = input.data_ptr()
    tk = ctx.tickets.data_ptr() + 4 * op
    assert h.hipEventRecord(ctx.ev_fork, cur_h) == 0
    assert h.hipStreamWaitEvent(ls_h, ctx.ev_fork, 0) == 0
    assert h.hipStreamWaitEvent(lr_h, ctx.ev_fork, 0) == 0
    for j in range(1, R):
        d = (rank + j) % R
        if op > P:          # home d consumed my op-P block (same slot set p)
            _wait_value_gte(ctx.ack + 4 * (p * R + d), op - P, ls)
        memcpy_async(ctx.p_slots[d] + (p * R + rank) * sb, iva + d * sb, sb,
                     ls.cuda_stream)
        memcpy_async(ctx.p_arr[d] + 4 * (p * R + rank), tk, 4, ls.cuda_stream)
    assert h.hipEventRecord(ctx.ev_s[ln], ls_h) == 0
    for s in range(R):
        if s != rank:
            _wait_value_gte(ctx.arr + 4 * (p * R + s), op, lr)
    assert h.hipStreamWaitEvent(lr_h, ctx.ev_s[ln], 0) == 0
    with torch.cuda.stream(lr):
        _launch_sum(out.data_ptr(), iva + rank * sb, ctx.slots + p * R * sb,
                    ctx.sh, rank, R, ctx.dtype,
                    1.0 if out_scale is None else out_scale)
        for s in range(R):
            if s != rank:
                _launch_flag_xchg(ctx.p_ack[s] + 4 * (p * R + rank), op)
    ev = ctx.ev_done[op % len(ctx.ev_done)]
    assert h.hipEventRecord(ev, lr_h) == 0
    if not defer_fence:
        assert h.hipStreamWaitEvent(cur_h, ev, 0) == 0
    return _Handle(ev)


# --------------------------------------------------------------------------
# fused row-parallel GEMM + RS (vehicle MGD_FUSED form)
# --------------------------------------------------------------------------
class DXGemmRSContext:
    """slots single-buffered + ack (the fused op is synchronous: the caller
    consumes the output right away, and the next op's GEMMs are stream-
    ordered after this op's lane join)."""

    def __init__(self, M, N, dtype, rank, R):
        assert M % R == 0
        host = dcu_state()["host"]
        h = hip()
        _bind(h)
        self.rank, self.R, self.M, self.N, self.dtype = rank, R, M, N, dtype
        self.mp = M // R
        self.elemsz = torch.empty((), dtype=dtype).element_size()
        self.blk_elems = self.mp * N
        self.blk_byt = self.blk_elems * self.elemsz
        self.slots = host.malloc(R * self.blk_byt)
        self.arr = host.malloc(R * 4)
        self.ack = host.malloc(R * 4)
        assert self.slots and self.arr and self.ack, "dushmem_malloc returned NULL"
        assert h.hipMemset(ctypes.c_void_p(self.arr), 0, R * 4) == 0
        assert h.hipMemset(ctypes.c_void_p(self.ack), 0, R * 4) == 0
        self.p_slots = [host.ptr(self.slots, d) for d in range(R)]
        self.p_arr = [host.ptr(self.arr, d) for d in range(R)]
        self.p_ack = [host.ptr(self.ack, d) for d in range(R)]
        self.partial = torch.empty(M, N, dtype=dtype, device="cuda")
        self.tickets = torch.arange(_MAX_OPS, dtype=torch.int32, device="cuda")
        self.lane = torch.cuda.Stream()
        self.ev = [_new_event(h) for _ in range(R + 1)]   # [0]=join, [1..R-1]=blocks, [R]=fork
        self.op = 0
        torch.cuda.synchronize()
        host.barrier_all()


def dx_gemm_rs_op(a, w, out, ctx):
    """out [M/R, N] = rows-block `rank` of sum_r a_r @ w_r^T.
    a [M, K_r] (my K shard), w [N, K_r] (my weight shard)."""
    M = a.shape[0]
    N = w.shape[0]
    assert M == ctx.M and N == ctx.N and a.dtype == ctx.dtype
    R, rank, mp, bb = ctx.R, ctx.rank, ctx.mp, ctx.blk_byt
    ctx.op += 1
    op = ctx.op
    assert op < _MAX_OPS
    h = hip()
    cur = torch.cuda.current_stream()
    cur_h = ctypes.c_void_p(cur.cuda_stream)
    lane, lq = ctx.lane, ctx.lane.cuda_stream
    lane_h = ctypes.c_void_p(lq)
    wt = w.t()
    tk = ctx.tickets.data_ptr() + 4 * op
    pva = ctx.partial.data_ptr()
    assert h.hipEventRecord(ctx.ev[R], cur_h) == 0           # fork
    assert h.hipStreamWaitEvent(lane_h, ctx.ev[R], 0) == 0
    # compute side: remote blocks in permutation order, own block last
    for j in range(1, R):
        d = (rank + j) % R
        torch.matmul(a[d * mp:(d + 1) * mp], wt, out=ctx.partial[d * mp:(d + 1) * mp])
        assert h.hipEventRecord(ctx.ev[j], cur_h) == 0
    own = ctx.partial[rank * mp:(rank + 1) * mp]
    torch.matmul(a[rank * mp:(rank + 1) * mp], wt, out=own)
    # lane: push each block as soon as its GEMM is done
    for j in range(1, R):
        d = (rank + j) % R
        assert h.hipStreamWaitEvent(lane_h, ctx.ev[j], 0) == 0
        if op > 1:          # home d consumed my previous block
            _wait_value_gte(ctx.ack + 4 * d, op - 1, lane)
        memcpy_async(ctx.p_slots[d] + rank * bb, pva + d * bb, bb, lq)
        memcpy_async(ctx.p_arr[d] + 4 * rank, tk, 4, lq)
    # home side: wait all arrivals, one fused sum, ack
    for s in range(R):
        if s != rank:
            _wait_value_gte(ctx.arr + 4 * s, op, cur)
    _launch_sum(out.data_ptr(), own.data_ptr(), ctx.slots, ctx.blk_elems,
                rank, R, ctx.dtype)
    for s in range(R):
        if s != rank:
            _launch_flag_xchg(ctx.p_ack[s] + 4 * rank, op)
    # join: partial[] may be overwritten by the next op only after my pushes
    assert h.hipEventRecord(ctx.ev[0], lane_h) == 0
    assert h.hipStreamWaitEvent(cur_h, ctx.ev[0], 0) == 0
    return out


# --------------------------------------------------------------------------
# fused row-parallel BACKWARD: DX all-gather of grad_out + per-block dgrad
# --------------------------------------------------------------------------
class DXAGBwdContext:
    """grad_out shards [M/R, N] gathered by permutation push into the
    peers' heap slots; the home stages each block heap->plain (GEMM
    operands must not live on the heap) and runs its dgrad rows as soon as
    the block lands; wgrad runs once on the staged full buffer."""

    def __init__(self, M, N, dtype, rank, R):
        assert M % R == 0
        host = dcu_state()["host"]
        h = hip()
        _bind(h)
        self.rank, self.R, self.M, self.N, self.dtype = rank, R, M, N, dtype
        self.mp = M // R
        self.blk_byt = self.mp * N * torch.empty((), dtype=dtype).element_size()
        self.slots = host.malloc(R * self.blk_byt)
        self.arr = host.malloc(R * 4)
        self.ack = host.malloc(R * 4)
        assert self.slots and self.arr and self.ack, "dushmem_malloc returned NULL"
        assert h.hipMemset(ctypes.c_void_p(self.arr), 0, R * 4) == 0
        assert h.hipMemset(ctypes.c_void_p(self.ack), 0, R * 4) == 0
        self.p_slots = [host.ptr(self.slots, d) for d in range(R)]
        self.p_arr = [host.ptr(self.arr, d) for d in range(R)]
        self.p_ack = [host.ptr(self.ack, d) for d in range(R)]
        self.tickets = torch.arange(_MAX_OPS, dtype=torch.int32, device="cuda")
        self.lane = torch.cuda.Stream()
        self.ev = [_new_event(h) for _ in range(2)]     # [0]=fork [1]=join
        self.op = 0
        torch.cuda.synchronize()
        host.barrier_all()


def dx_ag_bwd_op(g_sh, w, a, ctx, full=None):
    """g_sh [M/R, N] (my grad_out shard), w [N, K_r], a [M, K_r].
    Returns (grad_input [M, K_r], grad_weight [N, K_r], full [M, N])."""
    M, N = ctx.M, ctx.N
    assert g_sh.shape == (ctx.mp, N) and g_sh.dtype == ctx.dtype
    g_sh = g_sh.contiguous()
    R, rank, mp, bb = ctx.R, ctx.rank, ctx.mp, ctx.blk_byt
    ctx.op += 1
    op = ctx.op
    assert op < _MAX_OPS
    h = hip()
    cur = torch.cuda.current_stream()
    cur_h = ctypes.c_void_p(cur.cuda_stream)
    lq = ctx.lane.cuda_stream
    lane_h = ctypes.c_void_p(lq)
    tk = ctx.tickets.data_ptr() + 4 * op
    if full is None:
        full = torch.empty(M, N, dtype=ctx.dtype, device=g_sh.device)
    gi = torch.empty(M, w.shape[1], dtype=ctx.dtype, device=g_sh.device)
    if _ISSUE_THREAD:
        _issue_pushes(ctx, h, cur_h, g_sh, op)
    else:
        assert h.hipEventRecord(ctx.ev[0], cur_h) == 0
        assert h.hipStreamWaitEvent(lane_h, ctx.ev[0], 0) == 0
        for j in range(1, R):
            d = (rank + j) % R
            if op > 1:          # home d staged my previous block out of its slot
                _wait_value_gte(ctx.ack + 4 * d, op - 1, ctx.lane)
            memcpy_async(ctx.p_slots[d] + rank * bb, g_sh.data_ptr(), bb, lq)
            memcpy_async(ctx.p_arr[d] + 4 * rank, tk, 4, lq)
    full[rank * mp:(rank + 1) * mp].copy_(g_sh)
    torch.matmul(full[rank * mp:(rank + 1) * mp], w, out=gi[rank * mp:(rank + 1) * mp])
    for j in range(1, R):   # arrival order: sender r-j lands at step j
        s = (rank - j) % R
        _wait_value_gte(ctx.arr + 4 * s, op, cur)
        if _AG_STAGE_SM:
            if not hasattr(ctx, "heap16"):
                ctx.heap16 = _heap_i16(ctx.slots, R * bb)
            be = bb // 2
            full.view(torch.int16).view(-1)[s * be:(s + 1) * be].copy_(
                ctx.heap16[s * be:(s + 1) * be])
        else:
            memcpy_async(full.data_ptr() + s * bb, ctx.slots + s * bb, bb, cur.cuda_stream)
        torch.matmul(full[s * mp:(s + 1) * mp], w, out=gi[s * mp:(s + 1) * mp])
    for s in range(R):
        if s != rank:
            _launch_flag_xchg(ctx.p_ack[s] + 4 * rank, op)
    gw = full.t().matmul(a)
    # join: g_sh must outlive my pushes (threaded: closure + record_stream)
    if not _ISSUE_THREAD:
        assert h.hipEventRecord(ctx.ev[1], lane_h) == 0
        assert h.hipStreamWaitEvent(cur_h, ctx.ev[1], 0) == 0
    return gi, gw, full


# --------------------------------------------------------------------------
# fused column-parallel FORWARD: DX all-gather of the SP input + block GEMMs
# --------------------------------------------------------------------------
_AG_PLANES = int(_os.environ.get("OPUS_AG_PLANES", "1"))
# 11-01 OPUS_AG_GATE=ev: the ack waits (hipStreamWaitValue32) go on a separate
# gate stream, the lane only waits on an event recorded there. Law (T21 R12):
# an engine hipMemcpyAsync enqueued on a stream with a pending WaitValue32
# blocks the host until the wait is processed; with the waits off the lane,
# the lane holds only event barriers + copies. Default "" = original form.
_AG_GATE = _os.environ.get("OPUS_AG_GATE", "")
# 11-01 OPUS_AG_FORK=wv: cross-stream fork/join of the fused AG via local
# stream memory ops (hipStreamWriteValue32 on the producer stream +
# hipStreamWaitValue32 GTE on the consumer) instead of hipEventRecord +
# hipStreamWaitEvent. Probe (bench/opus/opus_copycap*, 2 cards): engine
# copies enqueued on a stream gated by a hipStreamWaitEvent whose event the
# GPU has not reached yet block the host (~32 queued copies, then the host
# waits for a share of the GPU lag): 40 ops x 14 copies behind a 400 ms
# sleep = 53-195 ms host in data pushes; the same with a value-gated fork
# = 1.5-6 ms. Default "" = original event form.
_AG_FORK = _os.environ.get("OPUS_AG_FORK", "")


def _write_value(ptr, val, stream_handle):
    rc = hip().hipStreamWriteValue32(ctypes.c_void_p(stream_handle), ctypes.c_void_p(ptr),
                                     ctypes.c_uint32(val), 0)
    assert rc == 0, f"hipStreamWriteValue32 rc={rc}"


class DXAGGemmContext:
    """input shards [M/R, K] pushed in permutation order into the peers'
    heap slots; a side stream stages each landed block heap->plain (GEMM
    operands must not live on the heap) in arrival order and the caller
    stream runs that block's GEMM as soon as it is staged (opus_dx_ag2
    form). Slots single-buffered + ack (acks released right after
    staging, before the GEMMs finish)."""

    def __init__(self, M, K, dtype, rank, R, planes=None):
        assert M % R == 0
        host = dcu_state()["host"]
        h = hip()
        _bind(h)
        self.rank, self.R, self.M, self.K, self.dtype = rank, R, M, K, dtype
        # 11-01 OPUS_AG_PLANES = P slot planes (default 1 = the original
        # single-buffered form). Op n uses plane n % P and its pushes wait
        # for the ack of op n-P instead of op n-1: with P=1 every fused call
        # waits for the GPU to finish the previous one, and an engine copy
        # issued after an unsatisfied stream wait blocks the host, so the
        # host lead is reset to zero at every call (host time ~ eta x layer
        # GPU time, opus_day1101/A_70b_host).
        self.P = max(1, int(planes if planes is not None else _AG_PLANES))
        P = self.P
        self.mp = M // R
        self.blk_byt = self.mp * K * torch.empty((), dtype=dtype).element_size()
        self.slots = host.malloc(P * R * self.blk_byt)
        self.arr = host.malloc(P * R * 4)
        self.ack = host.malloc(P * R * 4)
        assert self.slots and self.arr and self.ack, "dushmem_malloc returned NULL"
        assert h.hipMemset(ctypes.c_void_p(self.arr), 0, P * R * 4) == 0
        assert h.hipMemset(ctypes.c_void_p(self.ack), 0, P * R * 4) == 0
        self.p_slots = [host.ptr(self.slots, d) for d in range(R)]
        self.p_arr = [host.ptr(self.arr, d) for d in range(R)]
        self.p_ack = [host.ptr(self.ack, d) for d in range(R)]
        self.tickets = torch.arange(_MAX_OPS, dtype=torch.int32, device="cuda")
        self.lane = torch.cuda.Stream()
        self.sst = torch.cuda.Stream()
        self.ev = [_new_event(h) for _ in range(2)]      # [0]=fork [1]=lane join
        self.sev = [_new_event(h) for _ in range(R)]     # staged block i
        # local fork/join flags (OPUS_AG_FORK=wv): [0]=fork [1..R-1]=staged block i [R]=lane join
        self.fflag = torch.zeros(R + 2, dtype=torch.int32, device="cuda") if _AG_FORK == "wv" else None
        self.gst = torch.cuda.Stream() if _AG_GATE == "ev" else None
        self.gev = [_new_event(h)] if _AG_GATE == "ev" else []
        self.op = 0
        torch.cuda.synchronize()
        host.barrier_all()

    def close(self):
        """release heap slots/flags, streams and events. Collective: every
        rank must call it at the same point (the autotuner drops a path on
        all ranks identically)."""
        host = dcu_state()["host"]
        h = hip()
        torch.cuda.synchronize()
        host.barrier_all()
        for p in (self.slots, self.arr, self.ack):
            host.free(p)
        for e in self.ev + self.sev + getattr(self, "gev", []):
            h.hipEventDestroy(e)
        self.slots = self.arr = self.ack = None
        self.tickets = self.lane = self.sst = self.gst = None
        self.ev, self.sev, self.gev = [], [], []
        host.barrier_all()


_HB =_os.environ.get("OPUS_CP_TIMING", "") == "hb"
_hb_acc = {}
_hb_n = [0]


def _hb_flush():
    n = _hb_n[0]
    if not n:
        return
    tot = sum(_hb_acc.values())
    parts = " ".join(f"{k}={v / n * 1e3:.3f}" for k, v in sorted(_hb_acc.items(), key=lambda kv: -kv[1]))
    print(f"[dx-hb] calls={n} host/call ms total={tot / n * 1e3:.3f} | {parts}", flush=True)
    _hb_acc.clear()
    _hb_n[0] = 0


class _T:
    """accumulate host seconds per call category (diagnostic, OPUS_CP_TIMING=hb)"""
    __slots__ = ("k", "t")

    def __init__(self, k):
        self.k = k

    def __enter__(self):
        if _HB:
            self.t = time.perf_counter()

    def __exit__(self, *a):
        if _HB:
            _hb_acc[self.k] = _hb_acc.get(self.k, 0.0) + time.perf_counter() - self.t


def dx_ag_gemm_op(a_sh, w, ctx, need_full=False):
    """a_sh [M/R, K] (my SP shard), w [Nc, K]. Returns (out [M, Nc], full)
    where full [M, K] is the gathered input (own block filled only when
    need_full, e.g. for a wgrad that reuses it)."""
    M, K = ctx.M, ctx.K
    assert a_sh.shape == (ctx.mp, K) and a_sh.dtype == ctx.dtype
    a_sh = a_sh.contiguous()
    R, rank, mp, bb = ctx.R, ctx.rank, ctx.mp, ctx.blk_byt
    ctx.op += 1
    op = ctx.op
    assert op < _MAX_OPS
    h = hip()
    cur = torch.cuda.current_stream()
    cur_h = ctypes.c_void_p(cur.cuda_stream)
    lq = ctx.lane.cuda_stream
    lane_h = ctypes.c_void_p(lq)
    sst = ctx.sst
    sst_h = ctypes.c_void_p(sst.cuda_stream)
    tk = ctx.tickets.data_ptr() + 4 * op
    P = getattr(ctx, "P", 1)
    pl = op % P                       # slot plane of this op
    so = pl * R * bb                  # byte offset of the plane in slots
    fo = 4 * pl * R                   # byte offset of the plane in arr / ack
    with _T("alloc"):
        full = torch.empty(M, K, dtype=ctx.dtype, device=a_sh.device)
        out = torch.empty(M, w.shape[0], dtype=ctx.dtype, device=a_sh.device)
    wt = w.t()
    wv = getattr(ctx, "fflag", None) is not None and not _ISSUE_THREAD
    fb = ctx.fflag.data_ptr() if wv else 0
    if _ISSUE_THREAD:
        assert P == 1, "OPUS_ISSUE=thread supports only OPUS_AG_PLANES=1"
        with _T("issue_submit"):
            fev = _issue_pushes(ctx, h, cur_h, a_sh, op)
            assert h.hipStreamWaitEvent(sst_h, fev, 0) == 0
    else:
        with _T("event"):
            if wv:
                _write_value(fb, op, cur.cuda_stream)
                _wait_value_gte(fb, op, ctx.lane)
                _wait_value_gte(fb, op, sst)
            else:
                assert h.hipEventRecord(ctx.ev[0], cur_h) == 0
                assert h.hipStreamWaitEvent(lane_h, ctx.ev[0], 0) == 0
                assert h.hipStreamWaitEvent(sst_h, ctx.ev[0], 0) == 0
        gate = ctx.gst is not None and op > P
        if gate:                # all ack waits on the gate stream, lane waits one event
            with _T("wait_ack"):
                for j in range(1, R):
                    d = (rank + j) % R
                    _wait_value_gte(ctx.ack + fo + 4 * d, op - P, ctx.gst)
                assert h.hipEventRecord(ctx.gev[0], ctypes.c_void_p(ctx.gst.cuda_stream)) == 0
                assert h.hipStreamWaitEvent(lane_h, ctx.gev[0], 0) == 0
        for j in range(1, R):
            d = (rank + j) % R
            if op > P and not gate:   # home d staged my block of op-P out of this plane
                with _T("wait_ack"):
                    _wait_value_gte(ctx.ack + fo + 4 * d, op - P, ctx.lane)
            with _T("push_data"):
                memcpy_async(ctx.p_slots[d] + so + rank * bb, a_sh.data_ptr(), bb, lq)
            with _T("push_ticket"):
                memcpy_async(ctx.p_arr[d] + fo + 4 * rank, tk, 4, lq)
    order = [(rank - j) % R for j in range(1, R)]   # sender r-j lands at step j
    if _AG_STAGE_SM and not hasattr(ctx, "heap16"):
        ctx.heap16 = _heap_i16(ctx.slots, P * R * bb)
    full16 = full.view(torch.int16).view(-1) if _AG_STAGE_SM else None
    be = bb // 2
    hbase = so // 2                                 # plane offset in heap16 elements
    for i, s in enumerate(order):                   # side stream: land -> stage
        with _T("wait_arr"):
            _wait_value_gte(ctx.arr + fo + 4 * s, op, sst)
        if _AG_STAGE_SM:
            with torch.cuda.stream(sst):
                full16[s * be:(s + 1) * be].copy_(ctx.heap16[hbase + s * be:hbase + (s + 1) * be])
        else:
            with _T("stage_copy"):
                memcpy_async(full.data_ptr() + s * bb, ctx.slots + so + s * bb, bb, sst.cuda_stream)
        with _T("event"):
            if wv:
                _write_value(fb + 4 * (1 + i), op, sst.cuda_stream)
            else:
                assert h.hipEventRecord(ctx.sev[i], sst_h) == 0
    with _T("ack_xchg"):
        with torch.cuda.stream(sst):
            for s in order:
                _launch_flag_xchg(ctx.p_ack[s] + fo + 4 * rank, op)
    with _T("matmul"):
        torch.matmul(a_sh, wt, out=out[rank * mp:(rank + 1) * mp])
    if need_full:
        full[rank * mp:(rank + 1) * mp].copy_(a_sh)
    for i, s in enumerate(order):
        with _T("event"):
            if wv:
                _wait_value_gte(fb + 4 * (1 + i), op, cur)
            else:
                assert h.hipStreamWaitEvent(cur_h, ctx.sev[i], 0) == 0
        with _T("matmul"):
            torch.matmul(full[s * mp:(s + 1) * mp], wt, out=out[s * mp:(s + 1) * mp])
    # join: a_sh must outlive my pushes (threaded path: closure ref +
    # record_stream on the lane instead of a stream join)
    if not _ISSUE_THREAD:
        with _T("event"):
            if wv:
                _write_value(fb + 4 * R, op, lq)
                _wait_value_gte(fb + 4 * R, op, cur)
            else:
                assert h.hipEventRecord(ctx.ev[1], lane_h) == 0
                assert h.hipStreamWaitEvent(cur_h, ctx.ev[1], 0) == 0
    if _HB:
        _hb_n[0] += 1
        if _hb_n[0] >= 256:
            _hb_flush()
    return out, full
