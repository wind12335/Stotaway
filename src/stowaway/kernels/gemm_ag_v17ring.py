"""GEMM + ALL-GATHER (Megatron-SP form) via INPUT ring relay + per-arrival
vendor GEMM chunks (v17ag) -- the v9ring recipe (S49: chunked vendor GEMM
overlapped with ring relay) mirrored onto the AG side. NEW FILE per the
port law; v11ring (R==2 output exchange) and v12ring (vLLM form: input
replicated, output N-gather) are left untouched.

Semantics (Megatron SP ColumnParallelLinear): `a` is MY row-shard
[m_pr, K] (sequence-parallel), `b` is MY weight shard [N, K], and
    out [M, N] = cat_r(a_r) @ b^T      (full M on every rank)
i.e. out's row-block s = a_s @ b^T. Rank r can compute only its own
row-block (it alone has a_r), so the ROW-BLOCK INPUTS travel the ring and
each arrival triggers a vendor GEMM chunk on the caller stream -- the
overlap RCCL stock cannot do (stock: finish AG, then one GEMM).

Ring choreography (v12ag geometry, payload swapped to inputs):
  caller stream : own GEMM a@b^T -> out[rank block] (no wire), then for
                  each arrival seg: (event-gated) staging[seg] @ b^T ->
                  out[seg block].  Fork AFTER the own GEMM (S47/S51).
  ring lane     : ORIGIN -- forward MY a (seg=rank; a is a call arg,
                  ready at fork; no GEMM gate needed, unlike v12ag).
                  step k=1..R-1: seg (rank-k)%R arrives -> DtoD the heap
                  slot into staging[seg] (vendor GEMM needs a torch
                  tensor; the symmetric heap is raw VA), record
                  ev_arr[seg] (disable-timing), forward the slot (last
                  arrival is not forwarded).
Laws honored: v8ring ticket plane [src][seg][sub] + single-use slots +
per-round stride; tickets REMOTE-written into the waiter's plane (S15);
disable-timing events only (S43/S47); ORIGIN step (S39 a1); SUB auto =
v8ring's ~14.7MB engine-proven subchunk. The heap lays out as [M, K]
row-major for ANY SUB (sub splits rows within a seg), so the SPOT
verify/repair kernels from the i1 path work verbatim with ws_ptr=heap.
"""
import ctypes
import os

import torch
import triton  # noqa: F401  (kept for parity with the family files)

from stowaway.kernels.common_ops import (
    dcu_state,
    hip,
    init_dcu,
    memcpy_async,
)
from stowaway.kernels.gemm_ar_rd import (
    MAX_TICKETS,
    _wait_ticket,
)

# BIG-subchunk auto: t21_v17_probe at the vehicle shape (M65536 K4096
# bf16 R=8) measured SUB=2 (33.6MB pushes) at N3072 7.1ms vs SUB=8
# (v8ring's 14.7MB target) 9.6ms -- bigger engine chunks, fewer ticket
# round-trips, higher effective relay bandwidth. Clean at 0 spot hits.
_SUB_TARGET_BYTES = 36 * 1024 * 1024


# t21 R12s: zero-copy torch view over a raw device VA (the symmetric
# heap). The staging DtoD bridged heap->torch purely because vendor GEMM
# needs a tensor; 448MiB/op of engine copies = HALF the v17 engine bytes
# (AGRES2-agonly ledger: +869ms/it for 8 fused ops). __cuda_array_interface__
# typestr '<u2' (numpy has no bf16) + .view(torch.bfloat16) rebinds the
# interpretation with zero copies -- GEMM then reads the heap rows
# directly (LOCAL memory once the ring lands them; same bytes, same
# vendor dispatch => bit-identical output vs the staging path).
_HEAPVIEW = os.environ.get("MGD_AG_HEAPVIEW", "") == "1"


class _RawDevView:
    """Fake cuda-array-interface object: torch.as_tensor builds a cuda
    tensor over ptr with NO copy and NO provenance check."""

    def __init__(self, ptr, shape):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": "<u2",            # uint16 window; bf16 via .view()
            "data": (ptr, False),        # (ptr, read-only)
            "version": 2,
        }


def _dev_view_bf16(ptr, shape):
    return torch.as_tensor(_RawDevView(ptr, shape)).view(torch.bfloat16)



class GemmAGv17RingContext:
    pass


def create_gemm_ag_v17ring_context(max_M, K, input_dtype, output_dtype,
                                   rank, num_ranks, mpi, comm,
                                   wait_mode=None):
    """Recv heap [R segs x SUB subs, slot = sub_rows*K] + arrival plane
    [src x seg x sub] + ring lane + per-seg staging tensors. The heap is
    [M, K] row-major (ws_ptr-compatible with the SPOT verify kernels).
    Caller must sync between rounds (v1: no acks, v8ring contract)."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert max_M % num_ranks == 0
    m_pr = max_M // num_ranks
    assert m_pr % 64 == 0
    elemsz = torch.empty((), dtype=output_dtype).element_size()

    sub = int(os.environ.get("S17AG_SUB", 0))
    if sub <= 0:
        sub = 1
        for cand in (2, 4, 8):          # fewest/biggest subs first (probe)
            if m_pr % cand == 0 \
                    and m_pr // cand * K * elemsz <= _SUB_TARGET_BYTES:
                sub = cand
                break
    assert m_pr % sub == 0, "S17AG_SUB must divide m_per_rank"
    sub_rows = m_pr // sub
    sub_elems = sub_rows * K
    seg_bytes = sub_elems * elemsz

    s = dcu_state()
    host = s["host"]
    heap_ptr = host.malloc(num_ranks * sub * seg_bytes)
    arr_bytes = 4 * num_ranks * num_ranks * sub
    arr_ptr = host.malloc(arr_bytes)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(arr_ptr), 0, arr_bytes) == 0
    peer_heap = [host.ptr(heap_ptr, d) for d in range(num_ranks)]
    peer_arr = [host.ptr(arr_ptr, d) for d in range(num_ranks)]

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

    ev_arr = [_ev() for _ in range(num_ranks)]

    ctx = GemmAGv17RingContext()
    ctx.rank = rank
    ctx.num_ranks = num_ranks
    ctx.max_M = max_M
    ctx.K = K
    ctx.m_pr = m_pr
    ctx.sub = sub
    ctx.sub_rows = sub_rows
    ctx.sub_elems = sub_elems
    ctx.elemsz = elemsz
    ctx.input_dtype = input_dtype
    ctx.output_dtype = output_dtype
    ctx.heap_ptr = heap_ptr            # [M, K] row-major (ws_ptr layout)
    ctx.ws_ptr = heap_ptr              # SPOT verify/repair compatibility
    ctx.arr_ptr = arr_ptr
    ctx.peer_heap = peer_heap
    ctx.peer_arr = peer_arr
    ctx.tickets = tickets
    if _HEAPVIEW:
        # arrival GEMMs read the heap rows in place (R12s): no staging
        # DtoD at all. Slot layout is seg-major contiguous [m_pr, K]
        # (_slot_va), so each view is dense/contiguous -- same strides
        # the staging tensors had => same vendor GEMM dispatch.
        ctx.staging = [_dev_view_bf16(heap_ptr + seg * m_pr * K * elemsz,
                                      (m_pr, K))
                       for seg in range(num_ranks)]
    else:
        ctx.staging = [torch.empty(m_pr, K, dtype=output_dtype,
                                   device="cuda") for _ in range(num_ranks)]
    ctx.ring_lane = torch.cuda.Stream()
    ctx.wait_mode = wait_mode or os.environ.get("RD_WAIT_MODE", "stream")
    ctx.round_base = 0
    ctx.round_stride = num_ranks * sub + 8
    ctx._ev_fork = _ev()
    ctx._ev_join = _ev()
    ctx._ev_arr = ev_arr

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ctx


def finalize_gemm_ag_v17ring_context(ctx):
    s = dcu_state()
    for p in (ctx.heap_ptr, ctx.arr_ptr):
        if p:
            s["host"].free(p)
    ctx.heap_ptr = ctx.arr_ptr = 0


def _slot_va(heap_base, seg, sub_idx, ctx):
    return heap_base + (seg * ctx.sub + sub_idx) * ctx.sub_elems * ctx.elemsz


def _arr_cell(arr_base, src, seg, sub_idx, ctx):
    return arr_base + ((src * ctx.num_ranks + seg) * ctx.sub + sub_idx) * 4


def gemm_ag_v17ring_op(a, b, out, ctx):
    """out [M, N] = cat_r(a_r) @ b^T; a my [m_pr, K] row-shard, b my
    [N, K] weight shard, out full [M, N] on every rank."""
    m_pr, K = a.shape
    N = b.shape[0]
    rank, R = ctx.rank, ctx.num_ranks
    SUB = ctx.sub
    M = m_pr * R
    assert M == ctx.max_M and K == ctx.K and m_pr == ctx.m_pr
    assert b.shape == (N, K) and out.shape == (M, N)
    assert a.dtype == ctx.input_dtype and b.dtype == ctx.input_dtype
    assert a.stride(1) == 1 and a.stride(0) == K
    assert b.stride(1) == 1 and b.stride(0) == K

    cur = torch.cuda.current_stream()
    h = hip()
    lane = ctx.ring_lane
    nxt = (rank + 1) % R
    prv = (rank - 1 + R) % R
    elemsz = ctx.elemsz
    mode = ctx.wait_mode

    ctx.round_base += ctx.round_stride
    assert ctx.round_base + R * SUB < MAX_TICKETS
    base = ctx.round_base
    tickets_va = ctx.tickets.data_ptr()

    def ticket(seg, si):
        return base + seg * SUB + si + 1

    def forward(seg, src_base):
        for si in range(SUB):
            memcpy_async(_slot_va(ctx.peer_heap[nxt], seg, si, ctx),
                         src_base + si * ctx.sub_elems * elemsz,
                         ctx.sub_elems * elemsz, lane.cuda_stream)
            memcpy_async(_arr_cell(ctx.peer_arr[nxt], rank, seg, si, ctx),
                         tickets_va + 4 * ticket(seg, si), 4,
                         lane.cuda_stream)

    # caller stream: own row-block GEMM (no wire)...
    torch.matmul(a, b.t(), out=out[rank * m_pr:(rank + 1) * m_pr, :])

    # ...then fork the ring lane (S47/S51 laws). The fork MUST be recorded
    # before anything that transitively waits on the lane (the arrival
    # GEMMs): a fork point placed after them would order the lane behind
    # work that itself waits on the lane = deadlock.
    rc = h.hipEventRecord(ctx._ev_fork, ctypes.c_void_p(cur.cuda_stream))
    assert rc == 0, f"fork hipEventRecord rc={rc}"
    rc = h.hipStreamWaitEvent(ctypes.c_void_p(lane.cuda_stream),
                              ctx._ev_fork, 0)
    assert rc == 0, f"fork hipStreamWaitEvent rc={rc}"

    # The lane's ENTIRE relay sequence is host-enqueued BEFORE the caller
    # waits ev_arr: ev_arr is a reused per-round event, and a wait binds
    # to the most recent RECORD enqueued at wait-enqueue time -- enqueueing
    # the caller waits first would bind them to the PREVIOUS round's
    # records (a stale-release race), or to nothing in round 1.
    with torch.cuda.stream(lane):
        # ORIGIN: my a rides seg `rank` (a is a call arg -- ready at fork)
        forward(rank, a.data_ptr())
        # ...and lands in MY OWN heap seg too (LOCAL DtoD): without it the
        # heap's seg==rank rows are never written (a ring seg never returns
        # to its origin), and the SPOT verify/repair kernels -- which read
        # the heap as the complete [M, K] gather -- would false-positive
        # every tile of the own row-block (t21_v17_probe: rel==0 but hits
        # ~= own-block tiles x 4).
        memcpy_async(_slot_va(ctx.heap_ptr, rank, 0, ctx), a.data_ptr(),
                     ctx.m_pr * K * elemsz, lane.cuda_stream)
        # relay: seg (rank-k)%R arrives at step k (physically from prv --
        # every ring arrival does). Wait the whole seg, forward it
        # batched (sub-interleaved forwards broke the engine's copy
        # pipelining -- t21_v17_probe3 was 1ms slower), then the staging
        # DtoD.
        for k in range(1, R):
            seg = (rank - k) % R
            for si in range(SUB):
                _wait_ticket(_arr_cell(ctx.arr_ptr, prv, seg, si, ctx),
                             ticket(seg, si), lane, mode)
            if k < R - 1:
                forward(seg, _slot_va(ctx.heap_ptr, seg, 0, ctx))
            if not _HEAPVIEW:
                memcpy_async(ctx.staging[seg].data_ptr(),
                             _slot_va(ctx.heap_ptr, seg, 0, ctx),
                             ctx.m_pr * K * elemsz, lane.cuda_stream)
            rc = h.hipEventRecord(ctx._ev_arr[seg],
                                  ctypes.c_void_p(lane.cuda_stream))
            assert rc == 0, f"arr record rc={rc}"
        # join record rides the lane (disable-timing; S47 law)
        rc = h.hipEventRecord(ctx._ev_join, ctypes.c_void_p(lane.cuda_stream))
        assert rc == 0, f"join hipEventRecord rc={rc}"

    # caller: the arrival GEMMs, each gated on its (freshly enqueued)
    # ev_arr record, then the join wait (record enqueued above -- the
    # binding is correct; it is also redundant, the arrival GEMMs
    # transitively wait the whole relay anyway)
    for k in range(1, R):
        seg = (rank - k) % R
        rc = h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream),
                                  ctx._ev_arr[seg], 0)
        assert rc == 0, f"arr gate rc={rc}"
        torch.matmul(ctx.staging[seg], b.t(),
                     out=out[seg * m_pr:(seg + 1) * m_pr, :])
    rc = h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream),
                              ctx._ev_join, 0)
    assert rc == 0, f"join hipStreamWaitEvent rc={rc}"
