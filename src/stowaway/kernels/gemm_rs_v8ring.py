"""GEMM reduce-scatter via RING RELAY (v8ring) -- S49 verdict GO, first real
op. One ring lane per rank; classic ring RS with running sums:

  ring seg s <-> output block (s-1+R)%R
  step k (0..R-2) on rank r: forward seg (r-k)%R.
    k==0 ORIGIN: src = own partial rows of block (r-1+R)%R (S39 a1 law --
      never wait for an arrival nobody originates).
    k>=1: wait arrival of seg (r-k)%R (received from prev at step k-1),
      add own partial of block ((r-k)-1+R)%R in-place on the heap slot,
      forward.
  final: wait seg (r+1)%R == output block r, add own partial of block r,
  write out.  No acks: v1 relies on per-round host sync (bench harness);
  slots are single-use within a round, ticket values monotonic across
  rounds (S49 ack-on-data-lane wedge law deferred).

Each segment payload is split into SUB ~14.7MB subchunks with per-subchunk
arrival tickets (the empirically clean engine-chunk size of v7 CR=256).

Laws honored: tickets REMOTE-written into the waiter's plane (S15);
fork/join events are hipEventDisableTiming (S43/S47: enable_timing events
on copy streams are poison); ORIGIN step; arrival-chained forwards;
adds are compute kernels on the ring lane, gated by GTE waits.
"""
import ctypes
import os

import torch
import triton
import triton.language as tl

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

# ~14.7 MB subchunk = v7 CR=256 x 28672 x 2B, the proven-clean engine size
_SUB_TARGET_BYTES = 14 * 1024 * 1024


@triton.jit(do_not_specialize=["dst", "src", "add", "n"])
def kernel_add_va(dst, src, add, n, BLOCK: tl.constexpr):
    """dst[i] = src[i] + add[i], elementwise fp16 on raw device VAs."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src.to(tl.pointer_type(tl.float16)) + offs, mask=mask)
    y = tl.load(add.to(tl.pointer_type(tl.float16)) + offs, mask=mask)
    tl.store(dst.to(tl.pointer_type(tl.float16)) + offs, x + y, mask=mask)


class GemmRSv8RingContext:
    pass


def create_gemm_rs_v8ring_context(max_M, N, output_dtype, rank, num_ranks,
                                  mpi, comm, wait_mode=None):
    """Recv heap [R segs x SUB sub] + arrival plane [src x seg x sub] +
    ring lane. Caller must sync between rounds (v1: no acks)."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert max_M % num_ranks == 0 and N % 128 == 0
    m_pr = max_M // num_ranks
    assert m_pr % 64 == 0
    elemsz = torch.empty((), dtype=output_dtype).element_size()

    sub = int(os.environ.get("S8R_SUB", 0))
    if sub <= 0:
        sub = 1
        for cand in (8, 4, 2):
            if m_pr % cand == 0 and m_pr // cand * N * elemsz <= _SUB_TARGET_BYTES:
                sub = cand
                break
    assert m_pr % sub == 0, "S8R_SUB must divide m_per_rank"
    sub_rows = m_pr // sub
    sub_elems = sub_rows * N
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
    ev_fork = ctypes.c_void_p()
    ev_join = ctypes.c_void_p()
    for ev in (ev_fork, ev_join):
        assert h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2) == 0

    ctx = GemmRSv8RingContext()
    ctx.rank = rank
    ctx.num_ranks = num_ranks
    ctx.max_M = max_M
    ctx.N = N
    ctx.m_pr = m_pr
    ctx.sub = sub
    ctx.sub_rows = sub_rows
    ctx.sub_elems = sub_elems
    ctx.elemsz = elemsz
    ctx.output_dtype = output_dtype
    ctx.heap_ptr = heap_ptr
    ctx.arr_ptr = arr_ptr
    ctx.peer_heap = peer_heap
    ctx.peer_arr = peer_arr
    ctx.tickets = tickets
    ctx.partial = torch.empty(max_M, N, dtype=output_dtype, device="cuda")
    ctx.ring_lane = torch.cuda.Stream()
    ctx.wait_mode = wait_mode or os.environ.get("RD_WAIT_MODE", "stream")
    ctx.round_base = 0
    ctx.round_stride = num_ranks * sub + 8
    ctx._ev_fork = ev_fork
    ctx._ev_join = ev_join

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ctx


def finalize_gemm_rs_v8ring_context(ctx):
    s = dcu_state()
    for p in (ctx.heap_ptr, ctx.arr_ptr):
        if p:
            s["host"].free(p)
    ctx.heap_ptr = ctx.arr_ptr = 0


def _slot_va(heap_base, seg, sub_idx, ctx):
    return heap_base + (seg * ctx.sub + sub_idx) * ctx.sub_elems * ctx.elemsz


def _arr_cell(arr_base, src, seg, sub_idx, ctx):
    return arr_base + ((src * ctx.num_ranks + seg) * ctx.sub + sub_idx) * 4


def gemm_rs_v8ring_op(a, b, out, ctx, component="full"):
    """out = rows block `rank` of sum_r a_r @ b_r^T, via ring relay.
    component="gemm": local partial GEMM only (for A/B isolation)."""
    M, K = a.shape
    N = b.shape[0]
    rank, R = ctx.rank, ctx.num_ranks
    m_pr, SUB = ctx.m_pr, ctx.sub
    assert M == ctx.max_M and N == ctx.N and M % R == 0
    assert out.shape == (m_pr, N)

    cur = torch.cuda.current_stream()
    h = hip()
    if component == "gemm":                   # local partial only
        torch.matmul(a, b.t(), out=ctx.partial)
        return
    nxt = (rank + 1) % R
    prv = (rank - 1 + R) % R

    if component == "full":
        ctx.round_base += ctx.round_stride
        assert ctx.round_base + R * SUB < MAX_TICKETS
    base = ctx.round_base
    tickets_va = ctx.tickets.data_ptr()

    # local partial product on the caller stream
    torch.matmul(a, b.t(), out=ctx.partial)

    # ring lane forks AFTER the partial GEMM (compute-stream event; S47 law)
    lane = ctx.ring_lane
    rc = h.hipEventRecord(ctx._ev_fork, ctypes.c_void_p(cur.cuda_stream))
    assert rc == 0, f"fork hipEventRecord rc={rc}"
    rc = h.hipStreamWaitEvent(ctypes.c_void_p(lane.cuda_stream),
                              ctx._ev_fork, 0)
    assert rc == 0, f"fork hipStreamWaitEvent rc={rc}"

    mode = ctx.wait_mode
    partial_va = ctx.partial.data_ptr()
    lane_h = lane.cuda_stream
    n_blks = triton.cdiv(ctx.sub_elems, 8192)

    def ticket(seg, sub_idx):
        return base + seg * SUB + sub_idx + 1

    def add(dst_va, src_va, add_va):
        kernel_add_va[(n_blks,)](dst_va, src_va, add_va, ctx.sub_elems,
                                 BLOCK=8192, num_warps=8)

    def wait_and_add(seg, dst_base):
        """wait prev's arrival for every subchunk of seg; add own block of
        the output segment seg carries ((seg-1+R)%R). dst_base == the seg's
        heap area while relaying, or `out` for the final segment."""
        blk = (seg - 1 + R) % R
        own_va = partial_va + blk * m_pr * N * ctx.elemsz
        for si in range(SUB):
            _wait_ticket(_arr_cell(ctx.arr_ptr, prv, seg, si, ctx),
                         ticket(seg, si), lane, mode)
            add(dst_base + si * ctx.sub_elems * ctx.elemsz,
                _slot_va(ctx.heap_ptr, seg, si, ctx),
                own_va + si * ctx.sub_elems * ctx.elemsz)

    def forward(seg, src_base):
        for si in range(SUB):
            memcpy_async(_slot_va(ctx.peer_heap[nxt], seg, si, ctx),
                         src_base + si * ctx.sub_elems * ctx.elemsz,
                         ctx.sub_elems * ctx.elemsz, lane_h)
            memcpy_async(_arr_cell(ctx.peer_arr[nxt], rank, seg, si, ctx),
                         tickets_va + 4 * ticket(seg, si), 4, lane_h)

    with torch.cuda.stream(lane):
        for k in range(R - 1):
            seg = (rank - k) % R
            if k == 0:
                # ORIGIN: own partial slice of the block this seg carries
                blk = (seg - 1 + R) % R
                forward(seg, partial_va + blk * m_pr * N * ctx.elemsz)
            else:
                wait_and_add(seg, _slot_va(ctx.heap_ptr, seg, 0, ctx))
                forward(seg, _slot_va(ctx.heap_ptr, seg, 0, ctx))
        # final segment: seg (rank+1)%R carries output block rank; ONE add
        # of the own contribution completes it (S50 fix: the old explicit
        # extra add double-counted -> rel ~1.3 on every rank)
        seg_fin = (rank + 1) % R
        wait_and_add(seg_fin, out.data_ptr())

    # join: caller waits the ring lane (disable-timing event; S47 law)
    rc = h.hipEventRecord(ctx._ev_join, ctypes.c_void_p(lane.cuda_stream))
    assert rc == 0, f"join hipEventRecord rc={rc}"
    rc = h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream),
                              ctx._ev_join, 0)
    assert rc == 0, f"join hipStreamWaitEvent rc={rc}"
