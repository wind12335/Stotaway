"""v18ring = v17ring with host-enqueue tables (pure python hoist).

Protocol, kernels, events, ticket/slot semantics are BYTE-IDENTICAL to
v17ring (gemm_ag_v17ring.py, untouched per the port law); the only change
is that per-op address arithmetic (_slot_va/_arr_cell recomputation),
closure construction, and ctypes wrapper objects are precomputed per ctx
(only round_base strides per op). Measured motivation (AGW3): the fused
AG enqueue floor is 2.4ms/op ~= the whole residual AG gap; the ctypes
calls themselves (~54 FFI calls/op) are the floor, the recomputation on
top of them is the hoistable part.

Select via MGD_AG_FORM=ring18. The 7 arrival GEMMs stay per-seg (event-
gated) -- batching them into one bmm would wait the full relay and break
the tile-ready arrival principle (core red line).
"""
import ctypes

import torch

from stowaway.kernels.common_ops import hip, memcpy_async
from stowaway.kernels.gemm_ag_v17ring import (
    _arr_cell,
    _slot_va,
    create_gemm_ag_v17ring_context,
    finalize_gemm_ag_v17ring_context,   # noqa: F401  (re-export)
    gemm_ag_v17ring_op,                 # noqa: F401  (fallback/debug)
)
from stowaway.kernels.gemm_ar_rd import MAX_TICKETS, _wait_ticket


def attach_tables(ctx):
    """Precompute every per-ctx-static host value the op's enqueue loop
    uses. Values match the v17ring per-call expressions exactly."""
    R, SUB, rank = ctx.num_ranks, ctx.sub, ctx.rank
    tickets_va = ctx.tickets.data_ptr()
    ctx._lane_q = ctx.ring_lane.cuda_stream
    ctx._lane_sp = ctypes.c_void_p(ctx._lane_q)
    ctx._seg_byt = ctx.sub_elems * ctx.elemsz
    ctx._own_byt = ctx.m_pr * ctx.K * ctx.elemsz
    ctx._nxt = (rank + 1) % R
    ctx._prv = (rank - 1 + R) % R
    nxt_heap = ctx.peer_heap[ctx._nxt]
    nxt_arr = ctx.peer_arr[ctx._nxt]
    # forward: slot dst (peer heap), ticket-cell dst (peer arr plane),
    # ticket-value src (my arange staging) -- all static per ctx
    ctx._fw_dst = [[_slot_va(nxt_heap, seg, si, ctx)
                    for si in range(SUB)] for seg in range(R)]
    ctx._fw_tkdst = [[_arr_cell(nxt_arr, rank, seg, si, ctx)
                      for si in range(SUB)] for seg in range(R)]
    ctx._fw_tksrc = [[tickets_va + 4 * (seg * SUB + si + 1)
                      for si in range(SUB)] for seg in range(R)]
    # relay waits: arrival cell VAs (from prv) + ticket value offsets
    ctx._wait_va = [[_arr_cell(ctx.arr_ptr, ctx._prv, seg, si, ctx)
                     for si in range(SUB)] for seg in range(R)]
    ctx._tk0 = [[seg * SUB + si + 1 for si in range(SUB)]
                for seg in range(R)]
    ctx._slot0 = [_slot_va(ctx.heap_ptr, seg, 0, ctx) for seg in range(R)]
    ctx._stg_va = [t.data_ptr() for t in ctx.staging]
    return ctx


def create_gemm_ag_v18ring_context(*args, **kwargs):
    return attach_tables(create_gemm_ag_v17ring_context(*args, **kwargs))


def gemm_ag_v18ring_op(a, b, out, ctx):
    """v17ring op, hoisted: table lookups + base adds instead of the
    per-call closures. See module docstring for the equivalence claim."""
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
    mode = ctx.wait_mode
    cur_sp = ctypes.c_void_p(cur.cuda_stream)
    lane_q = ctx._lane_q
    lane_sp = ctx._lane_sp
    b4 = 4 * (ctx.round_base + ctx.round_stride)
    ctx.round_base += ctx.round_stride          # stride + assert kept
    assert ctx.round_base + R * SUB < MAX_TICKETS
    fw_dst, fw_tkdst, fw_tksrc = ctx._fw_dst, ctx._fw_tkdst, ctx._fw_tksrc
    wait_va, tk0, slot0, stg_va = (ctx._wait_va, ctx._tk0, ctx._slot0,
                                    ctx._stg_va)
    seg_byt, own_byt = ctx._seg_byt, ctx._own_byt
    mc, wt = memcpy_async, _wait_ticket
    bt = b.t()

    def forward(seg, src_base):
        d, td, ts = fw_dst[seg], fw_tkdst[seg], fw_tksrc[seg]
        for si in range(SUB):
            mc(d[si], src_base + si * seg_byt, seg_byt, lane_q)
            mc(td[si], ts[si] + b4, 4, lane_q)

    # caller stream: own row-block GEMM (no wire), then fork the lane
    torch.matmul(a, bt, out=out[rank * m_pr:(rank + 1) * m_pr, :])
    rc = h.hipEventRecord(ctx._ev_fork, cur_sp)
    assert rc == 0
    rc = h.hipStreamWaitEvent(lane_sp, ctx._ev_fork, 0)
    assert rc == 0

    with torch.cuda.stream(lane):
        a_va = a.data_ptr()
        forward(rank, a_va)                     # ORIGIN (S39 a1)
        mc(slot0[rank], a_va, own_byt, lane_q)  # local DtoD into own slot
        for k in range(1, R):
            seg = (rank - k) % R
            wv, t0 = wait_va[seg], tk0[seg]
            for si in range(SUB):
                wt(wv[si], b4 // 4 + t0[si], lane, mode)  # = new_base + tk0
            if k < R - 1:
                forward(seg, slot0[seg])
            mc(stg_va[seg], slot0[seg], own_byt, lane_q)
            rc = h.hipEventRecord(ctx._ev_arr[seg], lane_sp)
            assert rc == 0
        rc = h.hipEventRecord(ctx._ev_join, lane_sp)
        assert rc == 0

    # caller: arrival GEMMs, each gated on its (freshly enqueued) ev_arr
    for k in range(1, R):
        seg = (rank - k) % R
        rc = h.hipStreamWaitEvent(cur_sp, ctx._ev_arr[seg], 0)
        assert rc == 0
        torch.matmul(ctx.staging[seg], bt,
                     out=out[seg * m_pr:(seg + 1) * m_pr, :])
    rc = h.hipStreamWaitEvent(cur_sp, ctx._ev_join, 0)
    assert rc == 0
