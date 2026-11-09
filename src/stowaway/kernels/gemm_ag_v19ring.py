"""v19ring = v18ring with SM-copy relay (t21 verdict-9 follow-up).

hipprof verdict 9 (10-24): in the fused step the D2D DMA engine is the
feed bottleneck -- each relay engine copy ~2.0ms/block vs block GEMM
0.79ms ("compute outruns feed"), leaving 708ms/step of stream holes, and
the engine runs ~17GB/s vs RCCL's SM copies (~84GB/s bus).  v19 replaces
every DATA relay copy (slot forwards / local DtoD / staging) with a
triton SM copy kernel launched on the ring lane; 4-byte ticket cells stay
on the engine (costless, keeps ticket law intact).

Protocol/tickets/events BYTE-IDENTICAL to v17/v18 (port law: NEW file,
cores untouched).  Pointer style follows the core consumer kernel
(gemm_ag_tr.py): raw int64 VAs as kernel args, cast inside via
tl.pointer_type.  Select via MGD_AG_FORM=ring19; S19AG_SM=1 opts INTO the SM relay
(default 0 = engine = v18 semantics; verdict 12: SM relay is slower,
kept for the record).
"""
import ctypes
import os

import torch
import triton
import triton.language as tl

from stowaway.kernels.common_ops import hip, memcpy_async
from stowaway.kernels.gemm_ag_v17ring import (
    create_gemm_ag_v17ring_context,
    finalize_gemm_ag_v17ring_context,   # noqa: F401  (re-export)
    gemm_ag_v17ring_op,                 # noqa: F401  (fallback/debug)
)
from stowaway.kernels.gemm_ag_v18ring import (
    attach_tables,
    gemm_ag_v18ring_op,                 # noqa: F401  (fallback/A-B)
)
from stowaway.kernels.gemm_ar_rd import MAX_TICKETS, _wait_ticket

_SM_RELAY = os.environ.get("S19AG_SM", "0") != "0"   # default OFF: t21_v19_probe
#   measured SM relay +2.3ms/op SLOWER (fabric 17GB/s/rank ceiling binds
#   both transports; SM adds kernel overhead) -- verdict 12, 10-24.


@triton.jit(do_not_specialize=["n"])
def _k_copy_i64(dst, src, n, BLOCK: tl.constexpr):
    """SM copy of n int64 elements between two raw int64 VAs (symmetric
    heap / staging tensors).  Same pointer idiom as the core consumer
    kernel (gemm_ag_tr.py flags_base/ws_ptr)."""
    d = dst.to(tl.pointer_type(tl.int64))
    s = src.to(tl.pointer_type(tl.int64))
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    msk = offs < n
    tl.store(d + offs, tl.load(s + offs, mask=msk, other=0), mask=msk)


_COPY_BLOCK = 4096     # 32KB per program


def _sm_copy(dst_va, src_va, nbytes):
    """Launch the SM copy on the CALLER's current stream (inside the op
    that is the ring lane). nbytes must be 8-aligned (heap slots are)."""
    n8 = nbytes >> 3
    assert (nbytes & 7) == 0
    _k_copy_i64[(triton.cdiv(n8, _COPY_BLOCK),)](
        dst_va, src_va, n8, BLOCK=_COPY_BLOCK)


def create_gemm_ag_v19ring_context(*args, **kwargs):
    return attach_tables(create_gemm_ag_v17ring_context(*args, **kwargs))


def gemm_ag_v19ring_op(a, b, out, ctx):
    """v18ring op with the relay's DATA copies on SM kernels.  See module
    docstring; every address/table/stride is the v18 expression."""
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
    sm = _sm_copy if _SM_RELAY else (lambda d_, s_, n_: mc(d_, s_, n_, lane_q))
    bt = b.t()

    def forward(seg, src_base):
        d, td, ts = fw_dst[seg], fw_tkdst[seg], fw_tksrc[seg]
        for si in range(SUB):
            sm(d[si], src_base + si * seg_byt, seg_byt)
            mc(td[si], ts[si] + b4, 4, lane_q)   # ticket cell: engine, 4B

    # caller stream: own row-block GEMM (no wire), then fork the lane
    torch.matmul(a, bt, out=out[rank * m_pr:(rank + 1) * m_pr, :])
    rc = h.hipEventRecord(ctx._ev_fork, cur_sp)
    assert rc == 0
    rc = h.hipStreamWaitEvent(lane_sp, ctx._ev_fork, 0)
    assert rc == 0

    with torch.cuda.stream(lane):
        a_va = a.data_ptr()
        forward(rank, a_va)                     # ORIGIN (S39 a1)
        sm(slot0[rank], a_va, own_byt)          # local copy into own slot
        for k in range(1, R):
            seg = (rank - k) % R
            wv, t0 = wait_va[seg], tk0[seg]
            for si in range(SUB):
                wt(wv[si], b4 // 4 + t0[si], lane, mode)  # = new_base + tk0
            if k < R - 1:
                forward(seg, slot0[seg])
            sm(stg_va[seg], slot0[seg], own_byt)   # staging copy
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
