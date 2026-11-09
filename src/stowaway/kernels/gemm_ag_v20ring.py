"""v20ring = v18ring with DEVICE-BASE tickets + CUDA-graph-captured relay
(t21 plan-1: attack the AG ~122ms/step python-enqueue floor).

AGW3 decomposition: the fused AG op's host floor is ~2.4ms/op ~= 54 ctypes
FFI calls (memcpy_async / hipStreamWaitValue32 / event ops) + 8 torch
matmuls.  The device path is already at the fabric ceiling (verdict 12),
so the only remaining lever is the ENQUEUE cost.  v20 captures the lane's
k-loop (waits + seg forwards + staging DtoD + ev_arr/ev_join records,
~40 of the 54 FFI calls) into ONE CUDA graph per ctx and replays it per
op -- 1 hipGraphLaunch instead of 40 host calls.

Two substitutions make the graph round-invariant (a captured graph
replays BAKED kernel args; v18's args embed round_base):
  1. wait: v18 bakes expected = round_base + pos as a hipStreamWaitValue32
     immediate.  v20's spin kernel kernel_wait_flag_dge computes
     expected = load32(base_cell) + pos on device -- the hybrid
     volatile/sys-atomic pattern of kernel_wait_flag_ge (S13 law).
  2. forward ticket: v18 writes the 4B value by engine memcpy FROM the
     arange staging at address tickets_va + 4*(base+pos) -- the SRC
     address embeds base.  v20 keeps the ENGINE write but reads the value
     from a per-ctx ticket-source cell table tsrc[R*SUB] whose CONTENT a
     1-launch fill kernel refreshes each round (it also writes the base
     cell, folding the bump).  (POC round 1: an SM xchg release for the
     ticket tore 2/800 rank-rounds at N768 -- a sys release orders the
     SM's OWN stores, not the ENGINE's in-flight data at the peer's L2;
     engine-data + engine-ticket on one stream is the proven ordering,
     S12/S15 law family.  tsrc restores it verbatim inside the graph.)

The per-round DEVICE bump (_launch_flag_xchg(base_cell, round_base), an
ABSOLUTE idempotent write) rides the lane before the origin forward, so
every rank's base cell advances in lockstep with the shared host counter
(SP lockstep: one op = one bump on every rank).  Ordering: bump >
graph-waits on the same lane stream (launch boundary = visibility), and
a rank's base cannot advance past round N+1 before its round-N waits
pass (single serialized lane) -- the S15 position-keyed single-use
ticket law keeps its pairing proof with >= semantics.

What stays OUTSIDE the graph (per-call VAs, unchanged from v18): own
GEMM + fork, the ORIGIN forward from `a` (tsrc-cell ticket memcpys --
value space identical), the own-slot DtoD, the arrival GEMMs + gates.
FFI count/op: 17 vs v18's ~54.

Select via MGD_AG_FORM=ring20; S20AG_GRAPH=0 runs the identical relay
body LIVE (no capture) -- the correctness bridge (v20-nograph must be
bit-exact vs v18 before the graph is trusted).  Port law: NEW file,
v17/v18/v19 and all cores untouched.
"""
import ctypes
import os
import time

import torch
import triton  # noqa: F401  (parity with the family files)
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
from stowaway.kernels.gemm_ar_rd import MAX_TICKETS

_graph_on = os.environ.get("S20AG_GRAPH", "1") != "0"


#   POLL=4 (fast wake ~us) re-opened the fabric arrival-order tear
#   (4/2400 rank-rounds, t21v20 run4): under HSA_FORCE_FINE_GRAIN_PCIE the
#   33.6MB data copy splits across engine channels and the 4B ticket can
#   reach the waiter's L2 first.  POLL=128 (60-130us wake overshoot)
#   masks it -- 0/4800 (run3).  v18's HW hipStreamWaitValue32 wake delay
#   is the same mask, which is why v18 never tore.  The vehicle's SPOT
#   verify/repair (MGD_AG_VSAMPLE) is the functional safety net on top.
#   NOTE: the 128 below is HARDCODED on purpose -- this das-triton build
#   counts tl.constexpr params in the cached-launch C signature
#   (12-vs-13 args), so a POLL knob via tl.constexpr breaks the
#   ck[(1,1,1)] launch path.


@triton.jit(do_not_specialize=["cell", "base_ptr", "off"])
def kernel_wait_flag_dge(cell, base_ptr, off):
    """Graph-safe relay wait: spin until *cell >= *base_ptr + off.  The
    expected value comes from DEVICE memory (the ctx base cell), so a
    captured replay needs no per-round args.  Body = kernel_wait_flag_ge
    (volatile batches + sys atomic probe at the L2 coherence point); the
    cell is written by a PEER (release xchg), the base cell is OWN."""
    cp = cell.to(tl.pointer_type(tl.int32))
    bp = base_ptr.to(tl.pointer_type(tl.int32))
    expected = tl.load(bp, volatile=True) + off
    v = tl.atomic_add(cp, 0, scope="sys")
    while v < expected:
        # poll batch 128 = the tear mask (see module comment above); do
        # NOT lower without a 300-round tear soak at BOTH N shapes.
        for _ in range(128):
            v = tl.load(cp, volatile=True)
        if v < expected:
            v = tl.atomic_add(cp, 0, scope="sys")


@triton.jit(do_not_specialize=["dst", "base_va", "base_val"])
def kernel_fill_tsrc(dst, base_va, base_val, SUB: tl.constexpr):
    """One launch per round: refresh the LOCAL ticket-source cells
    tsrc[p] = base + pos(p) (p = seg*SUB + si) and write the device base
    cell (folded bump).  Local plain stores -- consumed only by
    same-stream engine reads (the graph's 4B ticket memcpys) and the
    wait kernels, both stream-ordered after this launch boundary."""
    p = tl.program_id(0)
    if p == 0:
        tl.store(base_va.to(tl.pointer_type(tl.int32)), base_val)
    seg = p // SUB
    si = p % SUB
    tl.store(dst.to(tl.pointer_type(tl.int32)) + p,
             base_val + seg * SUB + si + 1)


_wait_dge_cache = {}


def _ck_wait(cell, base, off):
    """Cached launch of kernel_wait_flag_dge (int64-VA warmup trick --
    heap VAs exceed int32, see _wait_ticket).  Launches on the CALLER's
    current stream: the lane (live) or the captured lane (graph)."""
    ck = _wait_dge_cache.get("k")
    if ck is None:
        ck = kernel_wait_flag_dge.warmup(1 << 40, 1 << 40, 1,
                                         grid=(1,), num_warps=1)
        ck._init_handles()
        _wait_dge_cache["k"] = ck
    ck[(1, 1, 1)](cell, base, off)


def _fill_tsrc(ctx):
    """Launch kernel_fill_tsrc on the CALLER's current stream (the lane):
    refresh tsrc cells + base cell for this round.  Normal JIT launch
    (runs OUTSIDE any capture; the cached-launch path cannot pass the
    SUB constexpr on this triton build -- 12-vs-13 args, t21v20 P1)."""
    kernel_fill_tsrc[(ctx.num_ranks * ctx.sub,)](
        ctx._tsrc_va, ctx._base_va, ctx.round_base, SUB=ctx.sub,
        num_warps=1)


def _relay_body(ctx):
    """The graph-static k-loop: v18's lane relay with the two device-base
    substitutions.  Issues onto the CURRENT stream (the lane); under
    capture these become graph nodes, live they execute immediately."""
    h = hip()
    R, SUB, rank = ctx.num_ranks, ctx.sub, ctx.rank
    lane_q = ctx._lane_q
    lane_sp = ctx._lane_sp
    fw_dst, fw_tkdst = ctx._fw_dst, ctx._fw_tkdst
    wait_va, tk0 = ctx._wait_va, ctx._tk0
    slot0, stg_va = ctx._slot0, ctx._stg_va
    seg_byt, own_byt = ctx._seg_byt, ctx._own_byt
    base_va = ctx._base_va
    tsrc_va = ctx._tsrc_va
    mc = memcpy_async
    for k in range(1, R):
        seg = (rank - k) % R
        wv, t0 = wait_va[seg], tk0[seg]
        for si in range(SUB):
            _ck_wait(wv[si], base_va, t0[si])
        if k < R - 1:
            d, td = fw_dst[seg], fw_tkdst[seg]
            for si in range(SUB):
                mc(d[si], slot0[seg] + si * seg_byt, seg_byt, lane_q)
                mc(td[si], tsrc_va + 4 * (seg * SUB + si), 4, lane_q)
        mc(stg_va[seg], slot0[seg], own_byt, lane_q)
        rc = h.hipEventRecord(ctx._ev_arr[seg], lane_sp)
        assert rc == 0, f"arr record rc={rc}"
    rc = h.hipEventRecord(ctx._ev_join, lane_sp)
    assert rc == 0, f"join record rc={rc}"


_POOL = max(2, int(os.environ.get("S20AG_POOL", "4")))
#   EXEC-POOL LAW: HIP blocks hipGraphLaunch of a graphExec that is still
#   in flight -- in the vehicle (AG ops ~50ms apart behind layer work)
#   each launch serialized on the PREVIOUS replay's completion = +50ms/op
#   host block (AG20 round-1: enqueue med 51.8ms, 0.942x; probe hid it as
#   enqueue~=wall "backpressure").  K instances round-robin decouple.

_graph_api_bound = False


def _bind_graph_api(h):
    """One-time ctypes prototypes for the raw HIP graph API (torch's
    CUDAGraph wrapper exposes only ONE exec per capture -- the pool needs
    raw hipGraphInstantiate on the captured graph handle)."""
    global _graph_api_bound
    if _graph_api_bound:
        return
    for fn, at in (
        ("hipStreamBeginCapture", [ctypes.c_void_p, ctypes.c_uint]),
        ("hipStreamEndCapture",
         [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]),
        ("hipGraphInstantiate",
         [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]),
        ("hipGraphInstantiateWithFlags",
         [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
          ctypes.c_uint64]),
        ("hipGraphLaunch", [ctypes.c_void_p, ctypes.c_void_p]),
        ("hipGraphDestroy", [ctypes.c_void_p]),
    ):
        f = getattr(h, fn, None)
        if f is not None:
            f.argtypes = at
            f.restype = ctypes.c_int
    _graph_api_bound = True


def _capture_relay(ctx):
    """One-time per ctx: warm both kernels (compile/alloc is illegal
    mid-capture), raw-capture _relay_body on the lane (rc=900
    default-stream law: lane is a side stream), instantiate _POOL execs.
    Round-robin replay in the op (EXEC-POOL LAW above)."""
    h = hip()
    _bind_graph_api(h)
    lane_q = ctx._lane_q
    _ck_wait(ctx._base_va, ctx._base_va, 0)    # no-op spin (*b >= *b+0)
    _fill_tsrc(ctx)                            # tsrc cells + base cell
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g = ctypes.c_void_p()
    rc = h.hipStreamBeginCapture(lane_q, 1)    # 1 = ThreadLocal
    assert rc == 0, f"BeginCapture rc={rc}"
    try:
        _relay_body(ctx)
        rc = h.hipStreamEndCapture(lane_q, ctypes.byref(g))
        assert rc == 0 and g.value, f"EndCapture rc={rc} g={g.value}"
    except BaseException:
        gg = ctypes.c_void_p()
        h.hipStreamEndCapture(lane_q, ctypes.byref(gg))  # discard
        raise
    exs = []
    for _ in range(_POOL):
        ex = ctypes.c_void_p()
        rc = h.hipGraphInstantiateWithFlags(ctypes.byref(ex), g, 0)
        if rc != 0 or not ex.value:
            rc = h.hipGraphInstantiate(ctypes.byref(ex), g)
        assert rc == 0 and ex.value, f"instantiate rc={rc}"
        exs.append(ex)
    ctx._graph = g                              # keep the graph handle
    ctx._gexecs = exs
    ctx._gi = 0
    ctx._capture_ms = (time.perf_counter() - t0) * 1e3


def create_gemm_ag_v20ring_context(*args, **kwargs):
    ctx = attach_tables(create_gemm_ag_v17ring_context(*args, **kwargs))
    # device ticket-base cell: plain OWN torch memory (peers never read
    # it -- each rank's wait reads ITS OWN cell, writers read THEIRS;
    # lockstep host ops keep them equal).  Absolue bump per op.
    ctx._base_cell = torch.zeros(1, dtype=torch.int32, device="cuda")
    ctx._base_va = ctx._base_cell.data_ptr()
    # local ticket-source cells [R*SUB]: tsrc[seg*SUB+si] holds this
    # round's value (base + pos); the graph's 4B ticket writes are ENGINE
    # memcpys from these FIXED addresses (content refreshed per round).
    ctx._tsrc = torch.zeros(ctx.num_ranks * ctx.sub, dtype=torch.int32,
                            device="cuda")
    ctx._tsrc_va = ctx._tsrc.data_ptr()
    ctx._graph = None
    ctx._gexecs = []
    ctx._gi = 0
    ctx._capture_ms = -1.0
    return ctx


def gemm_ag_v20ring_op(a, b, out, ctx):
    """v18 op with the k-loop replayed from a CUDA graph (or live with
    the device-base kernels when S20AG_GRAPH=0).  See module docstring;
    caller-stream geometry and the origin step are byte-identical to
    v18."""
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
    cur_sp = ctypes.c_void_p(cur.cuda_stream)
    lane_q = ctx._lane_q
    lane_sp = ctx._lane_sp
    ctx.round_base += ctx.round_stride          # shared host counter
    assert ctx.round_base + R * SUB < MAX_TICKETS
    fw_dst, fw_tkdst = ctx._fw_dst, ctx._fw_tkdst
    slot0 = ctx._slot0
    tsrc_va = ctx._tsrc_va
    seg_byt, own_byt = ctx._seg_byt, ctx._own_byt
    mc = memcpy_async
    bt = b.t()

    # caller stream: own row-block GEMM, then fork the lane (v18 body)
    torch.matmul(a, bt, out=out[rank * m_pr:(rank + 1) * m_pr, :])
    rc = h.hipEventRecord(ctx._ev_fork, cur_sp)
    assert rc == 0
    rc = h.hipStreamWaitEvent(lane_sp, ctx._ev_fork, 0)
    assert rc == 0

    with torch.cuda.stream(lane):
        a_va = a.data_ptr()
        # refresh tsrc cells + base cell (one launch; the folded bump) --
        # the graph's engine ticket reads and wait kernels are
        # stream-ordered after this launch boundary.
        _fill_tsrc(ctx)
        # ORIGIN: rides the per-call `a` VA -- OUTSIDE any capture; the
        # ticket is the same ENGINE 4B memcpy, from tsrc cells (same
        # value space as v18: base + seg*SUB + si + 1).
        d, td = fw_dst[rank], fw_tkdst[rank]
        for si in range(SUB):
            mc(d[si], a_va + si * seg_byt, seg_byt, lane_q)
            mc(td[si], tsrc_va + 4 * (rank * SUB + si), 4, lane_q)
        mc(slot0[rank], a_va, own_byt, lane_q)  # local DtoD into own slot
        if _graph_on:
            if not ctx._gexecs:
                _capture_relay(ctx)             # first op only
            rc = h.hipGraphLaunch(ctx._gexecs[ctx._gi], lane_q)
            assert rc == 0, f"hipGraphLaunch rc={rc}"
            ctx._gi = (ctx._gi + 1) % len(ctx._gexecs)
        else:
            _relay_body(ctx)

    # caller: arrival GEMMs, each gated on its (freshly recorded) ev_arr
    for k in range(1, R):
        seg = (rank - k) % R
        rc = h.hipStreamWaitEvent(cur_sp, ctx._ev_arr[seg], 0)
        assert rc == 0
        torch.matmul(ctx.staging[seg], bt,
                     out=out[seg * m_pr:(seg + 1) * m_pr, :])
    rc = h.hipStreamWaitEvent(cur_sp, ctx._ev_join, 0)
    assert rc == 0
