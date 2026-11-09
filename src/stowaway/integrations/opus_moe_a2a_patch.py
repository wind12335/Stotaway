"""opus fork: Megatron MoE all-to-all data plane -> DX permutation engine push
(OPUS_MOE_SF=1).

Megatron's MoEAlltoAllTokenDispatcher already syncs the split sizes to the
host before every EP all-to-all (token_dispatcher.py "before_ep_alltoall"),
so the data plane can be sized on the host.  This patch replaces
megatron.core.transformer.moe.token_dispatcher.all_to_all (the _AllToAll
autograd function, forward AND backward) with:

  sender   for j = 1..R-1, d = (rank + j) % R  (permutation order, no incast):
             [wait ack_d >= round-1]  my previous chunk in d's region was
                                      compacted by d (own ack cell, remote-
                                      written by d)
             engine push input[off_d : off_d + n_d] -> d's region[rank]
             4B ticket (round) -> d's arrival cell [rank], same stream
  self     local copy of my own chunk straight into the output
  receiver wait my 7 arrival cells >= round (hipStreamWaitValue32, own cells)
           compact region[s][0:n_s] -> output (contiguous by source, the
           exact all_to_all_single layout)
           4B ack (round) -> every sender's ack cell [rank], same stream
Everything runs on the CURRENT (compute) stream, in program order; every
rank issues the same sequence of all-to-alls (collective), so per-buffer
round counters agree across ranks.  One buffer set per row width (tokens
bf16 x hidden, probs fp32), region capacity = max rows any source can send
to one destination (OPUS_MOE_CAP_ROWS, default tokens/rank x topk).
Verification mode OPUS_MOE_VERIFY=N: the first N calls also run RCCL and
compare bit-exactly (pure data movement).
"""
import ctypes
import os

import torch

_CAP_ROWS = int(os.environ.get("OPUS_MOE_CAP_ROWS", "0"))
_VERIFY = int(os.environ.get("OPUS_MOE_VERIFY", "0"))
_eng = {}
_bufs = {}
_stats = {"calls": 0, "verified": 0, "verify_fail": 0, "fallback": 0}


def _bootstrap():
    import pydushmem.host as pHost
    from stowaway.kernels.common_ops import init_dcu
    mpi, world = pHost.mpi_init()
    r = pHost.mpi_comm_rank(mpi, world)
    n = pHost.mpi_comm_size(mpi, world)
    assert r == torch.distributed.get_rank() and n == torch.distributed.get_world_size()
    init_dcu(mpi, world, r, n)
    _eng.update({"rank": r, "size": n})
    if r == 0:
        print(f"[opus-moe] engine world up: R={n} cap_rows={_CAP_ROWS}", flush=True)


class _Buf:
    def __init__(self, row_bytes):
        from stowaway.kernels.common_ops import dcu_state, hip
        host = dcu_state()["host"]
        h = hip()
        h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        R, rank = _eng["size"], _eng["rank"]
        self.R, self.rank, self.rb = R, rank, row_bytes
        self.cap = _CAP_ROWS
        self.region_bytes = self.cap * row_bytes
        self.region = host.malloc(R * self.region_bytes)
        self.arr = host.malloc(R * 4)
        self.ack = host.malloc(R * 4)
        assert self.region and self.arr and self.ack, "dushmem_malloc returned NULL"
        for p in (self.arr, self.ack):
            assert h.hipMemset(ctypes.c_void_p(p), 0, R * 4) == 0
        self.p_region = [host.ptr(self.region, d) for d in range(R)]
        self.p_arr = [host.ptr(self.arr, d) for d in range(R)]
        self.p_ack = [host.ptr(self.ack, d) for d in range(R)]
        self.tickets = torch.arange(1 << 22, dtype=torch.int32, device="cuda")
        self.round = 0
        torch.cuda.synchronize()
        host.barrier_all()
        if rank == 0:
            print(f"[opus-moe] buffer row_bytes={row_bytes} cap_rows={self.cap} "
                  f"heap {R * self.region_bytes / 2**20:.0f}MB", flush=True)


def _get_buf(row_bytes):
    if row_bytes not in _bufs:
        _bufs[row_bytes] = _Buf(row_bytes)
    return _bufs[row_bytes]


def _splits(x, rows, R):
    if x is None:
        assert rows % R == 0
        return [rows // R] * R
    return [int(v) for v in x]


def _dx_a2a(input, output_split_sizes, input_split_sizes):
    from stowaway.kernels.common_ops import memcpy_async
    from stowaway.kernels.gemm_ar_rd import _wait_value_gte
    R, rank = _eng["size"], _eng["rank"]
    input = input.contiguous()
    rows_in = input.shape[0]
    rb = input.numel() * input.element_size() // max(1, rows_in) if rows_in else 0
    in_sp = _splits(input_split_sizes, rows_in, R)
    out_sp = _splits(output_split_sizes, rows_in, R)
    out = input.new_empty([sum(out_sp)] + list(input.shape[1:]))
    if rb == 0:
        return out
    b = _get_buf(rb)
    assert max(in_sp) <= b.cap and max(out_sp) <= b.cap, \
        f"opus moe a2a: split {max(max(in_sp), max(out_sp))} > cap {b.cap}"
    b.round += 1
    rnd = b.round
    cur = torch.cuda.current_stream()
    q = cur.cuda_stream
    in_off = [0] * R
    for d in range(1, R):
        in_off[d] = in_off[d - 1] + in_sp[d - 1]
    out_off = [0] * R
    for s in range(1, R):
        out_off[s] = out_off[s - 1] + out_sp[s - 1]
    iva, ova = input.data_ptr(), out.data_ptr()
    tk = b.tickets.data_ptr() + 4 * rnd
    for j in range(1, R):
        d = (rank + j) % R
        if rnd > 1:
            _wait_value_gte(b.ack + 4 * d, rnd - 1, cur)
        if in_sp[d]:
            memcpy_async(b.p_region[d] + rank * b.region_bytes, iva + in_off[d] * rb,
                         in_sp[d] * rb, q)
        memcpy_async(b.p_arr[d] + 4 * rank, tk, 4, q)
    if in_sp[rank]:
        memcpy_async(ova + out_off[rank] * rb, iva + in_off[rank] * rb, in_sp[rank] * rb, q)
    for s in range(R):
        if s == rank:
            continue
        _wait_value_gte(b.arr + 4 * s, rnd, cur)
        if out_sp[s]:
            memcpy_async(ova + out_off[s] * rb, b.region + s * b.region_bytes, out_sp[s] * rb, q)
    for s in range(R):
        if s != rank:
            memcpy_async(b.p_ack[s] + 4 * rank, tk, 4, q)
    return out


_MODE = os.environ.get("OPUS_MOE_SF_MODE", "engine")      # engine | sm
_k = {}


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


def _kernels():
    if _k:
        return _k

    @triton.jit
    def k_push(prm, me, NE_BLK: tl.constexpr, BLOCK: tl.constexpr):
        """program (d, b): copy chunk d (int16 elements) block b to its
        destination.  prm[d] = (src_addr, dst_addr, n_elems, 0) int64."""
        d = tl.program_id(0)
        b = tl.program_id(1)
        src = tl.load(prm + d * 4).to(tl.pointer_type(tl.int16))
        dst = tl.load(prm + d * 4 + 1).to(tl.pointer_type(tl.int16))
        n = tl.load(prm + d * 4 + 2)
        base = b.to(tl.int64) * NE_BLK
        for off in range(0, NE_BLK, BLOCK):
            offs = base + off + tl.arange(0, BLOCK)
            m = offs < n
            tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)

    @triton.jit
    def k_flags(cells, val, R: tl.constexpr):
        """release-xchg val into R remote cells (addresses in cells[0:R],
        0 = skip)"""
        for i in tl.static_range(R):
            a = tl.load(cells + i)
            if a != 0:
                tl.atomic_xchg(a.to(tl.pointer_type(tl.int32)), val, sem="release", scope="sys")

    _k.update(push=k_push, flags=k_flags)
    return _k


_NE_BLK = 1 << 16          # int16 elements per program (128 KB)


_RING = 256                 # pinned param slots (host is re-synced by Megatron every layer)
_ring = {}


def _param_slot(rows):
    """async H2D of a small int64 param block through a pinned ring: no
    pageable copy (which would drain the stream on every call)."""
    if not _ring:
        _ring["h"] = torch.zeros(_RING, 8, 4, dtype=torch.int64, pin_memory=True)
        _ring["d"] = torch.zeros(_RING, 8, 4, dtype=torch.int64, device="cuda")
        _ring["i"] = 0
    i = _ring["i"] % _RING
    _ring["i"] += 1
    hs = _ring["h"][i]
    hs.zero_()
    hs[:len(rows)] = torch.tensor(rows, dtype=torch.int64)
    ds = _ring["d"][i]
    ds.copy_(hs, non_blocking=True)
    return ds


def _sm_copy(prm_rows, max_elems):
    k = _kernels()
    prm = _param_slot(prm_rows)
    grid = (len(prm_rows), max(1, triton.cdiv(max_elems, _NE_BLK)))
    k["push"][grid](prm, 0, NE_BLK=_NE_BLK, BLOCK=4096, num_warps=8)


def _sm_flags(cells_t, val):
    k = _kernels()
    k["flags"][(1,)](cells_t, val, R=cells_t.numel(), num_warps=1)


def _dx_a2a_sm(input, output_split_sizes, input_split_sizes):
    from stowaway.kernels.gemm_ar_rd import _wait_value_gte
    R, rank = _eng["size"], _eng["rank"]
    input = input.contiguous()
    rows_in = input.shape[0]
    rb = input.numel() * input.element_size() // max(1, rows_in) if rows_in else 0
    in_sp = _splits(input_split_sizes, rows_in, R)
    out_sp = _splits(output_split_sizes, rows_in, R)
    out = input.new_empty([sum(out_sp)] + list(input.shape[1:]))
    if rb == 0:
        return out
    assert rb % 2 == 0
    b = _get_buf(rb)
    assert max(in_sp) <= b.cap and max(out_sp) <= b.cap
    b.round += 1
    rnd = b.round
    cur = torch.cuda.current_stream()
    in_off = [0] * R
    for d in range(1, R):
        in_off[d] = in_off[d - 1] + in_sp[d - 1]
    out_off = [0] * R
    for s in range(1, R):
        out_off[s] = out_off[s - 1] + out_sp[s - 1]
    iva, ova = input.data_ptr(), out.data_ptr()
    if rnd > 1:                         # every dest compacted my previous chunk
        for d in range(R):
            if d != rank:
                _wait_value_gte(b.ack + 4 * d, rnd - 1, cur)
    rows = []
    for d in range(R):
        dst = (ova + out_off[rank] * rb) if d == rank else (b.p_region[d] + rank * b.region_bytes)
        rows.append([iva + in_off[d] * rb, dst, in_sp[d] * rb // 2, 0])
    _sm_copy(rows, max(r[2] for r in rows))
    if not hasattr(b, "arr_cells"):
        b.arr_cells = torch.tensor([b.p_arr[d] + 4 * rank if d != rank else 0 for d in range(R)],
                                   dtype=torch.int64, device="cuda")
        b.ack_cells = torch.tensor([b.p_ack[s] + 4 * rank if s != rank else 0 for s in range(R)],
                                   dtype=torch.int64, device="cuda")
    _sm_flags(b.arr_cells, rnd)
    for s in range(R):
        if s != rank:
            _wait_value_gte(b.arr + 4 * s, rnd, cur)
    crow = []
    for s in range(R):
        if s != rank:
            crow.append([b.region + s * b.region_bytes, ova + out_off[s] * rb, out_sp[s] * rb // 2, 0])
    _sm_copy(crow, max(r[2] for r in crow))
    _sm_flags(b.ack_cells, rnd)
    return out


class _OpusAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, use_nccl_stream=False):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.use_nccl_stream = use_nccl_stream
        if group.size() == 1:
            return input
        _stats["calls"] += 1
        out = (_dx_a2a_sm if _MODE == "sm" else _dx_a2a)(input, output_split_sizes, input_split_sizes)
        if _stats["verified"] < _VERIFY:
            ref = input.new_empty(out.shape)
            torch.distributed.all_to_all_single(
                ref, input.contiguous(), output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes, group=group)
            ok = torch.equal(out, ref)
            t = torch.tensor([0.0 if ok else 1.0], device="cuda")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
            _stats["verified"] += 1
            if t.item() != 0:
                _stats["verify_fail"] += 1
                if _eng.get("rank", 0) == 0:
                    print(f"[opus-moe] VERIFY FAIL call {_stats['calls']} shape {tuple(out.shape)}", flush=True)
            if _stats["verified"] == _VERIFY and _eng.get("rank", 0) == 0:
                print(f"[opus-moe] verify done: {_stats}", flush=True)
        return out

    @staticmethod
    def backward(ctx, *grad_output):
        return (None,
                _OpusAllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes,
                                    ctx.output_split_sizes, ctx.use_nccl_stream),
                None, None, None)


def _opus_all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None,
                     use_nccl_stream=False):
    assert group is not None
    if not _eng:
        # lazy bootstrap: the EP all-to-all is collective over the world
        # group here (EP = world), so every rank reaches this in lockstep
        global _CAP_ROWS
        if group.size() == torch.distributed.get_world_size():
            if not _CAP_ROWS:
                from megatron.training import get_args
                args = get_args()
                _CAP_ROWS = args.micro_batch_size * args.seq_length * args.moe_router_topk
            _bootstrap()
            import atexit
            atexit.register(lambda: _eng.get("rank", 0) == 0 and print(f"[opus-moe] exit stats {_stats}", flush=True))
    if group.size() != _eng.get("size"):
        _stats["fallback"] += 1
        from megatron.core.tensor_parallel.mappings import _AllToAll
        return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes, use_nccl_stream)
    return _OpusAllToAll.apply(group, input_, output_split_sizes_, input_split_sizes, use_nccl_stream)


def install():
    global _CAP_ROWS
    import megatron.training.training as _mt
    from megatron.core.transformer.moe import token_dispatcher as td
    td.all_to_all = _opus_all_to_all
    _orig_train = _mt.train

    def train(*a, **kw):
        global _CAP_ROWS
        if not _eng:
            from megatron.training import get_args
            args = get_args()
            if not _CAP_ROWS:
                _CAP_ROWS = args.micro_batch_size * args.seq_length * args.moe_router_topk
            _bootstrap()
        r = _orig_train(*a, **kw)
        if _eng.get("rank", 0) == 0:
            print(f"[opus-moe] stats {_stats}", flush=True)
        return r

    _mt.train = train
    print("[opus-moe] token_dispatcher.all_to_all -> DX engine all-to-all", flush=True)
