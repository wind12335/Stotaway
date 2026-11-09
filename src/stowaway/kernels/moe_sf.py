"""Host-sync-free MoE layer for expert parallelism (EP = R ranks), forward + backward.

  dispatch (tokens -> expert ranks) --> sort by local expert ON DEVICE --> grouped expert MLP
  (moe_expert_mlp.py, device-side counts) --> return rows to their source (by tag) --> weighted sum

No step reads a count on the host: there is no count all-to-all, no .tolist(), no barrier.
Data plane = SM stores into the peer's symmetric heap (F21/F23: 1.6-2.5x RCCL varlen in isolation).
Control plane = 4-byte flags written with sys-scope release atomics, waited on with
hipStreamWaitValue32 (stream waits only: in-kernel cross-card spinning deadlocks on this stack, F35).
Buffers are reused every call; every buffer has an ACK plane (receiver -> writer) so a writer never
overwrites rows the receiver has not consumed yet. Round counters are monotone; all ranks issue the
same sequence of calls (same layer order in forward / recompute / backward), so rounds line up.

Two directions, one context:
  to-expert:  region[R, CAP, H] at the expert rank (CAP = T*K rows per source = worst case),
              tags/lid per row, per-source count header, flags fd[R], acks ack_d[R]
  to-source:  cbuf[T*K, H] at the source rank, flags fc[R], acks ack_c[R]
Compute rows on the expert rank are bounded by MAXROWS (grouped-GEMM grid); rows beyond it are
counted on the device in `overflow` (must stay 0; checked lazily by the caller, never synced here).
"""
import ctypes
import torch
import triton
import triton.language as tl

from stowaway.kernels.common_ops import dcu_state, hip
from stowaway.kernels.gemm_ar_rd import _wait_value_gte


# ---------------------------------------------------------------- kernels
@triton.jit(do_not_specialize=["x", "idx", "cnt", "rbase", "tbase", "lbase", "pdst", "prow", "me", "cap"])
def _k_dispatch(x, idx, cnt, rbase, tbase, lbase, pdst, prow, me, cap, H: tl.constexpr, K: tl.constexpr,
                EPR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t = (pid // K).to(tl.int64)
    e = tl.load(idx + pid)
    d = e // EPR
    slot = tl.atomic_add(cnt + d, 1)
    row = (me * cap + slot).to(tl.int64)
    dst = tl.load(rbase + d).to(tl.pointer_type(tl.bfloat16)) + row * H
    for off in tl.static_range(0, H, BLOCK):
        o = off + tl.arange(0, BLOCK)
        tl.store(dst + o, tl.load(x + t * H + o))
    tl.store(tl.load(tbase + d).to(tl.pointer_type(tl.int32)) + row, pid)
    tl.store(tl.load(lbase + d).to(tl.pointer_type(tl.int32)) + row, e % EPR)
    tl.store(pdst + pid, d)
    tl.store(prow + pid, row)


@triton.jit(do_not_specialize=["g", "pdst", "prow", "rbase"])
def _k_to_rows(g, pdst, prow, rbase, H: tl.constexpr, BLOCK: tl.constexpr):
    """source -> expert rows at a saved placement (combine backward)."""
    pid = tl.program_id(0).to(tl.int64)
    d = tl.load(pdst + pid)
    row = tl.load(prow + pid).to(tl.int64)
    dst = tl.load(rbase + d).to(tl.pointer_type(tl.bfloat16)) + row * H
    for off in tl.static_range(0, H, BLOCK):
        o = off + tl.arange(0, BLOCK)
        tl.store(dst + o, tl.load(g + pid * H + o))


@triton.jit(do_not_specialize=["y", "qsrc", "qtag", "total", "cbase"])
def _k_to_source(y, qsrc, qtag, total, cbase, H: tl.constexpr, BLOCK: tl.constexpr):
    """expert -> source: sorted row q goes to source qsrc[q], combine row qtag[q]."""
    q = tl.program_id(0)
    if q < tl.load(total):
        s = tl.load(qsrc + q)
        tag = tl.load(qtag + q).to(tl.int64)
        dst = tl.load(cbase + s).to(tl.pointer_type(tl.bfloat16)) + tag * H
        qq = q.to(tl.int64)
        for off in tl.static_range(0, H, BLOCK):
            o = off + tl.arange(0, BLOCK)
            tl.store(dst + o, tl.load(y + qq * H + o))


@triton.jit(do_not_specialize=["cnt", "hdr", "flg", "me", "rnd"])
def _k_signal(cnt, hdr, flg, me, rnd, R: tl.constexpr, WITH_HDR: tl.constexpr):
    for d in tl.static_range(R):
        if WITH_HDR:
            hp = tl.load(hdr + d).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(hp + me, tl.load(cnt + d), sem="release", scope="sys")
        fp = tl.load(flg + d).to(tl.pointer_type(tl.int32))
        tl.atomic_xchg(fp + me, rnd, sem="release", scope="sys")


@triton.jit
def _k_wsum(cbuf, w, out, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, WEIGHTED: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    for off in tl.static_range(0, H, BLOCK):
        o = off + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(K):
            v = tl.load(cbuf + (t * K + k) * H + o).to(tl.float32)
            if WEIGHTED:
                v = v * tl.load(w + t * K + k)
            acc += v
        tl.store(out + t * H + o, acc.to(tl.bfloat16))


class _CAI:
    def __init__(self, p, shape, ts):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": ts, "data": (p, False), "version": 2}


# ---------------------------------------------------------------- context
class SFMoEContext:
    def __init__(self, T, K, H, E_global, rank, R, maxrows, block=1024):
        host = dcu_state()["host"]
        h = hip()
        h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.T, self.K, self.H, self.R, self.rank = T, K, H, R, rank
        self.EPR = E_global // R
        self.CAP = T * K
        self.maxrows = maxrows
        self.BLOCK = min(block, triton.next_power_of_2(H))
        rb = H * 2
        RC = R * self.CAP
        self.region = host.malloc(RC * rb)
        self.tags = host.malloc(RC * 4)
        self.lid = host.malloc(RC * 4)
        self.hdr = host.malloc(R * 4)
        self.cbuf = host.malloc(self.CAP * rb)
        self.fd = host.malloc(R * 4)
        self.ackd = host.malloc(R * 4)
        self.fc = host.malloc(R * 4)
        self.ackc = host.malloc(R * 4)
        for p in (self.region, self.tags, self.lid, self.hdr, self.cbuf, self.fd, self.ackd, self.fc, self.ackc):
            assert p, "dushmem_malloc returned NULL (raise DUSHMEM_SYMMETRIC_SIZE)"
        for p, n in ((self.hdr, R * 4), (self.fd, R * 4), (self.ackd, R * 4), (self.fc, R * 4),
                     (self.ackc, R * 4), (self.lid, RC * 4)):
            assert h.hipMemset(ctypes.c_void_p(p), 0, n) == 0
        V = lambda p: torch.tensor([host.ptr(p, d) for d in range(R)], dtype=torch.int64, device="cuda")  # noqa
        self.rbase, self.tbase, self.lbase = V(self.region), V(self.tags), V(self.lid)
        self.hdr_t, self.fd_t, self.ackd_t, self.fc_t, self.ackc_t, self.cbase = (
            V(self.hdr), V(self.fd), V(self.ackd), V(self.fc), V(self.ackc), V(self.cbuf))
        self.region_v = torch.as_tensor(_CAI(self.region, (RC, H), "<i2"), device="cuda").view(torch.bfloat16)
        self.tags_v = torch.as_tensor(_CAI(self.tags, (RC,), "<i4"), device="cuda")
        self.lid_v = torch.as_tensor(_CAI(self.lid, (RC,), "<i4"), device="cuda")
        self.hdr_v = torch.as_tensor(_CAI(self.hdr, (R,), "<i4"), device="cuda")
        self.cbuf_v = torch.as_tensor(_CAI(self.cbuf, (self.CAP, H), "<i2"), device="cuda").view(torch.bfloat16)
        self.cnt = torch.zeros(R, dtype=torch.int32, device="cuda")
        self.src_of = (torch.arange(RC, device="cuda") // self.CAP).to(torch.int32)
        self.j_of = torch.arange(RC, device="cuda") % self.CAP
        self.overflow = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.rd = 0         # to-expert rounds
        self.rc = 0         # to-source rounds
        self.n_calls = 0
        torch.cuda.synchronize()
        host.barrier_all()

    # ------------------------------------------------------------ waits / acks (stream side)
    def _wait_all(self, base, val, stream):
        if val <= 0:
            return
        for s in range(self.R):
            _wait_value_gte(base + 4 * s, val, stream)

    def _ack(self, flg_t, val):
        _k_signal[(1,)](self.cnt, self.hdr_t, flg_t, self.rank, val, R=self.R, WITH_HDR=False, num_warps=1)

    # ------------------------------------------------------------ to-expert
    def dispatch(self, x, idx):
        """x [T, H] bf16, idx [T*K] int32 global expert ids. Returns the expert-side sorted problem."""
        st = torch.cuda.current_stream()
        self.rd += 1
        r = self.rd
        self._wait_all(self.ackd, r - 1, st)          # every expert rank consumed my previous rows
        self.cnt.zero_()
        P = self.T * self.K
        pdst = torch.empty(P, dtype=torch.int32, device="cuda")
        prow = torch.empty(P, dtype=torch.int32, device="cuda")
        _k_dispatch[(P,)](x, idx, self.cnt, self.rbase, self.tbase, self.lbase, pdst, prow, self.rank, self.CAP,
                          H=self.H, K=self.K, EPR=self.EPR, BLOCK=self.BLOCK, num_warps=4)
        _k_signal[(1,)](self.cnt, self.hdr_t, self.fd_t, self.rank, r, R=self.R, WITH_HDR=True, num_warps=1)
        self._wait_all(self.fd, r, st)                # all sources landed their rows + counts
        counts = self.hdr_v.clone().to(torch.int64)   # per-source rows (device copy, no host read)
        valid = self.j_of < counts[self.src_of.long()]
        key = torch.where(valid, self.lid_v.long(), torch.full_like(self.j_of, self.EPR))
        order = torch.argsort(key, stable=True)[: self.maxrows]
        hist = torch.bincount(key, minlength=self.EPR + 1)[: self.EPR]
        total_real = hist.sum()
        self.overflow += (total_real - self.maxrows).clamp_min(0)
        offs = torch.cumsum(hist, 0).clamp_max(self.maxrows)
        total = offs[-1:].clone().to(torch.int32)
        x_perm = self.region_v.index_select(0, order)
        q_src = self.src_of.index_select(0, order).contiguous()
        q_tag = self.tags_v.index_select(0, order).contiguous()
        self._ack(self.ackd_t, r)                    # I consumed every source's rows of round r
        return x_perm, offs, total, q_src, q_tag, order, pdst, prow

    def to_rows(self, g, pdst, prow, order):
        """combine backward: source rows g [T*K, H] -> expert rows at the saved placement -> sorted [maxrows, H]."""
        st = torch.cuda.current_stream()
        self.rd += 1
        r = self.rd
        self._wait_all(self.ackd, r - 1, st)
        _k_to_rows[(self.T * self.K,)](g, pdst, prow, self.rbase, H=self.H, BLOCK=self.BLOCK, num_warps=4)
        _k_signal[(1,)](self.cnt, self.hdr_t, self.fd_t, self.rank, r, R=self.R, WITH_HDR=False, num_warps=1)
        self._wait_all(self.fd, r, st)
        out = self.region_v.index_select(0, order)
        self._ack(self.ackd_t, r)
        return out

    # ------------------------------------------------------------ to-source
    def to_source(self, y, q_src, q_tag, total, w=None, keep=False):
        """expert rows y [maxrows, H] (sorted) -> sources; returns sum_k (w *) rows [T, H] (and the rows if keep)."""
        st = torch.cuda.current_stream()
        self.rc += 1
        r = self.rc
        self._wait_all(self.ackc, r - 1, st)          # every source consumed my previous rows
        _k_to_source[(self.maxrows,)](y, q_src, q_tag, total, self.cbase, H=self.H, BLOCK=self.BLOCK, num_warps=4)
        _k_signal[(1,)](self.cnt, self.hdr_t, self.fc_t, self.rank, r, R=self.R, WITH_HDR=False, num_warps=1)
        self._wait_all(self.fc, r, st)
        out = torch.empty(self.T, self.H, dtype=torch.bfloat16, device="cuda")
        _k_wsum[(self.T,)](self.cbuf_v, w if w is not None else out, out, H=self.H, K=self.K, BLOCK=self.BLOCK,
                           WEIGHTED=w is not None, num_warps=4)
        rows = self.cbuf_v.clone() if keep else None
        self._ack(self.ackc_t, r)
        return out, rows


# ---------------------------------------------------------------- autograd
class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, sf):
        x_perm, offs, total, q_src, q_tag, order, pdst, prow = sf.dispatch(x.contiguous(), idx)
        ctx.sf = sf
        ctx.save_for_backward(q_src, q_tag, total)
        ctx.mark_non_differentiable(offs, total, q_src, q_tag, order, pdst, prow)
        return x_perm, offs, total, q_src, q_tag, order, pdst, prow

    @staticmethod
    def backward(ctx, g_perm, *_):
        q_src, q_tag, total = ctx.saved_tensors
        gx, _ = ctx.sf.to_source(g_perm.contiguous(), q_src, q_tag, total, w=None)
        return gx, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_perm, w, q_src, q_tag, total, order, pdst, prow, sf):
        out, rows = sf.to_source(y_perm.contiguous(), q_src, q_tag, total, w=w.contiguous(), keep=True)
        ctx.sf = sf
        ctx.save_for_backward(w, rows, order, pdst, prow)
        return out

    @staticmethod
    def backward(ctx, gout):
        w, rows, order, pdst, prow = ctx.saved_tensors
        sf = ctx.sf
        gout = gout.contiguous()
        T, K, H = sf.T, sf.K, sf.H
        g_tk = (gout.float().unsqueeze(1) * w.view(T, K, 1)).to(torch.bfloat16).reshape(T * K, H)
        gw = (gout.float().unsqueeze(1) * rows.view(T, K, H).float()).sum(-1)
        g_perm = sf.to_rows(g_tk, pdst, prow, order)
        return g_perm, gw.view_as(w), None, None, None, None, None, None, None


def sf_moe(x, idx, w, sf, experts_fn):
    """x [T, H] bf16, idx [T*K] int32, w [T, K] fp32 routing weights (grad flows to the router),
    experts_fn(x_perm, offs) -> y_perm: the local expert MLP on expert-sorted rows."""
    x_perm, offs, total, q_src, q_tag, order, pdst, prow = _Dispatch.apply(x, idx, sf)
    y_perm = experts_fn(x_perm, offs)
    return _Combine.apply(y_perm, w, q_src, q_tag, total, order, pdst, prow, sf)
