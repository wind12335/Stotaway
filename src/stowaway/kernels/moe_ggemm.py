"""Sync-free MoE expert GEMM (innovation 1, MoE side). hipBLASLt grouped GEMM driven by DEVICE user
arguments: the per-expert token counts / row offsets stay on the GPU, the host never reads them.
Replaces Megatron's `tokens_per_expert.tolist()` + per-expert GEMM loop (one host sync per layer).

  gg = SyncFreeGroupedGemm(E, K, N, max_rows, rep_m)     # one per (shape, max_rows)
  gg(x, w, offs, y)   x [rows, K] grouped by expert, w [E, K, N], offs int32/int64 [E] exclusive-end
                      cumulative counts ON DEVICE, y [rows, N] out.  No host sync.
Library: src/csrc/libmoe_ggemm.so (hipcc -shared moe_ggemm.cpp -lhipblaslt)."""
import ctypes, os
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ua", "xb", "yb", "wb"])
def _k_fill_userargs(ua, offs, start, clamp, xb, yb, wb, E, K2, N2, KN2, S4: tl.constexpr,
                     HAS_START: tl.constexpr, HAS_CLAMP: tl.constexpr, BLOCK: tl.constexpr):
    """one launch writes every expert's UserArguments fields (n, d, c, a, b) on the device."""
    e = tl.arange(0, BLOCK)
    m = e < E
    end = tl.load(offs + e, mask=m, other=0).to(tl.int64)
    prev = tl.load(offs + e - 1, mask=m & (e > 0), other=0).to(tl.int64)
    if HAS_START:
        st = tl.load(start + e, mask=m, other=0).to(tl.int64)
    else:
        st = prev
    cnt = tl.maximum(end - st, 0)
    if HAS_CLAMP:
        cnt = tl.minimum(cnt, tl.load(clamp + e, mask=m, other=0).to(tl.int64))
    u = ua.to(tl.pointer_type(tl.int32))
    base = e * S4
    xp = xb + st * K2
    yp = yb + st * N2
    wp = wb + e.to(tl.int64) * KN2
    msk32 = 0xFFFFFFFF
    tl.store(u + base + 1, cnt.to(tl.int32), mask=m)
    tl.store(u + base + 4, (yp & msk32).to(tl.int32), mask=m)
    tl.store(u + base + 5, (yp >> 32).to(tl.int32), mask=m)
    tl.store(u + base + 6, (yp & msk32).to(tl.int32), mask=m)
    tl.store(u + base + 7, (yp >> 32).to(tl.int32), mask=m)
    tl.store(u + base + 8, (wp & msk32).to(tl.int32), mask=m)
    tl.store(u + base + 9, (wp >> 32).to(tl.int32), mask=m)
    tl.store(u + base + 10, (xp & msk32).to(tl.int32), mask=m)
    tl.store(u + base + 11, (xp >> 32).to(tl.int32), mask=m)

_LIB = None
_FUSED_ARGS = os.environ.get("MOE_GG_ARGS", "fused") == "fused"


def _lib():
    global _LIB
    if _LIB is None:
        # src/stowaway/kernels/moe_ggemm.py -> src/csrc/libmoe_ggemm.so (build: src/csrc/build.sh)
        p = os.environ.get("SW_MOE_GG_LIB") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "csrc", "libmoe_ggemm.so")
        L = ctypes.CDLL(p)
        L.gg_create.restype = ctypes.c_void_p
        L.gg_create.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        L.gg_create_v.restype = ctypes.c_void_p
        L.gg_create_v.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.POINTER(ctypes.c_long),
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        L.gg_destroy.restype = None
        L.gg_destroy.argtypes = [ctypes.c_void_p]
        L.gg_userargs.restype = ctypes.c_void_p
        L.gg_userargs.argtypes = [ctypes.c_void_p]
        L.gg_run.restype = ctypes.c_int
        L.gg_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.gg_userargs_size.restype = ctypes.c_int
        _LIB = L
    return _LIB


class _CAI:
    def __init__(self, ptr, n, typestr):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": typestr, "data": (ptr, False), "version": 2}


class SyncFreeGroupedGemm:
    def __init__(self, E, K, N, max_rows, rep_m, w=None, ws_bytes=32 << 20, bounds=None):
        """rep_m: per-expert row bound (grid size) for every expert, or bounds = list of E per-expert bounds.
        Rows of expert e beyond its bound are NOT computed (caller clamps and handles the spill)."""
        L = _lib()
        self.E, self.K, self.N = E, K, N
        dev = torch.device("cuda")
        # dummy buffers only for the heuristic / default user-args (real pointers written per call)
        self.bounds = list(bounds) if bounds is not None else [rep_m] * E
        bm = max(self.bounds)
        self._xd = torch.empty(bm, K, dtype=torch.bfloat16, device=dev)
        self._yd = torch.empty(bm, N, dtype=torch.bfloat16, device=dev)
        self._wd = w if w is not None else torch.empty(E, K, N, dtype=torch.bfloat16, device=dev)
        arr = (ctypes.c_long * E)(*self.bounds)
        self.h = L.gg_create_v(E, N, K, arr, self._wd.data_ptr(), self._xd.data_ptr(), self._yd.data_ptr(), ws_bytes)
        assert self.h, "gg_create failed"
        self.S = L.gg_userargs_size()           # UserArguments is packed (196 B on DTK 26.04): byte-level writes
        ua = L.gg_userargs(self.h)
        self.ua = ua
        self.u8 = torch.as_tensor(_CAI(ua, self.S * E, "|u1"), device=dev)
        e = torch.arange(E, device=dev, dtype=torch.int64)
        b8 = torch.arange(8, device=dev, dtype=torch.int64)
        b4 = torch.arange(4, device=dev, dtype=torch.int64)
        base = e * self.S
        self.i_n = (base[:, None] + 4 + b4).reshape(-1)          # .n   uint32 @4
        self.i_d = (base[:, None] + 16 + b8).reshape(-1)         # .d   ptr    @16
        self.i_c = (base[:, None] + 24 + b8).reshape(-1)         # .c   ptr    @24
        self.i_a = (base[:, None] + 32 + b8).reshape(-1)         # .a   ptr    @32
        self.i_b = (base[:, None] + 40 + b8).reshape(-1)         # .b   ptr    @40
        self.e = e

    def _put(self, idx, vals):
        self.u8.index_copy_(0, idx, vals.contiguous().view(torch.uint8).reshape(-1))

    def close(self):
        if self.h:
            _lib().gg_destroy(self.h)
            self.h = None

    def __call__(self, x, w, offs, y, stream=None, clamp=None, start=None):
        """x [rows, K] bf16, w [E, K, N] bf16 contiguous, offs [E] device cumulative end offsets, y [rows, N]."""
        if _FUSED_ARGS:
            assert self.S % 4 == 0
            _k_fill_userargs[(1,)](self.ua, offs, start if start is not None else offs,
                                   clamp if clamp is not None else offs, x.data_ptr(), y.data_ptr(),
                                   w.data_ptr(), self.E, self.K * 2, self.N * 2, self.K * self.N * 2,
                                   S4=self.S // 4, HAS_START=start is not None, HAS_CLAMP=clamp is not None,
                                   BLOCK=triton.next_power_of_2(self.E))
        else:
            offs = offs.to(torch.int64)
            if start is None:
                start = torch.cat([offs.new_zeros(1), offs[:-1]])
            cnt = (offs - start).clamp_min(0).to(torch.int32)
            if clamp is not None:
                cnt = torch.minimum(cnt, clamp)
            self._put(self.i_n, cnt)
            self._put(self.i_b, x.data_ptr() + start * (self.K * 2))
            yp = y.data_ptr() + start * (self.N * 2)
            self._put(self.i_d, yp)
            self._put(self.i_c, yp)
            self._put(self.i_a, w.data_ptr() + self.e * (self.K * self.N * 2))
        s = stream or torch.cuda.current_stream()
        rc = _lib().gg_run(self.h, ctypes.c_void_p(s.cuda_stream))
        assert rc == 0, f"gg_run rc={rc}"
        return y


# ----------------------------------------------------------------------------------------------
# Dropless, sync-free, fixed grid: every expert is cut ON THE DEVICE into virtual groups of at most
# B rows; G = E + ceil(rows_max / B) groups of bound B cover ANY routing (sum_e ceil(n_e/B) <= E + T/B).
# The hipBLASLt grid is fixed at G x B rows (host-side, built once); unused groups get 0 rows.
# B is the only knob: small B -> fewer wasted tiles per expert tail but more groups; the predictor
# picks B per layer from the predicted load profile (no host sync, no token drop).
# ----------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["ua", "xb", "yb", "wb"])
def _k_fill_virtual(ua, offs, xb, yb, wb, E, G, Bv, K2, N2, KN2, S4: tl.constexpr,
                    EB: tl.constexpr, GB: tl.constexpr):
    e = tl.arange(0, EB)
    me = e < E
    end = tl.load(offs + e, mask=me, other=0).to(tl.int64)
    prev = tl.load(offs + e - 1, mask=me & (e > 0), other=0).to(tl.int64)
    n = tl.where(me, end - prev, 0)
    ch = (n + Bv - 1) // Bv                                  # chunks per expert
    cum = tl.cumsum(ch, 0)                                   # inclusive
    g = tl.arange(0, GB)
    mg = g < G
    # expert of group g = #experts whose inclusive cum <= g
    eg = tl.sum((cum[None, :] <= g[:, None]).to(tl.int64) * me[None, :].to(tl.int64), 1)
    sel = (e[None, :] == eg[:, None]).to(tl.int64)
    cum_e = tl.sum(sel * cum[None, :], 1)
    ch_e = tl.sum(sel * ch[None, :], 1)
    st_e = tl.sum(sel * prev[None, :], 1)
    n_e = tl.sum(sel * n[None, :], 1)
    j = g - (cum_e - ch_e)                                   # chunk index inside the expert
    valid = mg & (eg < E)
    rs = st_e + j * Bv
    cnt = tl.where(valid, tl.minimum(Bv, n_e - j * Bv), 0)
    cnt = tl.maximum(cnt, 0)
    rs = tl.where(valid, rs, 0)
    egc = tl.where(valid, eg, 0)
    u = ua.to(tl.pointer_type(tl.int32))
    base = g * S4
    xp = xb + rs * K2
    yp = yb + rs * N2
    wp = wb + egc * KN2
    msk32 = 0xFFFFFFFF
    tl.store(u + base + 1, cnt.to(tl.int32), mask=mg)
    tl.store(u + base + 4, (yp & msk32).to(tl.int32), mask=mg)
    tl.store(u + base + 5, (yp >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 6, (yp & msk32).to(tl.int32), mask=mg)
    tl.store(u + base + 7, (yp >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 8, (wp & msk32).to(tl.int32), mask=mg)
    tl.store(u + base + 9, (wp >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 10, (xp & msk32).to(tl.int32), mask=mg)
    tl.store(u + base + 11, (xp >> 32).to(tl.int32), mask=mg)


class DroplessGroupedGemm:
    """y = per-expert x_e @ w_e for ANY routing (dropless), no host sync, fixed grid of G groups x B rows."""

    def __init__(self, E, K, N, max_rows, B, w=None, ws_bytes=32 << 20):
        self.E, self.K, self.N, self.B = E, K, N, B
        self.G = E + (max_rows + B - 1) // B
        self.inner = SyncFreeGroupedGemm(self.G, K, N, max_rows, B, w=None, ws_bytes=ws_bytes)

    def close(self):
        self.inner.close()

    def __call__(self, x, w, offs, y, stream=None):
        g = self.inner
        _k_fill_virtual[(1,)](g.ua, offs, x.data_ptr(), y.data_ptr(), w.data_ptr(), self.E, self.G, self.B,
                              self.K * 2, self.N * 2, self.K * self.N * 2, S4=g.S // 4,
                              EB=triton.next_power_of_2(self.E), GB=triton.next_power_of_2(self.G))
        s = stream or torch.cuda.current_stream()
        rc = _lib().gg_run(g.h, ctypes.c_void_p(s.cuda_stream))
        assert rc == 0, f"gg_run rc={rc}"
        return y


# ----------------------------------------------------------------------------------------------
# Backward. Column-major view (hipBLASLt):  forward  Y^T[N,m]  = W^T(opN, N x K) * X^T(opN, K x m)
#   dgrad  dX^T[K,m] = W(opT of col-major N x K) * dY^T(opN, N x m)    -> varying n = rows (virtual groups)
#   wgrad  dW^T[N,K] = dY^T(opN, N x m) * X(opT of col-major K x m)     -> varying k = rows (one group/expert;
#          the grid depends only on (N, K), so any k is legal; the host k only steers the heuristic)
# ----------------------------------------------------------------------------------------------
class _GroupedGemmDev:
    def __init__(self, G, opA, opB, m, n_bound, k_bound, ws_bytes=32 << 20):
        L = _lib()
        if not hasattr(L, "_g_bound"):
            L.gg_create_g.restype = ctypes.c_void_p
            L.gg_create_g.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long,
                                      ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long),
                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            L._g_bound = True
        nb = (ctypes.c_long * G)(*([n_bound] * G))
        kb = (ctypes.c_long * G)(*([k_bound] * G))
        rows = max(n_bound, k_bound, m)
        self._dummy = torch.empty(rows * max(m, 1) + 1024, dtype=torch.bfloat16, device="cuda")
        p = self._dummy.data_ptr()
        self.h = L.gg_create_g(G, opA, opB, m, nb, kb, p, p, p, ws_bytes)
        assert self.h, "gg_create_g failed"
        self.S = L.gg_userargs_size()
        self.ua = L.gg_userargs(self.h)
        self.G = G

    def run(self, stream=None):
        s = stream or torch.cuda.current_stream()
        rc = _lib().gg_run(self.h, ctypes.c_void_p(s.cuda_stream))
        assert rc == 0, f"gg_run rc={rc}"

    def close(self):
        if self.h:
            _lib().gg_destroy(self.h)
            self.h = None


@triton.jit(do_not_specialize=["ua", "ab", "bb", "db"])
def _k_fill_dgrad(ua, offs, ab, bb, db, E, G, Bv, AE, B2, D2, S4: tl.constexpr, EB: tl.constexpr, GB: tl.constexpr):
    """virtual groups over rows; A = per-expert weight (stride AE bytes), B/D = row-sliced (B2/D2 bytes per row)."""
    e = tl.arange(0, EB)
    me = e < E
    end = tl.load(offs + e, mask=me, other=0).to(tl.int64)
    prev = tl.load(offs + e - 1, mask=me & (e > 0), other=0).to(tl.int64)
    n = tl.where(me, end - prev, 0)
    ch = (n + Bv - 1) // Bv
    cum = tl.cumsum(ch, 0)
    g = tl.arange(0, GB)
    mg = g < G
    eg = tl.sum((cum[None, :] <= g[:, None]).to(tl.int64) * me[None, :].to(tl.int64), 1)
    sel = (e[None, :] == eg[:, None]).to(tl.int64)
    cum_e = tl.sum(sel * cum[None, :], 1)
    ch_e = tl.sum(sel * ch[None, :], 1)
    st_e = tl.sum(sel * prev[None, :], 1)
    n_e = tl.sum(sel * n[None, :], 1)
    j = g - (cum_e - ch_e)
    valid = mg & (eg < E)
    cnt = tl.maximum(tl.where(valid, tl.minimum(Bv, n_e - j * Bv), 0), 0)
    rs = tl.where(valid, st_e + j * Bv, 0)
    egc = tl.where(valid, eg, 0)
    u = ua.to(tl.pointer_type(tl.int32))
    base = g * S4
    ap = ab + egc * AE
    bp = bb + rs * B2
    dp = db + rs * D2
    M32 = 0xFFFFFFFF
    tl.store(u + base + 1, cnt.to(tl.int32), mask=mg)
    tl.store(u + base + 4, (dp & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 5, (dp >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 6, (dp & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 7, (dp >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 8, (ap & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 9, (ap >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 10, (bp & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 11, (bp >> 32).to(tl.int32), mask=mg)


@triton.jit(do_not_specialize=["ua", "ab", "bb", "db"])
def _k_fill_wgrad(ua, offs, ab, bb, db, E, A2, B2, DE, S4: tl.constexpr, EB: tl.constexpr):
    """one group per expert, k = rows of the expert; A = dY rows (A2 bytes/row), B = X rows (B2), D = dW_e (DE bytes)."""
    e = tl.arange(0, EB)
    me = e < E
    end = tl.load(offs + e, mask=me, other=0).to(tl.int64)
    prev = tl.load(offs + e - 1, mask=me & (e > 0), other=0).to(tl.int64)
    n = tl.maximum(end - prev, 0)
    u = ua.to(tl.pointer_type(tl.int32))
    base = e * S4
    ap = ab + prev * A2
    bp = bb + prev * B2
    dp = db + e.to(tl.int64) * DE
    M32 = 0xFFFFFFFF
    tl.store(u + base + 3, n.to(tl.int32), mask=me)
    tl.store(u + base + 4, (dp & M32).to(tl.int32), mask=me)
    tl.store(u + base + 5, (dp >> 32).to(tl.int32), mask=me)
    tl.store(u + base + 6, (dp & M32).to(tl.int32), mask=me)
    tl.store(u + base + 7, (dp >> 32).to(tl.int32), mask=me)
    tl.store(u + base + 8, (ap & M32).to(tl.int32), mask=me)
    tl.store(u + base + 9, (ap >> 32).to(tl.int32), mask=me)
    tl.store(u + base + 10, (bp & M32).to(tl.int32), mask=me)
    tl.store(u + base + 11, (bp >> 32).to(tl.int32), mask=me)


class DroplessExpertLinear:
    """Forward + backward of the expert linear y_e = x_e @ w_e (w [E, K, N]) for any routing, no host sync.
    offs: device int [E] cumulative end offsets of the expert-grouped rows."""

    def __init__(self, E, K, N, max_rows, B=2048, ws_bytes=32 << 20):
        self.E, self.K, self.N, self.B = E, K, N, B
        G = E + (max_rows + B - 1) // B
        self.G = G
        self.fwd = DroplessGroupedGemm(E, K, N, max_rows, B, ws_bytes=ws_bytes)
        self.dg = _GroupedGemmDev(G, 1, 0, K, B, N, ws_bytes)                      # m=K, n=rows, k=N
        self.wg = _GroupedGemmDev(E, 0, 1, N, K, max(1, max_rows // E), ws_bytes)  # m=N, n=K, k=rows
        self.EB = triton.next_power_of_2(E)
        self.GB = triton.next_power_of_2(G)

    def forward(self, x, w, offs, y):
        return self.fwd(x, w, offs, y)

    def dgrad(self, dy, w, offs, dx):
        _k_fill_dgrad[(1,)](self.dg.ua, offs, w.data_ptr(), dy.data_ptr(), dx.data_ptr(), self.E, self.G, self.B,
                            self.K * self.N * 2, self.N * 2, self.K * 2, S4=self.dg.S // 4, EB=self.EB, GB=self.GB)
        self.dg.run()
        return dx

    def wgrad(self, x, dy, offs, dw):
        """dw [E, K, N]; experts with 0 rows must read 0 -> dw is zeroed first (one memset)."""
        dw.zero_()
        _k_fill_wgrad[(1,)](self.wg.ua, offs, dy.data_ptr(), x.data_ptr(), dw.data_ptr(), self.E,
                            self.N * 2, self.K * 2, self.K * self.N * 2, S4=self.wg.S // 4, EB=self.EB)
        self.wg.run()
        return dw

    def close(self):
        self.fwd.close(); self.dg.close(); self.wg.close()
