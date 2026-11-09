"""Sync-free, dropless MoE expert MLP for Megatron (innovation 1, MoE side): forward + backward of every
expert linear as ONE hipBLASLt grouped GEMM driven by device-side user arguments. The per-expert token
counts never go to the host (Megatron's SequentialMLP does `tokens_per_expert.tolist()` and loops).

Weights stay as the experts' own parameters (Megatron Linear layout W_e [N_out, K_in], row-major); a device
table of their pointers is built once. Gradients are returned per expert parameter (views of one [E, N, K]
buffer), so Megatron's DDP grad hooks fire as usual.

Column-major mapping (hipBLASLt), rows = tokens of the expert:
  fwd    Y^T[N,m]  = op_T(W)[N,K]  * X^T[K,m]       opA=T opB=N   m'=N n'=rows(virtual) k'=K
  dgrad  dX^T[K,m] = W(col K x N)  * dY^T[N,m]      opA=N opB=N   m'=K n'=rows(virtual) k'=N
  wgrad  dW^T[K,N] = X^T[K,m]      * op_T(dY^T)     opA=N opB=T   m'=K n'=N k'=rows (one group / expert)
"""
import torch
import triton
import triton.language as tl

from stowaway.kernels.moe_ggemm import _GroupedGemmDev


@triton.jit(do_not_specialize=["ua", "pa", "pb", "pd"])
def _k_fill(ua, offs, pa, pb, pd, ta, tb, td, E, G, Bv, ra, rb, rd,
            S4: tl.constexpr, VIRT: tl.constexpr, A_TAB: tl.constexpr, B_TAB: tl.constexpr,
            D_TAB: tl.constexpr, EB: tl.constexpr, GB: tl.constexpr):
    """Write n (VIRT) or k (not VIRT), d, c, a, b of every group.
    A/B/D: *_TAB -> per-expert pointer table t*[e]; else row-sliced base p* + row_start * r* bytes."""
    e = tl.arange(0, EB)
    me = e < E
    end = tl.load(offs + e, mask=me, other=0).to(tl.int64)
    prev = tl.load(offs + e - 1, mask=me & (e > 0), other=0).to(tl.int64)
    n = tl.where(me, tl.maximum(end - prev, 0), 0)
    if VIRT:
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
        ge = tl.where(valid, eg, 0)
    else:
        g = e
        mg = me
        cnt = n
        rs = prev
        ge = e
    if A_TAB:
        ap = tl.load(ta + ge, mask=mg, other=0)
    else:
        ap = pa + rs * ra
    if B_TAB:
        bp = tl.load(tb + ge, mask=mg, other=0)
    else:
        bp = pb + rs * rb
    if D_TAB:
        dp = tl.load(td + ge, mask=mg, other=0)
    else:
        dp = pd + rs * rd
    u = ua.to(tl.pointer_type(tl.int32))
    base = g * S4
    M32 = 0xFFFFFFFF
    if VIRT:
        tl.store(u + base + 1, cnt.to(tl.int32), mask=mg)
    else:
        tl.store(u + base + 3, cnt.to(tl.int32), mask=mg)
    tl.store(u + base + 4, (dp & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 5, (dp >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 6, (dp & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 7, (dp >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 8, (ap & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 9, (ap >> 32).to(tl.int32), mask=mg)
    tl.store(u + base + 10, (bp & M32).to(tl.int32), mask=mg)
    tl.store(u + base + 11, (bp >> 32).to(tl.int32), mask=mg)


class ExpertLinearPlan:
    """the three grouped GEMMs of one expert linear (K_in -> N_out) for up to max_rows rows."""

    def __init__(self, E, K, N, max_rows, B=2048, ws_bytes=32 << 20):
        self.E, self.K, self.N, self.B = E, K, N, B
        self.G = E + (max_rows + B - 1) // B
        self.fwd = _GroupedGemmDev(self.G, 1, 0, N, B, K, ws_bytes)
        self.dg = _GroupedGemmDev(self.G, 0, 0, K, B, N, ws_bytes)
        self.wg = _GroupedGemmDev(E, 0, 1, K, N, max(1, max_rows // E), ws_bytes)
        self.EB, self.GB = triton.next_power_of_2(E), triton.next_power_of_2(self.G)
        self.z = torch.zeros(1, dtype=torch.int64, device="cuda")

    def _fill(self, gg, offs, pa, pb, pd, ta, tb, td, ra, rb, rd, virt, at, bt, dt):
        z = self.z
        _k_fill[(1,)](gg.ua, offs, pa, pb, pd, ta if ta is not None else z, tb if tb is not None else z,
                      td if td is not None else z, self.E, self.G, self.B, ra, rb, rd, S4=gg.S // 4,
                      VIRT=virt, A_TAB=at, B_TAB=bt, D_TAB=dt, EB=self.EB, GB=self.GB)
        gg.run()

    def forward(self, x, wtab, offs, y):
        self._fill(self.fwd, offs, 0, x.data_ptr(), y.data_ptr(), wtab, None, None, 0, self.K * 2, self.N * 2,
                   True, True, False, False)
        return y

    def dgrad(self, dy, wtab, offs, dx):
        self._fill(self.dg, offs, 0, dy.data_ptr(), dx.data_ptr(), wtab, None, None, 0, self.N * 2, self.K * 2,
                   True, True, False, False)
        return dx

    def wgrad(self, x, dy, offs, dwtab):
        self._fill(self.wg, offs, x.data_ptr(), dy.data_ptr(), 0, None, None, dwtab, self.K * 2, self.N * 2, 0,
                   False, False, False, True)


_PLANS = {}


def get_plan(E, K, N, max_rows, B):
    key = (E, K, N, max_rows, B)
    if key not in _PLANS:
        _PLANS[key] = ExpertLinearPlan(E, K, N, max_rows, B)
    return _PLANS[key]


def _ptr_table(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64).to("cuda", non_blocking=True)


class _ExpertLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, offs, plan, wtab, *weights):
        y = torch.empty(x.shape[0], plan.N, dtype=x.dtype, device=x.device)
        plan.forward(x.contiguous(), wtab, offs, y)
        ctx.save_for_backward(x, offs)
        ctx.plan, ctx.wtab, ctx.n_w = plan, wtab, len(weights)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, offs = ctx.saved_tensors
        plan = ctx.plan
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        plan.dgrad(dy, ctx.wtab, offs, dx)
        dw = torch.zeros(plan.E, plan.N, plan.K, dtype=x.dtype, device=x.device)   # 0-row experts read 0
        plan.wgrad(x, dy, offs, _ptr_table_fast(dw, plan))
        return (dx, None, None, None) + tuple(dw[e] for e in range(ctx.n_w))


def _ptr_table_fast(buf, plan):
    """device pointer table of buf[e] without a host round trip."""
    step = plan.N * plan.K * buf.element_size()
    return buf.data_ptr() + torch.arange(plan.E, dtype=torch.int64, device=buf.device) * step


def expert_linear(x, offs, weights, plan, wtab):
    """x [rows, K] grouped by expert, offs device int64 [E] cumulative ends, weights = E params [N, K]."""
    return _ExpertLinearFn.apply(x, offs, plan, wtab, *weights)
