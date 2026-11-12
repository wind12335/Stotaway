"""Correctness-first, device-offset grouped expert linear for NVIDIA GPUs.

The offsets are exclusive cumulative ends on the GPU.  This is a fixed-grid
Triton implementation, not a cuBLAS grouped-GEMM replacement or a tuned kernel.
Offsets must be monotone, nonnegative, and end at or before the input row count.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _rows_gemm(X, W, OFF, Y, E: tl.constexpr, T: tl.constexpr,
               K: tl.constexpr, N: tl.constexpr, TRANS: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    rn = tl.program_id(2) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    start = tl.load(OFF + e - 1, mask=e > 0, other=0)
    end = tl.load(OFF + e)
    row = start + rm
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(triton.cdiv(K, BK)):
        ki = k * BK + rk
        a = tl.load(X + row[:, None] * K + ki[None, :],
                    (row[:, None] < end) & (row[:, None] < T) & (ki[None, :] < K), 0)
        if TRANS:
            b = tl.load(W + e * N * K + rn[None, :] * K + ki[:, None],
                        (ki[:, None] < K) & (rn[None, :] < N), 0)
        else:
            b = tl.load(W + e * K * N + ki[:, None] * N + rn[None, :],
                        (ki[:, None] < K) & (rn[None, :] < N), 0)
        acc += tl.dot(a, b)
    tl.store(Y + row[:, None] * N + rn[None, :], acc,
             (row[:, None] < end) & (row[:, None] < T) & (rn[None, :] < N))


@triton.jit
def _wgrad(X, DY, OFF, DW, T: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
           BK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr):
    e = tl.program_id(0)
    ik = tl.program_id(1) * BK + tl.arange(0, BK)
    jn = tl.program_id(2) * BN + tl.arange(0, BN)
    rr = tl.arange(0, BR)
    start = tl.load(OFF + e - 1, mask=e > 0, other=0)
    end = tl.minimum(tl.load(OFF + e), T)
    acc = tl.zeros((BK, BN), tl.float32)
    for base in range(start, end, BR):
        row = base + rr
        a = tl.load(X + row[None, :] * K + ik[:, None],
                    (row[None, :] < end) & (ik[:, None] < K), 0)
        b = tl.load(DY + row[:, None] * N + jn[None, :],
                    (row[:, None] < end) & (jn[None, :] < N), 0)
        acc += tl.dot(a, b)
    tl.store(DW + e * K * N + ik[:, None] * N + jn[None, :], acc,
             (ik[:, None] < K) & (jn[None, :] < N))


def _check(x, w, offsets):
    if (x.ndim != 2 or w.ndim != 3 or offsets.ndim != 1 or
            w.shape[0] != offsets.numel() or w.shape[0] == 0 or
            w.shape[1] == 0 or w.shape[2] == 0 or
            x.shape[1] != w.shape[1] or x.dtype not in (torch.float16, torch.bfloat16) or
            w.dtype != x.dtype or offsets.dtype not in (torch.int32, torch.int64) or
            not x.is_cuda or w.device != x.device or offsets.device != x.device or
            not all(t.is_contiguous() for t in (x, w, offsets))):
        raise ValueError("expected contiguous CUDA x[T,K], w[E,K,N], integer offsets[E]")


class _ExpertLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offsets):
        _check(x, w, offsets)
        t, k = x.shape
        e, _, n = w.shape
        y = torch.zeros((t, n), dtype=x.dtype, device=x.device)
        if t:
            _rows_gemm[(e, triton.cdiv(t, 32), triton.cdiv(n, 64))](
                x, w, offsets, y, e, t, k, n, False, 32, 64, 32)
        ctx.save_for_backward(x, w, offsets)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, w, offsets = ctx.saved_tensors
        dy = dy.contiguous()
        t, k = x.shape
        e, _, n = w.shape
        dx = torch.zeros_like(x)
        dw = torch.empty_like(w)
        if t:
            _rows_gemm[(e, triton.cdiv(t, 32), triton.cdiv(k, 64))](
                dy, w, offsets, dx, e, t, n, k, True, 32, 64, 32)
            _wgrad[(e, triton.cdiv(k, 32), triton.cdiv(n, 64))](
                x, dy, offsets, dw, t, k, n, 32, 64, 32)
        else:
            dw.zero_()
        return dx, dw, None


def expert_linear(x, w, offsets):
    """Compute expert-sorted rows and first-order gradients without host offset reads.

    Rows after offsets[-1] are zero and contribute no gradient. Offsets are a
    trusted device-side input: invalid values are not checked or synchronized.
    """
    return _ExpertLinear.apply(x, w, offsets)
