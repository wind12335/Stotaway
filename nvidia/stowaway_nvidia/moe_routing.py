"""Fixed-size GPU histogram and offsets for expert-grouped routing."""
import torch
import triton
import triton.language as tl


@triton.jit
def _histogram(KEYS, COUNTS, N: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(KEYS + i, i < N, other=-1)
    valid = (i < N) & (expert >= 0) & (expert < E)
    tl.atomic_add(COUNTS + expert, 1, valid, sem="relaxed")


@triton.jit
def _prefix(COUNTS, OFFSETS, TOTAL, E: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    counts = tl.load(COUNTS + e, e < E, 0)
    prefix = tl.cumsum(counts, axis=0)
    tl.store(OFFSETS + e, prefix, e < E)
    tl.store(TOTAL, tl.sum(counts, axis=0))


def expert_offsets(keys, experts):
    """Return int32 cumulative counts and total, both on device.

    IDs outside [0, experts) are ignored, allowing a sentinel for padding.
    Input IDs must have already been grouped by expert before passing these
    offsets to expert_linear. Counts and offsets are restricted to int32.
    """
    if (not isinstance(experts, int) or isinstance(experts, bool) or
            experts < 1 or experts > 1024 or not keys.is_cuda or
            keys.ndim != 1 or not keys.is_contiguous() or
            keys.dtype not in (torch.int32, torch.int64) or keys.numel() >= 2**31):
        raise ValueError("expected contiguous CUDA integer keys and 1..1024 experts")
    counts = torch.zeros(experts, device=keys.device, dtype=torch.int32)
    offsets = torch.empty_like(counts)
    total = torch.empty((), device=keys.device, dtype=torch.int32)
    if keys.numel():
        _histogram[(triton.cdiv(keys.numel(), 256),)](keys, counts, keys.numel(), experts, 256)
    _prefix[(1,)](counts, offsets, total, experts, triton.next_power_of_2(experts))
    return offsets, total
