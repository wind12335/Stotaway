"""Explicit TP2 conversion between contiguous and rank-interleaved RS rows.

This module converts *logical row order* only. Backend-private packed tile
layouts must be normalized before use; in particular raw FlashOverlap output
is not accepted directly. The conversion uses a collective and all ranks must
call it in the same order with identical dimensions and direction.
"""
import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _pack_contiguous_to_interleaved(x, packed, HALF: tl.constexpr, N: tl.constexpr,
                                    BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = HALF * N
    row = idx // N
    col = idx % N
    dest = row // (HALF // 2)
    local_row = row % (HALF // 2)
    src_row = local_row * 2 + dest
    value = tl.load(x + src_row * N + col, idx < count, 0)
    tl.store(packed + idx, value, idx < count)


@triton.jit
def _unpack_interleaved_to_contiguous(recv, output, HALF: tl.constexpr, N: tl.constexpr,
                                       BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = HALF * N
    row = idx // N
    col = idx % N
    src_rank = row % 2
    src_row = row // 2
    value = tl.load(recv + (src_rank * (HALF // 2) + src_row) * N + col, idx < count, 0)
    tl.store(output + idx, value, idx < count)


def convert_rs_rows(tensor: torch.Tensor, direction: str, group=None) -> torch.Tensor:
    """Collectively convert a logical TP2 RS shard, returning independent storage.

    ``contiguous_to_interleaved`` maps global rank-contiguous row shards to
    rank-strided shards (global rows rank::2); the reverse mapping is
    ``interleaved_to_contiguous``. An even number of rows per rank is required.
    The operation is forward-only and performs no tile decoding or RMSNorm.
    """
    if direction not in ('contiguous_to_interleaved', 'interleaved_to_contiguous'):
        raise ValueError('unknown conversion direction')
    if not dist.is_initialized() or dist.get_world_size(group) != 2:
        raise ValueError('a two-rank process group is required')
    if (tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[0] % 2
            or tensor.shape[1] <= 0 or not tensor.is_cuda or not tensor.is_contiguous()
            or tensor.dtype not in (torch.float16, torch.bfloat16) or tensor.requires_grad):
        raise ValueError('expected a contiguous CUDA FP16/BF16 matrix with even rows and no gradient')
    rows, columns = tensor.shape
    block = 1024
    if direction == 'contiguous_to_interleaved':
        send = torch.empty_like(tensor)
        output = torch.empty_like(tensor)
        _pack_contiguous_to_interleaved[(triton.cdiv(tensor.numel(), block),)](
            tensor, send, rows, columns, block)
        dist.all_to_all_single(output, send, group=group)
    else:
        recv = torch.empty_like(tensor)
        output = torch.empty_like(tensor)
        dist.all_to_all_single(recv, tensor, group=group)
        _unpack_interleaved_to_contiguous[(triton.cdiv(tensor.numel(), block),)](
            recv, output, rows, columns, block)
    return output
