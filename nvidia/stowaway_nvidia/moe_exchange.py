"""Two-rank fixed-capacity MoE dispatch/return diagnostic over NVSHMEM."""
import torch
import triton
import triton.language as tl

from .dx import _publish
from .moe_routing import expert_offsets
from .runtime import SymmetricBuffers


@triton.jit
def _dispatch(X, IDS, DATA_PTRS, ID_PTRS, P: tl.constexpr, H: tl.constexpr,
              TOPK: tl.constexpr, EPR: tl.constexpr, RANK: tl.constexpr,
              BLOCK: tl.constexpr, PER_TAG: tl.constexpr):
    tag = tl.program_id(0)
    dest = tl.program_id(1)
    expert = tl.load(IDS + tag)
    valid = (expert >= dest * EPR) & (expert < (dest + 1) * EPR)
    ids = tl.load(ID_PTRS + dest).to(tl.pointer_type(tl.int32))
    tl.store(ids + RANK * P + tag, tl.where(valid, expert - dest * EPR, EPR))
    if valid:
        data = tl.load(DATA_PTRS + dest).to(tl.pointer_type(tl.bfloat16))
        col = tl.arange(0, BLOCK)
        for base in range(triton.cdiv(H, BLOCK)):
            h = base * BLOCK + col
            source_row = tag if PER_TAG else tag // TOPK
            val = tl.load(X + source_row * H + h, h < H, 0)
            tl.store(data + (RANK * P + tag) * H + h, val, h < H)


@triton.jit
def _return(Y, ORDER, TOTAL, PTRS, P: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row < tl.load(TOTAL):
        original = tl.load(ORDER + row)
        source = original // P
        tag = original % P
        dest = tl.load(PTRS + source).to(tl.pointer_type(tl.bfloat16))
        col = tl.arange(0, BLOCK)
        for base in range(triton.cdiv(H, BLOCK)):
            h = base * BLOCK + col
            value = tl.load(Y + row * H + h, h < H, 0)
            tl.store(dest + tag * H + h, value, h < H)


class MoEExchange:
    """EP=2 exchange; collective calls and context lifetimes must match on both ranks.

    Each rank handles the same fixed T, top-k and hidden size; IDs are trusted
    valid global expert IDs. At most one dispatch may be outstanding per context.
    This prototype is inference/forward-only and deliberately reserves 2*T*K
    rows per expert rank to cover worst-case routing without dropping tokens.
    """
    def __init__(self, cuda, rank, tokens, topk, hidden, experts):
        if (rank not in (0, 1) or min(tokens, topk, hidden, experts) <= 0 or
                experts % 2 or experts > 2048 or tokens * topk * 2 >= 2**31):
            raise ValueError("requires positive sizes, at most 2048 even experts and rank 0/1")
        self.cuda, self.rank = cuda, rank
        self.tokens, self.topk, self.hidden, self.experts = tokens, topk, hidden, experts
        self.p, self.epr = tokens * topk, experts // 2
        self.buffers = SymmetricBuffers(rank, 2, [
            ((2, self.p, hidden), torch.bfloat16), ((2, self.p), torch.int32),
            ((self.p, hidden), torch.bfloat16), ((2,), torch.int32),
            ((2,), torch.int32), ((2,), torch.int32), ((2,), torch.int32),
        ])
        self.data, self.ids, self.result, self.ready, self.consumed, self.returned, self.return_ack = self.buffers.buffers
        def pointers(peers):
            return torch.tensor([p.data_ptr() if isinstance(p, torch.Tensor) else p for p in peers],
                                dtype=torch.int64, device="cuda")
        self.data_ptrs, self.id_ptrs, self.result_ptrs = map(pointers, (self.data, self.ids, self.result))
        self.turn = 0
        self.pending = False
        # Pointer tables are created asynchronously on the constructor stream.
        # Keep all subsequent calls on that stream to preserve their ordering.
        self.stream = torch.cuda.current_stream().cuda_stream
        self.closed = False

    def _check_stream(self):
        if self.closed:
            raise RuntimeError("context is closed")
        current = torch.cuda.current_stream()
        if self.stream != current.cuda_stream:
            raise RuntimeError("MoEExchange requires a single caller stream")
        return current

    def dispatch(self, x, ids, *, per_tag=False):
        current = self._check_stream()
        if self.pending:
            raise RuntimeError("finish the previous exchange first")
        if (x.shape != (self.p if per_tag else self.tokens, self.hidden) or x.dtype != torch.bfloat16 or
                ids.shape != (self.tokens, self.topk) or ids.dtype not in (torch.int32, torch.int64) or
                x.device != self.data[self.rank].device or ids.device != x.device or
                not x.is_contiguous() or not ids.is_contiguous() or x.requires_grad):
            raise ValueError("invalid dispatch inputs; prototype is forward-only")
        self.turn += 1
        if self.turn >= 2**31:
            raise RuntimeError("ticket counter exhausted")
        for peer in range(2):
            if self.turn > 1:
                self.cuda.wait(self.consumed[self.rank].data_ptr() + 4 * peer, self.turn - 1, current)
        _dispatch[(self.p, 2)](x, ids, self.data_ptrs, self.id_ptrs, self.p, self.hidden,
                               self.topk, self.epr, self.rank, 256, per_tag)
        for peer in range(2):
            _publish[(1,)](self.ready[peer], self.rank, self.turn, num_warps=1)
        for peer in range(2):
            self.cuda.wait(self.ready[self.rank].data_ptr() + 4 * peer, self.turn, current)
        flat_ids = self.ids[self.rank].view(-1)
        order = torch.argsort(flat_ids, stable=True)
        offsets, total = expert_offsets(flat_ids, self.epr)
        rows = self.data[self.rank].view(-1, self.hidden).index_select(0, order)
        for peer in range(2):
            _publish[(1,)](self.consumed[peer], self.rank, self.turn, num_warps=1)
        self.pending = True
        return rows, offsets, total, order

    def combine(self, rows, total, order, weights, *, keep_rows=False):
        current = self._check_stream()
        if not self.pending:
            raise RuntimeError("dispatch must precede combine")
        if (rows.shape != (2 * self.p, self.hidden) or rows.dtype != torch.bfloat16 or
                rows.device != self.data[self.rank].device or not rows.is_contiguous() or rows.requires_grad or
                total.shape != () or total.dtype != torch.int32 or total.device != rows.device or not total.is_contiguous() or
                order.shape != (2 * self.p,) or order.dtype != torch.int64 or order.device != rows.device or not order.is_contiguous() or
                weights.shape != (self.tokens, self.topk) or weights.device != rows.device or
                weights.dtype not in (torch.float32, torch.float16, torch.bfloat16) or
                not weights.is_contiguous() or weights.requires_grad):
            raise ValueError("invalid combine inputs; prototype is forward-only")
        for peer in range(2):
            if self.turn > 1:
                self.cuda.wait(self.return_ack[self.rank].data_ptr() + 4 * peer, self.turn - 1, current)
        _return[(2 * self.p,)](rows, order, total, self.result_ptrs, self.p, self.hidden, 256)
        for peer in range(2):
            _publish[(1,)](self.returned[peer], self.rank, self.turn, num_warps=1)
        for peer in range(2):
            self.cuda.wait(self.returned[self.rank].data_ptr() + 4 * peer, self.turn, current)
        received = self.result[self.rank].view(self.tokens, self.topk, self.hidden)
        saved = received.clone() if keep_rows else None
        value = (received.float()
                 * weights.float().unsqueeze(-1)).sum(1).to(torch.bfloat16)
        for peer in range(2):
            _publish[(1,)](self.return_ack[peer], self.rank, self.turn, num_warps=1)
        self.pending = False
        return (value, saved) if keep_rows else value

    def close(self):
        if self.closed:
            return
        if self.pending or getattr(self, "training_pending", False):
            raise RuntimeError("cannot close while an exchange or training backward is outstanding")
        self.buffers.close()
        self.closed = True
