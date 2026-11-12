"""Experimental single-TP-group NVSHMEM reduce-scatter for eight ranks."""
import torch
import triton
import triton.language as tl
from .dx import ExchangeHandle, _publish
from .runtime import SymmetricBuffers


@triton.jit
def _reduce(out, own, remote, COUNT: tl.constexpr, RANK: tl.constexpr,
            SCALE: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(own + ix, ix < COUNT, 0).to(tl.float32)
    for peer in tl.static_range(8):
        if peer != RANK:
            val += tl.load(remote + peer * COUNT + ix, ix < COUNT, 0).to(tl.float32)
    tl.store(out + ix, val * SCALE, ix < COUNT)


class ReduceScatter8:
    def __init__(self, cuda, rank, world_size, numel, dtype, k_slots=1):
        if world_size != 8 or rank not in range(8) or numel <= 0 or numel % 8:
            raise ValueError('requires eight ranks and a positive divisible element count')
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('requires float16 or bfloat16')
        self.cuda, self.rank, self.numel, self.dtype = cuda, rank, numel, dtype
        self.shard = numel // 8
        self.buffers = SymmetricBuffers(rank, 8, [((8, self.shard), dtype),
                                                 ((8,), torch.int32), ((8,), torch.int32)])
        self.slots, self.ready, self.ack = self.buffers.buffers
        self.send = torch.cuda.Stream()
        self.tickets = torch.arange(1 << 20, dtype=torch.int32, device='cuda')
        self.turn = 0
        self.stream = None
        self.closed = False
        self.peers = [(rank + offset) % 8 for offset in range(1, 8)]

    def __call__(self, value, output=None, *, defer_fence=False, scale=1.0):
        if self.closed or value.numel() != self.numel or value.dtype != self.dtype or not value.is_contiguous() or value.requires_grad:
            raise ValueError('invalid input or closed context')
        if output is None:
            output = torch.empty(self.shard, dtype=self.dtype, device=value.device)
        if output.numel() != self.shard or output.dtype != self.dtype or not output.is_contiguous() or output.requires_grad or output.device != value.device:
            raise ValueError('invalid output')
        start, end = value.data_ptr(), value.data_ptr() + value.numel()*value.element_size()
        out_start = output.data_ptr()
        if out_start < end and out_start + output.numel()*output.element_size() > start:
            raise ValueError('aliased output is unsupported')
        current = torch.cuda.current_stream(value.device)
        if self.stream is None: self.stream = current.cuda_stream
        if self.stream != current.cuda_stream: raise RuntimeError('same caller stream required')
        self.turn += 1
        if self.turn >= self.tickets.numel(): raise RuntimeError('ticket limit reached')
        turn = self.turn
        size = self.shard * value.element_size()
        ready = torch.cuda.Event(); ready.record(current)
        with torch.cuda.stream(self.send):
            self.send.wait_event(ready)
            for peer in self.peers:
                if turn > 1:
                    self.cuda.wait(self.ack[self.rank].data_ptr()+peer*4, turn-1, self.send)
                self.cuda.copy(self.slots[peer]+self.rank*size, value.data_ptr()+peer*size, size, self.send)
                self.cuda.copy(self.ready[peer]+self.rank*4, self.tickets.data_ptr()+turn*4, 4, self.send)
            sent = torch.cuda.Event(); sent.record(self.send)
        for peer in self.peers:
            self.cuda.wait(self.ready[self.rank].data_ptr()+peer*4, turn, current)
        own = value.view(-1)[self.rank*self.shard:(self.rank+1)*self.shard]
        _reduce[(triton.cdiv(self.shard,1024),)](output, own, self.slots[self.rank],
                                                   self.shard, self.rank, float(scale), 1024)
        for peer in self.peers:
            _publish[(1,)](self.ack[peer], self.rank, turn, num_warps=1)
        current.wait_event(sent)
        value.record_stream(self.send)
        done = torch.cuda.Event(); done.record(current)
        handle = ExchangeHandle(done, value.device)
        if not defer_fence: handle.wait()
        return output, handle

    def close(self):
        if not self.closed:
            self.buffers.close(); self.closed = True
