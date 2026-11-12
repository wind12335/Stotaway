"""Experimental same-host multi-peer direct exchange, with bounded reuse.

All ranks must create/call/close contexts in the same order on one caller stream.
This is a correctness-first TP8 prototype, not the original DCU scheduling.
"""
import os
import torch
import triton
import triton.language as tl

from .runtime import SymmetricBuffers
from .dx import _publish


@triton.jit
def _sum_all(out, own, slots, COUNT: tl.constexpr, WORLD: tl.constexpr,
             RANK: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(own + ix, ix < COUNT, 0).to(tl.float32)
    for peer in tl.static_range(WORLD):
        if peer != RANK:
            total += tl.load(slots + peer * COUNT + ix, ix < COUNT, 0).to(tl.float32)
    tl.store(out + ix, total, ix < COUNT)


@triton.jit
def _publish_peers(p0, p1, p2, p3, p4, p5, p6, p7,
                   RANK: tl.constexpr, VALUE, WORLD: tl.constexpr):
    for peer in tl.static_range(WORLD):
        if peer != RANK:
            if peer == 0: ptr = p0
            elif peer == 1: ptr = p1
            elif peer == 2: ptr = p2
            elif peer == 3: ptr = p3
            elif peer == 4: ptr = p4
            elif peer == 5: ptr = p5
            elif peer == 6: ptr = p6
            else: ptr = p7
            tl.atomic_xchg(ptr.to(tl.pointer_type(tl.int32)) + RANK, VALUE,
                           sem="release", scope="sys")


class DirectExchange8:
    def __init__(self, cuda, rank, world, rows, inner, columns, dtype):
        if world != 8 or rank not in range(world) or rows <= 0 or rows % world or min(inner, columns) <= 0:
            raise ValueError("requires world=8 and positive dimensions with M divisible by 8")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("only BF16 and FP16 supported")
        self.cuda, self.rank, self.world = cuda, rank, world
        self.rows, self.inner, self.columns, self.dtype = rows, inner, columns, dtype
        self.schedule = os.environ.get("STOWAWAY_DX8_SCHEDULE", "fixed")
        self.batch_ack = os.environ.get("STOWAWAY_DX8_BATCH_ACK", "0") == "1"
        if self.schedule not in ("fixed", "shift"):
            raise ValueError("STOWAWAY_DX8_SCHEDULE must be fixed or shift")
        self.peers = ([p for p in range(world) if p != rank] if self.schedule == "fixed"
                      else [(rank + offset) % world for offset in range(1, world)])
        self.block = rows // world
        self.buffers = SymmetricBuffers(rank, world, [
            ((world, self.block, inner), dtype), ((world, self.block, columns), dtype),
            ((world,), torch.int32), ((world,), torch.int32),
            ((world,), torch.int32), ((world,), torch.int32),
        ])
        self.ag, self.rs, self.ag_ready, self.ag_ack, self.rs_ready, self.rs_ack = self.buffers.buffers
        self.send = torch.cuda.Stream()
        self.tickets = torch.arange(1 << 20, dtype=torch.int32, device="cuda")
        self.ag_turn = self.rs_turn = 0
        self.stream = None
        self.closed = False

    def _turn(self, name):
        if self.closed: raise RuntimeError("closed context")
        current = torch.cuda.current_stream()
        if self.stream is None: self.stream = current.cuda_stream
        if self.stream != current.cuda_stream: raise RuntimeError("same caller stream required")
        turn = getattr(self, name) + 1
        if turn >= self.tickets.numel(): raise RuntimeError("ticket limit reached")
        setattr(self, name, turn)
        return current, turn

    def _validate(self, value, weight, rows):
        if (value.shape != (rows, self.inner) or weight.shape != (self.columns, self.inner)
                or value.dtype != self.dtype or weight.dtype != self.dtype
                or not value.is_contiguous() or not weight.is_contiguous()
                or value.requires_grad or weight.requires_grad or value.device != weight.device
                or value.device != self.ag[self.rank].device):
            raise ValueError("invalid shape, dtype, layout or device")

    def all_gather_gemm(self, shard, weight):
        self._validate(shard, weight, self.block)
        current, turn = self._turn("ag_turn")
        output = torch.empty((self.rows, self.columns), dtype=self.dtype, device=shard.device)
        tmp = torch.empty((self.block, self.inner), dtype=self.dtype, device=shard.device)
        size = shard.numel() * shard.element_size()
        start = torch.cuda.Event(); start.record(current)
        with torch.cuda.stream(self.send):
            self.send.wait_event(start)
            for peer in self.peers:
                if turn > 1: self.cuda.wait(self.ag_ack[self.rank].data_ptr() + peer * 4, turn - 1, self.send)
                self.cuda.copy(self.ag[peer] + self.rank * size, shard.data_ptr(), size, self.send)
                self.cuda.copy(self.ag_ready[peer] + self.rank * 4, self.tickets.data_ptr() + turn * 4, 4, self.send)
            done = torch.cuda.Event(); done.record(self.send)
        shard.record_stream(self.send)
        torch.matmul(shard, weight.t(), out=output[self.rank*self.block:(self.rank+1)*self.block])
        for peer in self.peers:
            self.cuda.wait(self.ag_ready[self.rank].data_ptr() + peer * 4, turn, current)
            tmp.copy_(self.ag[self.rank][peer])
            _publish[(1,)](self.ag_ack[peer], self.rank, turn, num_warps=1)
            torch.matmul(tmp, weight.t(), out=output[peer*self.block:(peer+1)*self.block])
        current.wait_event(done)
        return output

    def all_gather_dgrad(self, grad_shard, weight, activation):
        """Experimental row-backward AG+dgrad; caller supplies a separate context."""
        if (grad_shard.shape != (self.block, self.inner)
                or weight.shape != (self.inner, self.columns)
                or activation.shape != (self.rows, self.columns)
                or any(t.dtype != self.dtype or not t.is_contiguous()
                       or t.device != self.ag[self.rank].device or t.requires_grad
                       for t in (grad_shard, weight, activation))):
            raise ValueError("invalid AG+dgrad inputs")
        current, turn = self._turn('ag_turn')
        full = torch.empty((self.rows, self.inner), dtype=self.dtype, device=grad_shard.device)
        grad_input = torch.empty((self.rows, self.columns), dtype=self.dtype, device=grad_shard.device)
        size = grad_shard.numel() * grad_shard.element_size()
        start = torch.cuda.Event(); start.record(current)
        with torch.cuda.stream(self.send):
            self.send.wait_event(start)
            for peer in self.peers:
                if turn > 1:
                    self.cuda.wait(self.ag_ack[self.rank].data_ptr() + peer * 4, turn - 1, self.send)
                self.cuda.copy(self.ag[peer] + self.rank * size, grad_shard.data_ptr(), size, self.send)
                self.cuda.copy(self.ag_ready[peer] + self.rank * 4, self.tickets.data_ptr() + turn * 4, 4, self.send)
            done = torch.cuda.Event(); done.record(self.send)
        grad_shard.record_stream(self.send)
        own = slice(self.rank*self.block, (self.rank+1)*self.block)
        full[own].copy_(grad_shard)
        torch.matmul(full[own], weight, out=grad_input[own])
        for peer in self.peers:
            sl = slice(peer*self.block, (peer+1)*self.block)
            self.cuda.wait(self.ag_ready[self.rank].data_ptr() + peer * 4, turn, current)
            full[sl].copy_(self.ag[self.rank][peer])
            _publish[(1,)](self.ag_ack[peer], self.rank, turn, num_warps=1)
            torch.matmul(full[sl], weight, out=grad_input[sl])
        grad_weight = full.t().matmul(activation)
        current.wait_event(done)
        return grad_input, grad_weight, full

    def gemm_reduce_scatter(self, activation, weight):
        self._validate(activation, weight, self.rows)
        current, turn = self._turn("rs_turn")
        partial = torch.empty((self.rows, self.columns), dtype=self.dtype, device=activation.device)
        output = torch.empty((self.block, self.columns), dtype=self.dtype, device=activation.device)
        size = self.block * self.columns * activation.element_size()
        for peer in self.peers:
            sl = slice(peer*self.block, (peer+1)*self.block)
            torch.matmul(activation[sl], weight.t(), out=partial[sl])
            ready = torch.cuda.Event(); ready.record(current)
            with torch.cuda.stream(self.send):
                self.send.wait_event(ready)
                if turn > 1: self.cuda.wait(self.rs_ack[self.rank].data_ptr() + peer * 4, turn - 1, self.send)
                self.cuda.copy(self.rs[peer] + self.rank * size, partial[sl].data_ptr(), size, self.send)
                self.cuda.copy(self.rs_ready[peer] + self.rank * 4, self.tickets.data_ptr() + turn * 4, 4, self.send)
        own = partial[self.rank*self.block:(self.rank+1)*self.block]
        torch.matmul(activation[self.rank*self.block:(self.rank+1)*self.block], weight.t(), out=own)
        done = torch.cuda.Event(); done.record(self.send)
        for peer in range(self.world):
            if peer != self.rank: self.cuda.wait(self.rs_ready[self.rank].data_ptr() + peer * 4, turn, current)
        count = own.numel()
        _sum_all[(triton.cdiv(count, 1024),)](output, own, self.rs[self.rank], count, self.world, self.rank, 1024)
        if self.batch_ack:
            _publish_peers[(1,)](*self.rs_ack, self.rank, turn, self.world, num_warps=1)
        else:
            for peer in range(self.world):
                if peer != self.rank: _publish[(1,)](self.rs_ack[peer], self.rank, turn, num_warps=1)
        partial.record_stream(self.send)
        current.wait_event(done)
        return output

    def close(self):
        if not self.closed:
            self.buffers.close()
            self.closed = True
