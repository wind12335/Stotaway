import torch
import triton
import triton.language as tl

from .runtime import SymmetricBuffers


@triton.jit
def _publish(flag, offset, value):
    pointer = flag.to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(pointer + offset, value, sem="release", scope="sys")


@triton.jit
def _sum_pair(output, own, remote, count: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    local_value = tl.load(own + index, index < count, 0).to(tl.float32)
    remote_value = tl.load(remote + index, index < count, 0).to(tl.float32)
    tl.store(output + index, (local_value + remote_value) * SCALE, index < count)


class ExchangeHandle:
    def __init__(self, event, device):
        self.event = event
        self.device = device

    def wait(self):
        torch.cuda.current_stream(self.device).wait_event(self.event)

    def is_completed(self):
        return self.event.query()


class ReduceScatter:
    def __init__(self, cuda, rank, world_size, numel, dtype, k_slots=1):
        if world_size != 2 or rank not in (0, 1) or numel <= 0 or numel % 2:
            raise ValueError("requires two ranks and a positive even element count")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("only float16 and bfloat16 are supported")
        if not isinstance(k_slots, int) or isinstance(k_slots, bool) or k_slots < 1:
            raise ValueError("k_slots must be a positive integer")
        self.cuda = cuda
        self.rank = rank
        self.peer = 1 - rank
        self.numel = numel
        self.shard = numel // 2
        self.dtype = dtype
        self.lane_count = k_slots
        self.planes = triton.cdiv(max(2, k_slots), k_slots) * k_slots
        self.buffers = SymmetricBuffers(rank, world_size, [
            ((self.planes, world_size, self.shard), dtype),
            ((self.planes, world_size), torch.int32),
            ((self.planes, world_size), torch.int32),
        ])
        self.slots, self.arrival, self.ack = self.buffers.buffers
        self.send = [torch.cuda.Stream() for _ in range(k_slots)]
        self.receive = [torch.cuda.Stream() for _ in range(k_slots)]
        self.tickets = torch.arange(1 << 22, dtype=torch.int32, device="cuda")
        self.round = 0
        self.closed = False
        self.caller_stream = None

    def __call__(self, value, output=None, *, defer_fence=False, scale=1.0, chunk_ev=None):
        if self.closed:
            raise RuntimeError("context is closed")
        if chunk_ev is not None:
            raise NotImplementedError("chunk_ev readiness is not supported")
        current = torch.cuda.current_stream()
        if self.caller_stream is None:
            self.caller_stream = current.cuda_stream
        elif self.caller_stream != current.cuda_stream:
            raise RuntimeError("a context must be used on the same caller stream")
        if (value.numel() != self.numel or value.dtype != self.dtype
                or not value.is_contiguous() or value.requires_grad
                or value.device != self.slots[self.rank].device):
            raise ValueError("invalid reduce-scatter input")
        if output is None:
            output = torch.empty(self.shard, dtype=self.dtype, device=value.device)
        if (output.numel() != self.shard or output.dtype != self.dtype
                or not output.is_contiguous() or output.requires_grad
                or output.device != value.device):
            raise ValueError("invalid reduce-scatter output")
        start = value.data_ptr()
        end = start + value.numel() * value.element_size()
        output_start = output.data_ptr()
        output_end = output_start + output.numel() * output.element_size()
        if output_start < end and start < output_end:
            own_start = start + self.rank * self.shard * value.element_size()
            if output_start != own_start or output_end != own_start + self.shard * value.element_size():
                raise ValueError("output may only alias this rank's input shard")
        turn = self.round + 1
        if turn >= self.tickets.numel():
            raise RuntimeError("ticket counter exhausted; recreate context")
        self.round = turn
        plane = turn % self.planes
        lane = turn % self.lane_count
        send = self.send[lane]
        receive = self.receive[lane]
        block_bytes = self.shard * value.element_size()
        slot_offset = (plane * 2 + self.rank) * block_bytes
        flag_offset = (plane * 2 + self.rank) * 4
        own_flag = (plane * 2 + self.peer) * 4
        ready = torch.cuda.Event()
        ready.record(current)
        with torch.cuda.stream(send):
            send.wait_event(ready)
            if turn > self.planes:
                self.cuda.wait(self.ack[self.rank].data_ptr() + own_flag,
                               turn - self.planes, send)
            self.cuda.copy(self.slots[self.peer] + slot_offset,
                           value.data_ptr() + self.peer * block_bytes,
                           block_bytes, send)
            self.cuda.copy(self.arrival[self.peer] + flag_offset,
                           self.tickets.data_ptr() + turn * 4, 4, send)
            sent = torch.cuda.Event()
            sent.record(send)
        with torch.cuda.stream(receive):
            receive.wait_event(ready)
            self.cuda.wait(self.arrival[self.rank].data_ptr() + own_flag, turn, receive)
            receive.wait_event(sent)
            _sum_pair[(triton.cdiv(self.shard, 1024),)](
                output, value.view(-1)[self.rank * self.shard:],
                self.slots[self.rank][plane, self.peer], self.shard,
                SCALE=float(scale), BLOCK=1024,
            )
            _publish[(1,)](self.ack[self.peer], plane * 2 + self.rank, turn, num_warps=1)
            done = torch.cuda.Event()
            done.record(receive)
        value.record_stream(send)
        value.record_stream(receive)
        output.record_stream(receive)
        handle = ExchangeHandle(done, value.device)
        if not defer_fence:
            handle.wait()
        return output, handle

    def close(self):
        if not self.closed:
            self.buffers.close()
            self.closed = True


class DirectExchange:
    def __init__(self, cuda, rank, world_size, rows, inner, columns, dtype, *, reuse_rs_scratch=False):
        if (world_size != 2 or rank not in (0, 1) or rows <= 0 or inner <= 0
                or columns <= 0 or rows % world_size):
            raise ValueError("requires two ranks and rows divisible by two")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("only float16 and bfloat16 are supported")
        if not isinstance(reuse_rs_scratch, bool):
            raise ValueError('reuse_rs_scratch must be bool')
        self.cuda = cuda
        self.rank = rank
        self.peer = 1 - rank
        self.rows = rows
        self.inner = inner
        self.columns = columns
        self.dtype = dtype
        self.block_rows = rows // world_size
        self.buffers = SymmetricBuffers(rank, world_size, [
            ((world_size, self.block_rows, columns), dtype),
            ((world_size, self.block_rows, inner), dtype),
            ((world_size,), torch.int32),
            ((world_size,), torch.int32),
            ((world_size,), torch.int32),
            ((world_size,), torch.int32),
        ])
        (self.rs_slots, self.ag_slots, self.rs_arrival, self.rs_ack,
         self.ag_arrival, self.ag_ack) = self.buffers.buffers
        self.send = torch.cuda.Stream()
        self.stage = torch.cuda.Stream()
        self.rs_round = 0
        self.ag_round = 0
        self.tickets = torch.arange(1 << 22, dtype=torch.int32, device="cuda")
        self.closed = False
        self.caller_stream = None
        # Opt-in experimental scratch: output remains independently allocated.
        # Every call waits for its send before returning control to this stream,
        # so the next call's writes are ordered after the preceding send.
        self.reuse_rs_scratch = reuse_rs_scratch
        self.rs_scratch = None

    def _next_round(self, field):
        if self.closed:
            raise RuntimeError("context is closed")
        stream = torch.cuda.current_stream().cuda_stream
        if self.caller_stream is None:
            self.caller_stream = stream
        elif self.caller_stream != stream:
            raise RuntimeError("a context must be used on the same caller stream")
        value = getattr(self, field) + 1
        if value >= self.tickets.numel():
            raise RuntimeError("ticket counter exhausted; recreate context")
        setattr(self, field, value)
        return value

    def gemm_reduce_scatter(self, activation, weight, *, output_layout='contiguous'):
        if output_layout not in ('contiguous', 'interleaved'):
            raise ValueError('unknown reduce-scatter output layout')
        if (activation.shape != (self.rows, self.inner)
                or weight.shape != (self.columns, self.inner)
                or activation.dtype != self.dtype or weight.dtype != self.dtype
                or not activation.is_contiguous() or not weight.is_contiguous()
                or activation.requires_grad or weight.requires_grad
                or activation.device != weight.device
                or activation.device != self.rs_slots[self.rank].device):
            raise ValueError("invalid GEMM+RS input shape, dtype, device or layout")
        turn = self._next_round("rs_round")
        current = torch.cuda.current_stream()
        if self.reuse_rs_scratch:
            if self.rs_scratch is None:
                self.rs_scratch = torch.empty((self.rows, self.columns), device=activation.device, dtype=self.dtype)
            partial = self.rs_scratch
        else:
            partial = torch.empty((self.rows, self.columns), device=activation.device, dtype=self.dtype)
        output = torch.empty((self.block_rows, self.columns), device=activation.device, dtype=self.dtype)
        remote_start = self.peer * self.block_rows
        own_start = self.rank * self.block_rows
        # Compute the peer's logical shard first to expose its copy while the
        # local shard computes. Interleaved mode produces rank::world rows
        # directly, avoiding a subsequent cross-rank row redistribution.
        remote_input = (activation[remote_start:remote_start + self.block_rows]
                        if output_layout == 'contiguous' else activation[self.peer::2])
        own_input = (activation[own_start:own_start + self.block_rows]
                     if output_layout == 'contiguous' else activation[self.rank::2])
        torch.matmul(remote_input, weight.t(),
                     out=partial[remote_start:remote_start + self.block_rows])
        ready = torch.cuda.Event()
        ready.record(current)
        torch.matmul(own_input, weight.t(),
                     out=partial[own_start:own_start + self.block_rows])
        block_bytes = self.block_rows * self.columns * partial.element_size()
        with torch.cuda.stream(self.send):
            self.send.wait_event(ready)
            if turn > 1:
                self.cuda.wait(self.rs_ack[self.rank].data_ptr() + self.peer * 4, turn - 1, self.send)
            self.cuda.copy(self.rs_slots[self.peer] + self.rank * block_bytes,
                           partial.data_ptr() + remote_start * self.columns * partial.element_size(),
                           block_bytes, self.send)
            self.cuda.copy(self.rs_arrival[self.peer] + self.rank * 4,
                           self.tickets.data_ptr() + turn * 4, 4, self.send)
        if not self.reuse_rs_scratch:
            partial.record_stream(self.send)
        self.cuda.wait(self.rs_arrival[self.rank].data_ptr() + self.peer * 4, turn, current)
        count = self.block_rows * self.columns
        _sum_pair[(triton.cdiv(count, 1024),)](
            output, partial[own_start:own_start + self.block_rows],
            self.rs_slots[self.rank][self.peer], count, SCALE=1.0, BLOCK=1024,
        )
        _publish[(1,)](self.rs_ack[self.peer], self.rank, turn, num_warps=1)
        done = torch.cuda.Event()
        done.record(self.send)
        current.wait_event(done)
        return output

    def all_gather_gemm(self, shard, weight, *, need_full=False):
        if (shard.shape != (self.block_rows, self.inner)
                or weight.shape != (self.columns, self.inner)
                or shard.dtype != self.dtype or weight.dtype != self.dtype
                or not shard.is_contiguous() or not weight.is_contiguous()
                or shard.requires_grad or weight.requires_grad
                or shard.device != weight.device
                or shard.device != self.ag_slots[self.rank].device):
            raise ValueError("invalid AG+GEMM input shape, dtype, device or layout")
        turn = self._next_round("ag_round")
        current = torch.cuda.current_stream()
        full = torch.empty((self.rows, self.inner), device=shard.device, dtype=self.dtype)
        output = torch.empty((self.rows, self.columns), device=shard.device, dtype=self.dtype)
        block_bytes = self.block_rows * self.inner * shard.element_size()
        ready = torch.cuda.Event()
        ready.record(current)
        with torch.cuda.stream(self.send):
            self.send.wait_event(ready)
            if turn > 1:
                self.cuda.wait(self.ag_ack[self.rank].data_ptr() + self.peer * 4, turn - 1, self.send)
            self.cuda.copy(self.ag_slots[self.peer] + self.rank * block_bytes,
                           shard.data_ptr(), block_bytes, self.send)
            self.cuda.copy(self.ag_arrival[self.peer] + self.rank * 4,
                           self.tickets.data_ptr() + turn * 4, 4, self.send)
        shard.record_stream(self.send)
        with torch.cuda.stream(self.stage):
            self.stage.wait_event(ready)
            self.cuda.wait(self.ag_arrival[self.rank].data_ptr() + self.peer * 4, turn, self.stage)
            self.cuda.copy(full.data_ptr() + self.peer * block_bytes,
                           self.ag_slots[self.rank].data_ptr() + self.peer * block_bytes,
                           block_bytes, self.stage)
            _publish[(1,)](self.ag_ack[self.peer], self.rank, turn, num_warps=1)
            staged = torch.cuda.Event()
            staged.record(self.stage)
        full.record_stream(self.stage)
        own_start = self.rank * self.block_rows
        remote_start = self.peer * self.block_rows
        torch.matmul(shard, weight.t(), out=output[own_start:own_start + self.block_rows])
        if need_full:
            full[own_start:own_start + self.block_rows].copy_(shard)
        current.wait_event(staged)
        torch.matmul(full[remote_start:remote_start + self.block_rows], weight.t(),
                     out=output[remote_start:remote_start + self.block_rows])
        done = torch.cuda.Event()
        done.record(self.send)
        current.wait_event(done)
        return (output, full) if need_full else output

    def all_gather_dgrad(self, grad_shard, weight, activation):
        if (grad_shard.shape != (self.block_rows, self.inner)
                or weight.shape != (self.inner, self.columns)
                or activation.shape != (self.rows, self.columns)
                or any(tensor.dtype != self.dtype or not tensor.is_contiguous()
                       or tensor.device != self.ag_slots[self.rank].device
                       or tensor.requires_grad for tensor in (grad_shard, weight, activation))):
            raise ValueError("invalid AG+dgrad input shape, dtype, device or layout")
        turn = self._next_round("ag_round")
        current = torch.cuda.current_stream()
        full = torch.empty((self.rows, self.inner), device=grad_shard.device, dtype=self.dtype)
        grad_input = torch.empty((self.rows, self.columns), device=grad_shard.device, dtype=self.dtype)
        block_bytes = self.block_rows * self.inner * grad_shard.element_size()
        ready = torch.cuda.Event()
        ready.record(current)
        with torch.cuda.stream(self.send):
            self.send.wait_event(ready)
            if turn > 1:
                self.cuda.wait(self.ag_ack[self.rank].data_ptr() + self.peer * 4, turn - 1, self.send)
            self.cuda.copy(self.ag_slots[self.peer] + self.rank * block_bytes,
                           grad_shard.data_ptr(), block_bytes, self.send)
            self.cuda.copy(self.ag_arrival[self.peer] + self.rank * 4,
                           self.tickets.data_ptr() + turn * 4, 4, self.send)
        grad_shard.record_stream(self.send)
        own_start = self.rank * self.block_rows
        remote_start = self.peer * self.block_rows
        full[own_start:own_start + self.block_rows].copy_(grad_shard)
        torch.matmul(full[own_start:own_start + self.block_rows], weight,
                     out=grad_input[own_start:own_start + self.block_rows])
        self.cuda.wait(self.ag_arrival[self.rank].data_ptr() + self.peer * 4, turn, current)
        self.cuda.copy(full.data_ptr() + self.peer * block_bytes,
                       self.ag_slots[self.rank].data_ptr() + self.peer * block_bytes,
                       block_bytes, current)
        torch.matmul(full[remote_start:remote_start + self.block_rows], weight,
                     out=grad_input[remote_start:remote_start + self.block_rows])
        _publish[(1,)](self.ag_ack[self.peer], self.rank, turn, num_warps=1)
        grad_weight = full.t().matmul(activation)
        done = torch.cuda.Event()
        done.record(self.send)
        current.wait_event(done)
        return grad_input, grad_weight, full

    def close(self):
        if not self.closed:
            self.buffers.close()
            self.rs_scratch = None
            self.closed = True
