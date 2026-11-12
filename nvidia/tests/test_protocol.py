import datetime
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from stowaway_nvidia.runtime import CudaOps, IpcBuffer, SymmetricBuffers, finalize, initialize


@triton.jit(do_not_specialize=["turn"])
def write_pattern(data, turn, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = index.to(tl.int32) ^ (turn * 7919)
    tl.store(data + index, value, index < count)


@triton.jit(do_not_specialize=["address", "turn"])
def check_pattern(address, errors, turn, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pointer = address.to(tl.pointer_type(tl.int32))
    actual = tl.load(pointer + index, index < count, 0)
    expected = index.to(tl.int32) ^ (turn * 7919)
    bad = tl.sum(((actual != expected) & (index < count)).to(tl.int64), 0)
    tl.atomic_add(errors, bad)


@triton.jit(do_not_specialize=["address", "turn"])
def publish_ack(address, turn):
    pointer = address.to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(pointer, turn, sem="release", scope="sys")


def main():
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or int(os.environ["LOCAL_WORLD_SIZE"]) != 2:
        raise ValueError("requires two local processes")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    transport = os.environ.get("PROTOCOL_TRANSPORT", "ipc")
    if transport not in ("ipc", "nvshmem"):
        raise ValueError("PROTOCOL_TRANSPORT must be ipc or nvshmem")
    dist.init_process_group("gloo" if transport == "ipc" else "nccl", timeout=datetime.timedelta(seconds=180))
    cuda = CudaOps() if transport == "ipc" else initialize()
    rounds = int(os.environ.get("PROTOCOL_ROUNDS", "10000"))
    size_mb = int(os.environ.get("PROTOCOL_MB", "1"))
    if rounds < 1 or rounds >= 2**31 or size_mb < 1 or size_mb > 64:
        raise ValueError("invalid rounds or size")
    size = size_mb << 20
    if transport == "ipc":
        buffer = IpcBuffer(cuda, size + 8, dist.group.WORLD)
        local, peer = buffer.local.value, buffer.peer.value
    else:
        buffer = SymmetricBuffers(rank, 2, [((size + 8,), torch.uint8)])
        local = buffer.buffers[0][rank].data_ptr()
        peer = buffer.buffers[0][1 - rank]
    source = torch.empty(size // 4, dtype=torch.int32, device="cuda")
    tickets = torch.arange(rounds + 1, dtype=torch.int32, device="cuda")
    errors = torch.zeros(1, dtype=torch.int64, device="cuda")
    flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    cuda.copy(local + size, flags.data_ptr(), 8, torch.cuda.current_stream())
    torch.cuda.synchronize()
    dist.barrier()
    stream = torch.cuda.current_stream()
    for turn in range(1, rounds + 1):
        if rank == 0:
            if turn > 1:
                cuda.wait(local + size + 4, turn - 1, stream)
            write_pattern[(triton.cdiv(source.numel(), 1024),)](source, turn, source.numel(), BLOCK=1024)
            cuda.copy(peer, source.data_ptr(), size, stream)
            cuda.copy(peer + size, tickets.data_ptr() + turn * 4, 4, stream)
        else:
            cuda.wait(local + size, turn, stream)
            check_pattern[(triton.cdiv(source.numel(), 1024),)](
                local, errors, turn, source.numel(), BLOCK=1024
            )
            publish_ack[(1,)](peer + size + 4, turn, num_warps=1)
    torch.cuda.synchronize()
    total_errors = errors.item()
    counts = [None, None]
    dist.all_gather_object(counts, total_errors)
    print(f"rank={rank} transport={transport} size_mb={size_mb} rounds={rounds} mismatched_elements={counts} "
          f"{'PASS' if sum(counts) == 0 else 'FAIL'}", flush=True)
    buffer.close()
    if transport == "nvshmem":
        finalize()
    else:
        dist.destroy_process_group()
    if sum(counts):
        raise AssertionError(f"data corruption: {counts}")


if __name__ == "__main__":
    main()
