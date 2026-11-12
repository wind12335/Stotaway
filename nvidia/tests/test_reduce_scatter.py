import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.dx import ReduceScatter
from stowaway_nvidia.runtime import finalize, initialize


def run_case(cuda, rank, dtype, numel, rounds, defer_fence, scale, alias=False, k_slots=1):
    context = ReduceScatter(cuda, rank, 2, numel, dtype, k_slots=k_slots)
    generator = torch.Generator(device="cuda").manual_seed(299 + rank)
    samples = []
    pending = []
    for turn in range(rounds):
        value = torch.randn(numel, device="cuda", dtype=dtype, generator=generator)
        original = value.clone()
        if turn % 7 == rank:
            torch.cuda._sleep(20000)
        output_buffer = value[rank * (numel // 2):(rank + 1) * (numel // 2)] if alias else None
        output, handle = context(value, output_buffer, defer_fence=defer_fence, scale=scale)
        if defer_fence:
            pending.append((value, original, output, handle))
            if turn % 8 == 7 or turn == rounds - 1:
                for pending_value, pending_original, pending_output, pending_handle in reversed(pending):
                    pending_handle.wait()
                    snapshot = pending_output.clone()
                    samples.append((pending_value, pending_original, snapshot))
                pending.clear()
        else:
            samples.append((value, original, output))
    torch.cuda.synchronize()
    worst = 0.0
    for value, original, output in samples:
        expected = torch.empty_like(output)
        dist.reduce_scatter_tensor(expected, original)
        expected = (expected.float() * scale).to(dtype)
        worst = max(worst, (output.float() - expected.float()).abs().max().item())
        if not alias:
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        else:
            untouched = 1 - rank
            torch.testing.assert_close(value[untouched * (numel // 2):(untouched + 1) * (numel // 2)],
                                       original[untouched * (numel // 2):(untouched + 1) * (numel // 2)], rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
    print(f"rank={rank} dtype={dtype} numel={numel} rounds={rounds} "
          f"deferred={defer_fence} scale={scale} alias={alias} lanes={k_slots} max_abs={worst:.6g} PASS", flush=True)
    context.close()


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    cuda = initialize()
    rounds = int(os.environ.get("TEST_ROUNDS", "128"))
    for dtype in (torch.float16, torch.bfloat16):
        for defer_fence, scale in ((False, 1.0), (True, 0.5)):
            run_case(cuda, rank, dtype, 8192, rounds, defer_fence, scale)
        run_case(cuda, rank, dtype, 8192, rounds, False, 1.0, alias=True)
        for lanes in (2, 4):
            run_case(cuda, rank, dtype, 8192, rounds, True, 1.0, k_slots=lanes)
            run_case(cuda, rank, dtype, 8192, rounds, True, 1.0, alias=True, k_slots=lanes)
    finalize()


if __name__ == "__main__":
    main()
