import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    cuda = initialize()
    rounds = int(os.environ.get("TEST_ROUNDS", "64"))
    for dtype in (torch.float16, torch.bfloat16):
        context = DirectExchange(cuda, rank, 2, 128, 96, 64, dtype)
        generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
        weight = torch.randn((96, 64), dtype=dtype, device="cuda", generator=generator) * 0.1
        results = []
        for turn in range(rounds):
            shard = torch.randn((64, 96), dtype=dtype, device="cuda", generator=generator) * 0.1
            activation = torch.randn((128, 64), dtype=dtype, device="cuda", generator=generator) * 0.1
            grad_input, grad_weight, full = context.all_gather_dgrad(shard, weight, activation)
            results.append((shard, activation, grad_input, grad_weight, full))
        torch.cuda.synchronize()
        for shard, activation, grad_input, grad_weight, full in results:
            expected = torch.empty_like(full)
            dist.all_gather_into_tensor(expected, shard)
            torch.testing.assert_close(full, expected, rtol=0, atol=0)
            torch.testing.assert_close(grad_input, expected @ weight, rtol=0.02, atol=0.02)
            torch.testing.assert_close(grad_weight, expected.t() @ activation, rtol=0.02, atol=0.02)
        print(f"rank={rank} dtype={dtype} rounds={rounds} AG+dgrad PASS", flush=True)
        context.close()
    finalize()


if __name__ == "__main__":
    main()
