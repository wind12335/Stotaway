import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    rounds = int(os.environ.get("TEST_ROUNDS", "128"))
    rows, inner, columns = 256, 192, 128
    for dtype in (torch.float16, torch.bfloat16):
        exchange = DirectExchange(cuda, rank, 2, rows, inner, columns, dtype)
        generator = torch.Generator(device="cuda").manual_seed(333 + rank)
        weight = torch.randn((columns, inner), dtype=dtype, device="cuda", generator=generator) * 0.1
        buffer = torch.empty((rows // 2, inner), dtype=dtype, device="cuda")
        outputs = []
        samples = []
        for turn in range(rounds):
            data = torch.randn(buffer.shape, dtype=dtype, device="cuda", generator=generator) * 0.1
            if turn % 4 == rank:
                torch.cuda._sleep(30000)
            buffer.copy_(data)
            output, full = exchange.all_gather_gemm(buffer, weight, need_full=True)
            outputs.append((output, full))
            samples.append(data)
        torch.cuda.synchronize()
        for sample, (output, full) in zip(samples, outputs):
            reference = torch.empty((rows, inner), dtype=dtype, device="cuda")
            dist.all_gather_into_tensor(reference, sample)
            torch.testing.assert_close(full, reference, rtol=0, atol=0)
            block = torch.cat([reference[index * (rows // 2):(index + 1) * (rows // 2)].matmul(weight.t())
                               for index in range(2)])
            torch.testing.assert_close(output, block, rtol=0, atol=0)
        print(f"rank={rank} dtype={dtype} AG reused_input rounds={rounds} bit_exact=PASS", flush=True)
        exchange.close()
    finalize()


if __name__ == "__main__":
    main()
