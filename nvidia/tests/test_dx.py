import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import finalize, initialize


def run_case(cuda, rank, dtype, rows, inner, columns, rounds, side_stream=False):
    context = DirectExchange(cuda, rank, 2, rows, inner, columns, dtype)
    generator = torch.Generator(device="cuda").manual_seed(431 + rank)
    weights = torch.randn((columns, inner), generator=generator, device="cuda", dtype=dtype) * 0.1
    results = []
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for turn in range(rounds):
            activation = torch.randn((rows, inner), generator=generator, device="cuda", dtype=dtype) * 0.1
            shard = torch.randn((rows // 2, inner), generator=generator, device="cuda", dtype=dtype) * 0.1
            if turn % 4 == rank:
                torch.cuda._sleep(50000)
            rs = context.gemm_reduce_scatter(activation, weights)
            ag = context.all_gather_gemm(shard, weights)
            results.append((activation, shard, rs, ag))
    torch.cuda.synchronize()
    max_rs = 0.0
    max_ag = 0.0
    for activation, shard, rs, ag in results:
        full = activation @ weights.t()
        expected_rs = torch.empty_like(rs)
        dist.reduce_scatter_tensor(expected_rs, full)
        gathered = torch.empty((rows, inner), device="cuda", dtype=dtype)
        dist.all_gather_into_tensor(gathered, shard)
        expected_ag = gathered @ weights.t()
        max_rs = max(max_rs, (rs.float() - expected_rs.float()).abs().max().item())
        max_ag = max(max_ag, (ag.float() - expected_ag.float()).abs().max().item())
        torch.testing.assert_close(rs, expected_rs, rtol=0.02, atol=0.02)
        torch.testing.assert_close(ag, expected_ag, rtol=0.02, atol=0.02)
    print(f"rank={rank} dtype={dtype} shape={rows},{inner},{columns} "
          f"rounds={rounds} side_stream={side_stream} max_rs={max_rs:.6g} max_ag={max_ag:.6g} PASS", flush=True)
    context.close()


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    cuda = initialize()
    rounds = int(os.environ.get("TEST_ROUNDS", "32"))
    if rounds < 2:
        raise ValueError("TEST_ROUNDS must be at least 2")
    shapes = ((64, 64, 96), (128, 96, 64), (1024, 256, 192))
    if os.environ.get("TEST_SOAK") == "1":
        shapes = shapes[:1]
    for dtype in (torch.float16, torch.bfloat16):
        for rows, inner, columns in shapes:
            for side_stream in ((False,) if os.environ.get("TEST_SOAK") == "1" else (False, True)):
                run_case(cuda, rank, dtype, rows, inner, columns, rounds, side_stream)
    finalize()


if __name__ == "__main__":
    main()
