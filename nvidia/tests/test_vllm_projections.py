"""Exercise vLLM adapter projections without importing vLLM itself."""
import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.runtime import finalize, initialize
from stowaway_nvidia.vllm_sp import LlamaSequenceParallel


class Method:
    def apply(self, layer, value, bias):
        assert bias is None
        return value @ layer.weight.t()


class Layer:
    def __init__(self, inner, columns, dtype, generator):
        self.weight = torch.randn(columns, inner, device="cuda", dtype=dtype, generator=generator)
        self.quant_method = Method()


class TestAdapter(LlamaSequenceParallel):
    @staticmethod
    def _weight(layer):
        return layer.weight


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    rounds = int(os.environ.get("TEST_ROUNDS", "128"))
    for mode in ("stock_sp", "dx"):
        adapter = TestAdapter(cuda, rank, mode, [128])
        generator = torch.Generator(device="cuda").manual_seed(145 + rank)
        for dtype in (torch.float16, torch.bfloat16):
            for turn in range(rounds):
                for inner, columns in ((64, 96), (96, 64)):
                    layer = Layer(inner, columns, dtype, generator)
                    shard = torch.randn(64, inner, device="cuda", dtype=dtype, generator=generator)
                    full = torch.empty((128, inner), device="cuda", dtype=dtype)
                    dist.all_gather_into_tensor(full, shard)
                    actual = adapter._col(layer, shard, 128, dist.group.WORLD)
                    torch.testing.assert_close(actual, full @ layer.weight.t(), rtol=0.02, atol=0.02)
                    value = torch.randn(128, inner, device="cuda", dtype=dtype, generator=generator)
                    expected = torch.empty((64, columns), device="cuda", dtype=dtype)
                    dist.reduce_scatter_tensor(expected, value @ layer.weight.t())
                    actual = adapter._row(layer, value, 128, dist.group.WORLD)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            print(f"rank={rank} mode={mode} dtype={dtype} rounds={rounds} projection PASS", flush=True)
        adapter.close()
    finalize()


if __name__ == "__main__":
    main()
