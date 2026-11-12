"""Exercise Megatron TP BRS with alternating producers and consumers."""
import datetime
import os

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel import layers, mappings

from stowaway_nvidia.dx import ReduceScatter
from stowaway_nvidia.megatron_brs import TensorParallelBRS
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    adapter = TensorParallelBRS(cuda, k_slots=2)
    adapter.install()
    original_call = ReduceScatter.__call__

    def delayed_call(self, *args, **kwargs):
        torch.cuda._sleep(3_000_000)
        return original_call(self, *args, **kwargs)

    ReduceScatter.__call__ = delayed_call
    streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]
    rounds = int(os.environ.get("TEST_ROUNDS", "32"))
    snapshots = []
    try:
        for turn in range(rounds):
            producer = streams[turn % 2]
            async_op = turn % 2 == 1
            operation = layers.dist_reduce_scatter_func if turn % 4 < 2 else mappings.dist_reduce_scatter_func
            with torch.cuda.stream(producer):
                value = torch.full((8192,), rank + turn % 8 + 1, dtype=torch.bfloat16, device="cuda")
                output = torch.empty(4096, dtype=value.dtype, device=value.device)
                handle = operation(output, value, group=dist.group.WORLD, async_op=async_op)
                if not async_op:
                    assert handle is None
                    snapshot = output.clone()
            if async_op:
                assert handle is not None
                with torch.cuda.stream(streams[2]):
                    handle.wait()
                    snapshot = output.clone()
            snapshots.append((snapshot, 2 * (turn % 8) + 3))
        torch.cuda.synchronize()
        for snapshot, expected in snapshots:
            torch.testing.assert_close(snapshot, torch.full_like(snapshot, expected), rtol=0, atol=0)
        assert sum(adapter.hits.values()) == rounds and sum(adapter.fallbacks.values()) == 0
        print(f"rank={rank} TP BRS alternating streams rounds={rounds} hits={adapter.hits} PASS", flush=True)
    finally:
        ReduceScatter.__call__ = original_call
        adapter.close()
        finalize()


if __name__ == "__main__":
    main()
