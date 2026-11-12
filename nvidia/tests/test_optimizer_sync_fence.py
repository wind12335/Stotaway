"""Check that synchronous optimizer RS orders its consumer after DX work."""
import datetime
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

from stowaway_nvidia.dx import ReduceScatter
from stowaway_nvidia.megatron_optimizer_brs import OptimizerBucketBRS
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    adapter = OptimizerBucketBRS(cuda, k_slots=2)
    adapter.install()
    original_call = ReduceScatter.__call__

    def delayed_call(self, *args, **kwargs):
        # This is enqueued on the adapter's worker stream before its ready event.
        # The immediately enqueued caller-side copy must still see the final RS.
        torch.cuda._sleep(30_000_000)
        return original_call(self, *args, **kwargs)

    ReduceScatter.__call__ = delayed_call
    try:
        config = SimpleNamespace(use_distributed_optimizer=True, overlap_param_gather=False,
                                 num_distributed_optimizer_instances=1,
                                 reduce_scatter_with_fp32_accumulation=False,
                                 check_for_nan_in_grad=False, check_for_large_grads=False,
                                 average_in_collective=False, overlap_grad_reduce=False)
        rounds = int(os.environ.get("TEST_ROUNDS", "32"))
        for size in (2048, 8192):
            bucket = SimpleNamespace(params_list=[], params_with_extra_main_grads=[],
                                     grad_data=torch.empty(size, dtype=torch.bfloat16, device="cuda"),
                                     gradient_scaling_factor=1.0)
            group = _ParamAndGradBucketGroup([bucket], config, dist.group.WORLD, 2)
            for turn in range(rounds):
                # Each rank contributes distinct data on every iteration.
                bucket.grad_data.fill_(rank + turn + 1)
                group.start_grad_sync()
                actual = bucket.grad_data.chunk(2)[rank].clone()
                expected = 2 * turn + 3
                torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=0, atol=0)
                group.reset()
        assert adapter.calls == rounds * 2 and adapter.fallbacks == 0
        print(f"rank={rank} synchronous delayed worker rounds={rounds} PASS", flush=True)
    finally:
        ReduceScatter.__call__ = original_call
        adapter.close()
        finalize()


if __name__ == "__main__":
    main()
