"""Exercise Megatron's actual bucket-group coalescer with multiple DX events."""
import datetime
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

from stowaway_nvidia.megatron_optimizer_brs import OptimizerBucketBRS
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    adapter = OptimizerBucketBRS(cuda, min_bytes=int(os.environ.get("TEST_MIN_BYTES", "128")), k_slots=2)
    adapter.install()
    generator = torch.Generator(device="cuda").manual_seed(311 + rank)
    rounds = int(os.environ.get("TEST_ROUNDS", "64"))
    for async_ops in (False, True):
        for average in (False, True):
            config = SimpleNamespace(use_distributed_optimizer=True, overlap_param_gather=False,
                                     num_distributed_optimizer_instances=1,
                                     reduce_scatter_with_fp32_accumulation=False,
                                     check_for_nan_in_grad=False, check_for_large_grads=False,
                                     average_in_collective=average, overlap_grad_reduce=async_ops)
            buckets = [SimpleNamespace(params_list=[], params_with_extra_main_grads=[],
                                       grad_data=torch.empty(size, dtype=torch.bfloat16, device="cuda"),
                                       gradient_scaling_factor=1.0) for size in (1024, 2048, 32)]
            group = _ParamAndGradBucketGroup(buckets, config, dist.group.WORLD, 2)
            for turn in range(rounds):
                references = []
                for bucket in buckets:
                    bucket.grad_data.normal_(generator=generator)
                    expected = torch.empty(bucket.grad_data.numel() // 2, dtype=torch.bfloat16, device="cuda")
                    dist.reduce_scatter_tensor(expected, bucket.grad_data, op=dist.ReduceOp.AVG if average else dist.ReduceOp.SUM)
                    references.append(expected)
                group.start_grad_sync()
                if async_ops:
                    assert group.grad_reduce_handle is not None
                    # Exercise Megatron's actual consumer, including first-batch
                    # idempotence and its post-wait bookkeeping.
                    group.finish_grad_sync()
                for bucket, expected in zip(buckets, references):
                    output = bucket.grad_data.chunk(2)[rank]
                    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
                group.reset()
            print(f"rank={rank} async={async_ops} average={average} rounds={rounds} "
                  f"multiple_buckets min_bytes={adapter.min_bytes} PASS", flush=True)
    assert adapter.multi_bucket_batches == 4 * rounds
    if adapter.min_bytes == 0:
        assert adapter.calls == 12 * rounds and adapter.fallbacks == 0
    elif adapter.min_bytes == 128:
        assert adapter.calls == 8 * rounds and adapter.fallbacks == 4 * rounds
    adapter.close()
    finalize()


if __name__ == "__main__":
    main()
