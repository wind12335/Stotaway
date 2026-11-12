"""Verify Inductor's private micro-pipeline TP rewrite in an isolated example."""
import datetime
import os

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory as symm_mem
import torch._inductor.config as config
from torch._inductor.utils import run_and_get_code


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    if dist.get_world_size() != 2:
        raise ValueError("requires two ranks")
    group = dist.group.WORLD
    symm_mem.enable_symm_mem_for_group(group.group_name)
    config._micro_pipeline_tp = True
    config.reorder_for_compute_comm_overlap = False

    def ag_mm(x, weight):
        return funcol.all_gather_tensor(x, 0, group) @ weight

    def mm_rs(x, weight):
        return funcol.reduce_scatter_tensor(x @ weight, "sum", 0, group)

    torch.manual_seed(80 + rank)
    shard = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    full = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 192, device="cuda", dtype=torch.bfloat16)
    for name, function, x, fused_name in (
        ("ag_mm", ag_mm, shard, "fused_all_gather_matmul"),
        ("mm_rs", mm_rs, full, "fused_matmul_reduce_scatter"),
    ):
        expected = funcol.wait_tensor(function(x, weight))
        actual, sources = run_and_get_code(torch.compile(function, fullgraph=True), x, weight)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        if not any(fused_name in source for source in sources):
            raise AssertionError(f"{name}: generated code does not call {fused_name}")
        print(f"rank={rank} case={name} generated={fused_name} correctness=PASS", flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
