"""Two-rank correctness probe for PyTorch 2.7's private symmetric-memory ops.

This exercises the operators directly, not Inductor's Async-TP graph rewrite.
Run with run_tests.sh in the standard-Triton Megatron environment.
"""
import datetime
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    if dist.get_world_size() != 2:
        raise ValueError("this probe requires exactly two ranks")
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    generator = torch.Generator(device="cuda").manual_seed(412 + rank)
    rounds = int(os.environ.get("TEST_ROUNDS", "32"))
    rows = int(os.environ.get("TEST_ROWS", "256"))
    inner = int(os.environ.get("TEST_INNER", "128"))
    columns = int(os.environ.get("TEST_COLUMNS", "192"))
    if rows % 2:
        raise ValueError("TEST_ROWS must be divisible by two")
    for dtype in (torch.float16, torch.bfloat16):
        worst_ag = worst_rs = 0.0
        for turn in range(rounds):
            shard = torch.randn(rows // 2, inner, dtype=dtype, device="cuda", generator=generator)
            weight = torch.randn(inner, columns, dtype=dtype, device="cuda", generator=generator)
            full = torch.empty(rows, inner, dtype=dtype, device="cuda")
            dist.all_gather_into_tensor(full, shard)
            gathered, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                shard, [weight], 0, group_name, return_A=True)
            torch.testing.assert_close(gathered, full, rtol=0, atol=0)
            expected_ag = full @ weight
            worst_ag = max(worst_ag, (outputs[0].float() - expected_ag.float()).abs().max().item())
            torch.testing.assert_close(outputs[0], expected_ag, rtol=0.02, atol=0.02)

            value = torch.randn(rows, inner, dtype=dtype, device="cuda", generator=generator)
            partial = value @ weight
            expected_rs = torch.empty(rows // 2, columns, dtype=dtype, device="cuda")
            dist.reduce_scatter_tensor(expected_rs, partial)
            actual_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                value, weight, "sum", 0, group_name)
            worst_rs = max(worst_rs, (actual_rs.float() - expected_rs.float()).abs().max().item())
            torch.testing.assert_close(actual_rs, expected_rs, rtol=0.02, atol=0.02)
        print(f"rank={rank} dtype={dtype} shape={rows}x{inner}x{columns} "
              f"rounds={rounds} ag_max_abs={worst_ag:g} rs_max_abs={worst_rs:g} PASS", flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
