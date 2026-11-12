"""Two-rank checks of the NVIDIA prototype against the DCU G2 predicates.

This is a new adapter-level harness, not execution of the original DCU
production kernels or its original benchmark. Uses fresh input in bursts.
"""
import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import finalize, initialize


def max_all(value):
    item = torch.tensor([value], dtype=torch.float32, device="cuda")
    dist.all_reduce(item, op=dist.ReduceOp.MAX)
    return item.item()


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    cuda = initialize()
    rows = int(os.environ.get("GATE_M", "2048"))
    inner = int(os.environ.get("GATE_K", "512"))
    columns = int(os.environ.get("GATE_N", "512"))
    ops = int(os.environ.get("GATE_OPS", "64"))
    burst = int(os.environ.get("GATE_BURST", "4"))
    if min(rows, inner, columns, ops, burst) <= 0 or rows % 2 or ops % burst:
        raise ValueError("positive sizes, even M, and ops divisible by burst required")
    dtype = torch.bfloat16
    forward = DirectExchange(cuda, rank, 2, rows, inner, columns, dtype)
    backward = DirectExchange(cuda, rank, 2, rows, columns, inner, dtype)
    gen = torch.Generator(device="cuda").manual_seed(433 + rank)
    weight = (torch.randn(columns, inner, device="cuda", dtype=dtype, generator=gen) * 0.05).contiguous()
    worst = {"rs": 0.0, "ag_full": 0.0, "ag_block": 0.0, "dgrad": 0.0,
             "dgrad_full": 0.0, "wgrad": 0.0}
    failures = 0
    for base in range(0, ops, burst):
        batch = []
        for turn in range(base, base + burst):
            activation = torch.randn(rows, inner, device="cuda", dtype=dtype, generator=gen) * 0.05
            shard = torch.randn(rows // 2, inner, device="cuda", dtype=dtype, generator=gen) * 0.05
            grad = torch.randn(rows // 2, columns, device="cuda", dtype=dtype, generator=gen) * 0.05
            if turn % 5 == rank:
                torch.cuda._sleep(20_000)
            rs = forward.gemm_reduce_scatter(activation, weight)
            ag, full = forward.all_gather_gemm(shard, weight, need_full=True)
            gi, gw, fg = backward.all_gather_dgrad(grad, weight, activation)
            batch.append((activation, shard, grad, rs, ag, full, gi, gw, fg))
        torch.cuda.synchronize()
        for activation, shard, grad, rs, ag, full, gi, gw, fg in batch:
            rs_ref = torch.empty_like(rs)
            dist.reduce_scatter_tensor(rs_ref, activation @ weight.t())
            ag_ref = torch.empty_like(full)
            dist.all_gather_into_tensor(ag_ref, shard)
            ag_block = torch.cat([part @ weight.t() for part in ag_ref.chunk(2)])
            grad_ref = torch.empty_like(fg)
            dist.all_gather_into_tensor(grad_ref, grad)
            gi_ref = grad_ref @ weight
            gw_ref = grad_ref.t() @ activation
            errors = {"rs": ((rs.float() - rs_ref.float()).abs() / rs_ref.float().abs().clamp_min(1)).max().item(),
                      "ag_full": (full.float() - ag_ref.float()).abs().max().item(),
                      "ag_block": (ag.float() - ag_block.float()).abs().max().item(),
                      "dgrad": ((gi.float() - gi_ref.float()).abs() / gi_ref.float().abs().clamp_min(1)).max().item(),
                      "dgrad_full": (fg.float() - grad_ref.float()).abs().max().item(),
                      "wgrad": (gw.float() - gw_ref.float()).abs().max().item()}
            for key, value in errors.items():
                worst[key] = max(worst[key], value)
            if (errors["rs"] >= .02 or errors["ag_full"] != 0 or errors["ag_block"] != 0
                    or errors["dgrad_full"] != 0 or errors["dgrad"] >= .02
                    or not torch.allclose(gw, gw_ref, rtol=.02, atol=.02)):
                failures += 1
    worst = {key: max_all(value) for key, value in worst.items()}
    failures = int(max_all(failures))
    print(f"rank={rank} M={rows} K={inner} N={columns} ops={ops} burst={burst} "
          f"failures={failures} worst={worst} {'PASS' if failures == 0 else 'FAIL'}", flush=True)
    backward.close()
    forward.close()
    finalize()
    if failures:
        raise AssertionError(f"G2-adapted correctness failed: {failures}")


if __name__ == "__main__":
    main()
