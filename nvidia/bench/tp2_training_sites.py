"""Measure full forward/backward site work for frozen TP2 planner decisions.

Reference follows the relevant unfused Megatron local SP communication order;
it does not include optimizer hooks, activation checkpointing, or a full step.
"""
import datetime
import time
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import triton

from stowaway_nvidia.autograd import column_parallel, row_parallel
from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import finalize, initialize
from bench.tp2_cells import configured_sites


def timed(fn, rounds, batch):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch):
            fn()
        end.record()
        end.synchronize()
        value = torch.tensor([start.elapsed_time(end) / batch], device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        samples.append(value.item())
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def main():
    total_start = time.monotonic()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
    cuda = initialize()
    m = int(os.environ.get("TRAIN_CELL_M", "4096"))
    repeats = int(os.environ.get("TRAIN_CELL_REPEATS", "6"))
    batch = int(os.environ.get("TRAIN_CELL_BATCH", "4"))
    checks = int(os.environ.get("TRAIN_CELL_CHECKS", "2"))
    seed = int(os.environ.get("TRAIN_CELL_SEED", "20261110"))
    if m <= 0 or m % 2 or min(repeats, batch, checks) < 1:
        raise ValueError("invalid measurement settings")
    gen = torch.Generator(device="cuda").manual_seed(881 + rank)
    metadata = {"tp": 2, "dtype": "bfloat16", "gpu": torch.cuda.get_device_name(),
                "p2p_bidirectional": torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0),
                "torch": torch.__version__, "triton": triton.__version__, "cuda_build": torch.version.cuda,
                "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset"),
                "M": m, "repeats": repeats, "batch": batch, "checks": checks, "seed": seed, "warmup": 3,
                "python": sys.executable, "classification": "diagnostic unfused site forward+backward"}
    output = {"metadata": metadata, "results": []}
    for model, site, op, _, k, n in configured_sites(m):
            x_shape = (m // 2, k) if op == "ag" else (m, k)
            y_shape = (m, n) if op == "ag" else (m // 2, n)
            x = (torch.randn(x_shape, device="cuda", dtype=torch.bfloat16, generator=gen) * .05).requires_grad_()
            w = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=gen) * .05).requires_grad_()
            grad = torch.randn(y_shape, device="cuda", dtype=x.dtype, generator=gen) * .05
            forward = DirectExchange(cuda, rank, 2, m, k, n, x.dtype)
            backward = DirectExchange(cuda, rank, 2, m, n, k, x.dtype) if op == "rs" else None

            def dx():
                y = column_parallel(x, w, forward) if op == "ag" else row_parallel(x, w, forward, backward)
                gx, gw = torch.autograd.grad(y, (x, w), grad)
                return y, gx, gw

            def stock():
                if op == "ag":
                    full = torch.empty((m, k), dtype=x.dtype, device=x.device)
                    dist.all_gather_into_tensor(full, x.detach())
                    y = full.matmul(w.detach().t())
                    gfull = torch.empty_like(full)
                    work = dist.all_gather_into_tensor(gfull, x.detach(), async_op=True)
                    gi = grad.matmul(w.detach())
                    work.wait()
                    gx = torch.empty_like(x)
                    work = dist.reduce_scatter_tensor(gx, gi, async_op=True)
                    gw = grad.t().matmul(gfull)
                    work.wait()
                else:
                    partial = x.detach().matmul(w.detach().t())
                    y = torch.empty(y_shape, dtype=x.dtype, device=x.device)
                    dist.reduce_scatter_tensor(y, partial)
                    gfull = torch.empty((m, n), dtype=x.dtype, device=x.device)
                    dist.all_gather_into_tensor(gfull, grad)
                    gx = gfull.matmul(w.detach())
                    gw = gfull.t().matmul(x.detach())
                return y, gx, gw

            calibration_start=time.monotonic()
            for _ in range(checks):
                with torch.no_grad():
                    x.normal_(generator=gen).mul_(.05)
                    grad.normal_(generator=gen).mul_(.05)
                a, b = stock(), dx()
                for result, reference in zip(b, a):
                    torch.testing.assert_close(result, reference, rtol=.02, atol=.02)
            funcs = [("stock", stock), ("dx", dx)]
            random.Random(f"{seed}:{model}:{site}:{m}").shuffle(funcs)
            measured = {name: timed(fn, repeats, batch) for name, fn in funcs}
            item = {"model": model, "site": site, "op": op, "M": m, "K": k, "N": n,
                    "measurements": measured, "calibration_wall_ms": (time.monotonic()-calibration_start)*1000}
            output["results"].append(item)
            if rank == 0:
                print(json.dumps({**{key: item[key] for key in ("model", "site", "M", "K", "N")},
                                  "stock_ms": measured["stock"]["median_ms"],
                                  "dx_ms": measured["dx"]["median_ms"]}), flush=True)
            if backward:
                backward.close()
            forward.close()
            del x, w, grad, a, b, forward, backward
            torch.cuda.empty_cache()
    if rank == 0:
        output['metadata']['collector_wall_ms']=(time.monotonic()-total_start)*1000
        path = Path(os.environ["TRAIN_CELL_OUT"])
        with path.open("x") as stream:
            json.dump(output, stream, indent=2)
            stream.write("\n")
    finalize()


if __name__ == "__main__":
    main()
