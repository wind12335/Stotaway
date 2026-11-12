import datetime
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist
import triton

from stowaway_nvidia.dx import DirectExchange, ReduceScatter
from stowaway_nvidia.runtime import finalize, initialize


def measure(function, iterations, batch):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(iterations):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(batch):
            function()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([begin.elapsed_time(end) / batch], device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    rows = int(os.environ.get("BENCH_M", "1024"))
    inner = int(os.environ.get("BENCH_K", "512"))
    columns = int(os.environ.get("BENCH_N", "512"))
    iterations = int(os.environ.get("BENCH_ITERS", "8"))
    batch = int(os.environ.get("BENCH_BATCH", "4"))
    correctness = int(os.environ.get("BENCH_CORRECTNESS", "8"))
    if min(rows, inner, columns, iterations, batch, correctness) <= 0 or rows % 2:
        raise ValueError("positive dimensions, iteration counts, and even M required")
    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(91 + rank)
    def random(shape):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.05
    exchange = DirectExchange(cuda, rank, 2, rows, inner, columns, dtype)
    grad_exchange = DirectExchange(cuda, rank, 2, rows, columns, inner, dtype)
    reducer = ReduceScatter(cuda, rank, 2, rows * columns, dtype)
    weight = random((columns, inner))
    activation = random((rows, inner))
    shard = random((rows // 2, inner))
    grad = random((rows // 2, columns))
    partial = random((rows * columns,))
    for turn in range(correctness):
        activation = random((rows, inner))
        shard = random((rows // 2, inner))
        grad = random((rows // 2, columns))
        partial = random((rows * columns,))
        dx_rs = exchange.gemm_reduce_scatter(activation, weight)
        dx_ag, dx_input = exchange.all_gather_gemm(shard, weight, need_full=True)
        dx_dgrad, dx_wgrad, dx_full = grad_exchange.all_gather_dgrad(grad, weight, activation)
        dx_reduce, _ = reducer(partial)
        gathered_input = torch.empty((rows, inner), device="cuda", dtype=dtype)
        dist.all_gather_into_tensor(gathered_input, shard)
        ref_ag = gathered_input.matmul(weight.t())
        ref_ag_block = torch.cat([gathered_input[block * (rows // 2):(block + 1) * (rows // 2)].matmul(weight.t()) for block in range(2)])
        ref_rs = torch.empty_like(dx_rs)
        dist.reduce_scatter_tensor(ref_rs, activation.matmul(weight.t()))
        ref_grad = torch.empty((rows, columns), device="cuda", dtype=dtype)
        dist.all_gather_into_tensor(ref_grad, grad)
        ref_reduce = torch.empty_like(dx_reduce)
        dist.reduce_scatter_tensor(ref_reduce, partial)
        torch.testing.assert_close(dx_rs, ref_rs, rtol=0.02, atol=0.02)
        torch.testing.assert_close(dx_ag, ref_ag, rtol=0.02, atol=0.02)
        torch.testing.assert_close(dx_input, gathered_input, rtol=0, atol=0)
        torch.testing.assert_close(dx_ag, ref_ag_block, rtol=0, atol=0)
        torch.testing.assert_close(dx_full, ref_grad, rtol=0, atol=0)
        torch.testing.assert_close(dx_dgrad, ref_grad.matmul(weight), rtol=0.02, atol=0.02)
        torch.testing.assert_close(dx_wgrad, ref_grad.t().matmul(activation), rtol=0.02, atol=0.02)
        torch.testing.assert_close(dx_reduce, ref_reduce, rtol=0.02, atol=0.02)
    print(f"rank={rank} correctness rounds={correctness} PASS", flush=True)
    ag_full = torch.empty((rows, inner), device="cuda", dtype=dtype)
    rs_full = torch.empty((rows, columns), device="cuda", dtype=dtype)
    rs_output = torch.empty((rows // 2, columns), device="cuda", dtype=dtype)
    grad_full = torch.empty((rows, columns), device="cuda", dtype=dtype)
    reduce_output = torch.empty_like(partial[:partial.numel() // 2])
    def stock_ag():
        dist.all_gather_into_tensor(ag_full, shard)
        return ag_full.matmul(weight.t())
    def stock_rs():
        torch.matmul(activation, weight.t(), out=rs_full)
        dist.reduce_scatter_tensor(rs_output, rs_full)
    def stock_dgrad():
        dist.all_gather_into_tensor(grad_full, grad)
        return grad_full.matmul(weight), grad_full.t().matmul(activation)
    def stock_reduce():
        dist.reduce_scatter_tensor(reduce_output, partial)
    functions = {
        "ag_gemm": (stock_ag, lambda: exchange.all_gather_gemm(shard, weight)),
        "gemm_rs": (stock_rs, lambda: exchange.gemm_reduce_scatter(activation, weight)),
        "ag_dgrad_wgrad": (stock_dgrad, lambda: grad_exchange.all_gather_dgrad(grad, weight, activation)),
        "reduce_scatter": (stock_reduce, lambda: reducer(partial)),
    }
    results = {}
    for name, (stock, dx) in functions.items():
        original = measure(stock, iterations, batch)
        overlap = measure(dx, iterations, batch)
        results[name] = {"stock": original, "prototype": overlap}
        if rank == 0:
            print(f"{name}: stock={original['median_ms']:.4f}ms prototype={overlap['median_ms']:.4f}ms", flush=True)
    if rank == 0:
        destination = os.environ.get("BENCH_JSON")
        if destination:
            record = {"classification": "functional smoke; not paper results", "world_size": 2,
                      "M": rows, "K": inner, "N": columns, "dtype": str(dtype),
                      "iterations": iterations, "batch": batch, "correctness_rounds": correctness,
                      "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset"),
                      "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "python": sys.executable, "triton": triton.__version__,
                      "triton_path": triton.__file__, "results": results}
            with open(destination, "w", encoding="utf-8") as output:
                json.dump(record, output, indent=2)
    reducer.close()
    grad_exchange.close()
    exchange.close()
    finalize()


if __name__ == "__main__":
    main()
