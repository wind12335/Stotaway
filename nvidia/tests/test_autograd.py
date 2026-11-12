import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.autograd import column_parallel, row_parallel
from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import finalize, initialize


def run(dtype, rank, cuda):
    rows, inner, columns = 128, 96, 64
    context = DirectExchange(cuda, rank, 2, rows, inner, columns, dtype)
    backward_context = DirectExchange(cuda, rank, 2, rows, columns, inner, dtype)
    generator = torch.Generator(device="cuda").manual_seed(913 + rank)
    for turn in range(8):
        weight = (torch.randn(columns, inner, dtype=dtype, device="cuda", generator=generator) * 0.1).requires_grad_()
        activation = (torch.randn(rows, inner, dtype=dtype, device="cuda", generator=generator) * 0.1).requires_grad_()
        grad = torch.randn(rows // 2, columns, dtype=dtype, device="cuda", generator=generator)
        output = row_parallel(activation, weight, context, backward_context)
        output.backward(grad)
        gathered = torch.empty(rows, columns, dtype=dtype, device="cuda")
        dist.all_gather_into_tensor(gathered, grad)
        full = activation.detach().matmul(weight.detach().t())
        expected = torch.empty_like(output)
        dist.reduce_scatter_tensor(expected, full)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(activation.grad, gathered.matmul(weight.detach()), rtol=0.02, atol=0.02)
        torch.testing.assert_close(weight.grad, gathered.t().matmul(activation.detach()), rtol=0.02, atol=0.02)
        shard = (torch.randn(rows // 2, inner, dtype=dtype, device="cuda", generator=generator) * 0.1).requires_grad_()
        col_weight = (torch.randn(columns, inner, dtype=dtype, device="cuda", generator=generator) * 0.1).requires_grad_()
        col_grad = torch.randn(rows, columns, dtype=dtype, device="cuda", generator=generator)
        col_output = column_parallel(shard, col_weight, context)
        col_output.backward(col_grad)
        gathered_shard = torch.empty(rows, inner, dtype=dtype, device="cuda")
        dist.all_gather_into_tensor(gathered_shard, shard.detach())
        full_input_grad = col_grad.matmul(col_weight.detach())
        expected_input_grad = torch.empty_like(shard)
        dist.reduce_scatter_tensor(expected_input_grad, full_input_grad)
        torch.testing.assert_close(col_output, gathered_shard.matmul(col_weight.detach().t()), rtol=0.02, atol=0.02)
        torch.testing.assert_close(shard.grad, expected_input_grad, rtol=0.02, atol=0.02)
        torch.testing.assert_close(col_weight.grad, col_grad.t().matmul(gathered_shard), rtol=0.02, atol=0.02)
    print(f"rank={rank} dtype={dtype} iterations=8 row/column autograd PASS", flush=True)
    backward_context.close()
    context.close()


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    cuda = initialize()
    for dtype in (torch.float16, torch.bfloat16):
        run(dtype, rank, cuda)
    finalize()


if __name__ == "__main__":
    main()
