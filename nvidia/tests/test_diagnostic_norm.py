import datetime
import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    torch.manual_seed(29)
    for dtype in (torch.float32, torch.bfloat16):
        full = torch.randn(16, 32, device="cuda", dtype=dtype)
        grad = torch.randn_like(full)
        weight = torch.randn(32, device="cuda", dtype=dtype)
        bias = torch.randn(32, device="cuda", dtype=dtype)
        local_input = full[rank * 8:(rank + 1) * 8].clone().requires_grad_()
        local_weight = weight.clone().requires_grad_()
        local_bias = bias.clone().requires_grad_()
        local_output = torch.nn.functional.layer_norm(local_input, (32,), local_weight, local_bias)
        local_output.backward(grad[rank * 8:(rank + 1) * 8])
        dist.all_reduce(local_weight.grad)
        dist.all_reduce(local_bias.grad)
        ref_input = full.clone().requires_grad_()
        ref_weight = weight.clone().requires_grad_()
        ref_bias = bias.clone().requires_grad_()
        reference = torch.nn.functional.layer_norm(ref_input, (32,), ref_weight, ref_bias)
        reference.backward(grad)
        tol = 0.02 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(local_output, reference[rank * 8:(rank + 1) * 8], rtol=tol, atol=tol)
        torch.testing.assert_close(local_input.grad, ref_input.grad[rank * 8:(rank + 1) * 8], rtol=tol, atol=tol)
        torch.testing.assert_close(local_weight.grad, ref_weight.grad, rtol=tol, atol=tol)
        torch.testing.assert_close(local_bias.grad, ref_bias.grad, rtol=tol, atol=tol)
        print(f"rank={rank} dtype={dtype} sharded LayerNorm and summed gradients PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
