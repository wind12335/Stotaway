"""Test device-offset expert linear against per-expert PyTorch autograd."""
import os

import torch

from stowaway_nvidia.moe_grouped import expert_linear


def main():
    torch.cuda.set_device(int(os.environ.get("TEST_DEVICE", "0")))
    torch.manual_seed(111)
    cases = [([0, 0, 0, 0], 0, 33, 47),
             ([0, 1, 1, 17], 23, 33, 47),
             ([65, 65, 65, 65], 65, 64, 96),
             ([3, 37, 41, 129], 137, 96, 80),
             ([0, 1, 4, 9, 17, 31, 64, 1024], 1024, 256, 512)]
    for dtype in (torch.float16, torch.bfloat16):
        for offset_values, t, k, n in cases:
            e = len(offset_values)
            x = torch.randn(t, k, device="cuda", dtype=dtype, requires_grad=True)
            w = torch.randn(e, k, n, device="cuda", dtype=dtype, requires_grad=True)
            offsets = torch.tensor(offset_values, device="cuda",
                                   dtype=torch.int64 if t == 1024 else torch.int32)
            grad = torch.randn(t, n, device="cuda", dtype=dtype)
            result = expert_linear(x, w, offsets)
            dx, dw = torch.autograd.grad(result, (x, w), grad)
            ref_x = x.detach().clone().requires_grad_()
            ref_w = w.detach().clone().requires_grad_()
            pieces = []
            start = 0
            for expert, end in enumerate(offset_values):
                pieces.append(ref_x[start:end] @ ref_w[expert])
                start = end
            pieces.append(ref_x[start:] @ ref_w[0] * 0)
            reference = torch.cat(pieces, dim=0)
            ref_dx, ref_dw = torch.autograd.grad(reference, (ref_x, ref_w), grad)
            for name, got, expected in (("fwd", result, reference), ("dx", dx, ref_dx), ("dw", dw, ref_dw)):
                torch.testing.assert_close(got, expected, rtol=0.02, atol=0.04)
                worst = (got.float() - expected.float()).abs().max().item() if got.numel() else 0
                print(f"dtype={dtype} offs={offset_values} TKN={t},{k},{n} {name} max_abs={worst:.6g} PASS", flush=True)

    # Compose two expert projections and a nonlinear activation, like an MLP.
    # The route changes each iteration while the offsets remain device-resident.
    for turn in range(12):
        e, t, k, h = 4, 96, 48, 80
        sizes = ([0, 0, 0, t] if turn % 3 == 0 else
                 [turn, 32 - turn, 24, 40])
        ends = list(torch.tensor(sizes).cumsum(0).tolist())
        offsets = torch.tensor(ends, dtype=torch.int64, device="cuda")
        x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w1 = torch.randn(e, k, h, device="cuda", dtype=x.dtype, requires_grad=True) * 0.1
        w2 = torch.randn(e, h, k, device="cuda", dtype=x.dtype, requires_grad=True) * 0.1
        grad = torch.randn_like(x)
        out = expert_linear(torch.nn.functional.silu(expert_linear(x, w1, offsets)), w2, offsets)
        actual_grads = torch.autograd.grad(out, (x, w1, w2), grad)
        rx = x.detach().clone().requires_grad_()
        rw1 = w1.detach().clone().requires_grad_()
        rw2 = w2.detach().clone().requires_grad_()
        start = 0
        parts = []
        for expert, end in enumerate(ends):
            parts.append(torch.nn.functional.silu(rx[start:end] @ rw1[expert]) @ rw2[expert])
            start = end
        ref = torch.cat(parts)
        ref_grads = torch.autograd.grad(ref, (rx, rw1, rw2), grad)
        for got, expected in zip((out, *actual_grads), (ref, *ref_grads)):
            torch.testing.assert_close(got, expected, rtol=0.02, atol=0.015)
    print("BF16 two-projection expert MLP dynamic routing 12 iterations PASS", flush=True)


if __name__ == "__main__":
    main()
