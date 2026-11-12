"""Exercise the medium EP expert GEMM shapes against cuBLAS on one GPU."""
import os
import torch
from stowaway_nvidia.moe_grouped import expert_linear

torch.cuda.set_device(0)
torch.manual_seed(912)
dtype = torch.bfloat16
e, t, k, n = 4, 1024, 1024, 1024
sizes = [243, 261, 250, 270]
offsets = torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(0).int()
x = (torch.randn(t, k, device="cuda", dtype=dtype) * .1).requires_grad_()
w = (torch.randn(e, k, n, device="cuda", dtype=dtype) * .1).requires_grad_()
dy = torch.randn(t, n, device="cuda", dtype=dtype)
y = expert_linear(x, w, offsets)
gx, gw = torch.autograd.grad(y, (x, w), dy)
rx = x.detach().clone().requires_grad_()
rw = w.detach().clone().requires_grad_()
ys, start = [], 0
for i, count in enumerate(sizes):
    ys.append(rx[start:start + count] @ rw[i])
    start += count
ref = torch.cat(ys)
rgx, rgw = torch.autograd.grad(ref, (rx, rw), dy)
for name, got, expected in (("y", y, ref), ("dx", gx, rgx), ("dw", gw, rgw)):
    delta = (got.float() - expected.float()).abs()
    outside = delta > .02 + .04 * expected.float().abs()
    print(f"{name} max_abs={delta.max().item()} outside={outside.sum().item()}/{delta.numel()} "
          f"relative_l2={(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())).item()}", flush=True)

if os.environ.get("MOE_MLP_DIAG") == "1":
    weights1 = (torch.randn(e, k, n * 2, device="cuda", dtype=dtype) * .1).requires_grad_()
    weights2 = (torch.randn(e, n, k, device="cuda", dtype=dtype) * .1).requires_grad_()
    route = torch.rand(t, 2, device="cuda")
    inp = (torch.randn(t, k, device="cuda", dtype=dtype)).requires_grad_()
    grad = torch.randn(t, k, device="cuda", dtype=dtype)
    a, b = expert_linear(inp, weights1, offsets).chunk(2, -1)
    out = expert_linear((torch.nn.functional.silu(a) * b).contiguous(), weights2, offsets)
    gs = torch.autograd.grad(out, (inp, weights1, weights2), grad)
    ix = inp.detach().clone().requires_grad_()
    iw1 = weights1.detach().clone().requires_grad_()
    iw2 = weights2.detach().clone().requires_grad_()
    parts, start = [], 0
    for i, count in enumerate(sizes):
        aa, bb = (ix[start:start+count] @ iw1[i]).chunk(2, -1)
        parts.append((torch.nn.functional.silu(aa) * bb) @ iw2[i])
        start += count
    refout = torch.cat(parts)
    rgs = torch.autograd.grad(refout, (ix, iw1, iw2), grad)
    for name, got, expected in zip(("mlp_y", "mlp_dx", "mlp_dw1", "mlp_dw2"), (out, *gs), (refout, *rgs)):
        delta = (got.float() - expected.float()).abs()
        outside = delta > .02 + .04 * expected.float().abs()
        print(f"{name} max_abs={delta.max().item()} outside={outside.sum().item()}/{delta.numel()} "
              f"relative_l2={(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())).item()}", flush=True)
