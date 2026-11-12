"""End-to-end top-k local routing, expert MLP and gradients vs PyTorch."""
import torch
import torch.nn.functional as F

from stowaway_nvidia.moe_local import local_moe


def main():
    torch.manual_seed(504)
    for dtype in (torch.float16, torch.bfloat16):
        for turn in range(8):
            t, h, ff, experts, topk = 37, 32, 48, 4, 2
            x = torch.randn(t, h, device="cuda", dtype=dtype, requires_grad=True)
            w1 = (torch.randn(experts, h, 2 * ff, device="cuda", dtype=dtype) * 0.1).requires_grad_()
            w2 = (torch.randn(experts, ff, h, device="cuda", dtype=dtype) * 0.1).requires_grad_()
            ids = torch.zeros((t, topk), device="cuda", dtype=torch.int64) if turn == 0 else \
                  torch.randint(experts, (t, topk), device="cuda")
            weights = torch.rand(t, topk, device="cuda", dtype=torch.float32, requires_grad=True)
            gout = torch.randn_like(x)
            if turn == 1:
                # Catch synchronizations detectable by PyTorch after compilation;
                # this mode does not detect every possible host synchronization.
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    local_moe(x, ids, weights, w1, w2)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            out = local_moe(x, ids, weights, w1, w2)
            actual = torch.autograd.grad(out, (x, weights, w1, w2), gout)
            rx, rw, ra, rb = [v.detach().clone().requires_grad_() for v in (x, weights, w1, w2)]
            ids_cpu = ids.cpu().reshape(-1).tolist()
            groups = [[i for i, expert in enumerate(ids_cpu) if expert == e] for e in range(experts)]
            pieces, permutation = [], []
            for expert, indices in enumerate(groups):
                permutation.extend(indices)
                rows = torch.tensor([i // topk for i in indices], device="cuda", dtype=torch.int64)
                gate, value = (rx.index_select(0, rows) @ ra[expert]).chunk(2, dim=-1)
                pieces.append((F.silu(gate) * value) @ rb[expert])
            inverse = torch.tensor(sorted(range(len(permutation)), key=permutation.__getitem__), device="cuda")
            restored = torch.cat(pieces).index_select(0, inverse).view(t, topk, h)
            ref = (restored.float() * rw.unsqueeze(-1)).sum(1).to(dtype)
            expected = torch.autograd.grad(ref, (rx, rw, ra, rb), gout)
            for got, want in zip((out, *actual), (ref, *expected)):
                torch.testing.assert_close(got, want, rtol=0.04, atol=0.02)
        print(f"dtype={dtype} local top-k SwiGLU 8 routes forward and gradients PASS", flush=True)


if __name__ == "__main__":
    main()
