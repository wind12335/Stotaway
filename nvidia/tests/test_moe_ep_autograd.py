"""EP2 forward and distributed autograd vs a global grouped reference."""
import datetime
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

from stowaway_nvidia.moe_ep import ep_moe
from stowaway_nvidia.moe_exchange import MoEExchange
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    t = int(os.environ.get("MOE_T", "16"))
    h, ff = int(os.environ.get("MOE_H", "32")), int(os.environ.get("MOE_FF", "48"))
    topk, experts = 2, int(os.environ.get("MOE_EXPERTS", "4"))
    if experts <= 0 or experts % 2:
        raise ValueError("MOE_EXPERTS must be positive and even")
    torch.manual_seed(333)
    scale = float(os.environ.get("MOE_WEIGHT_SCALE", ".1"))
    weights1 = torch.randn(experts, h, ff * 2, device="cuda", dtype=torch.bfloat16) * scale
    weights2 = torch.randn(experts, ff, h, device="cuda", dtype=torch.bfloat16) * scale
    epr = experts // 2
    w1 = weights1[rank * epr:(rank + 1) * epr].clone().requires_grad_()
    w2 = weights2[rank * epr:(rank + 1) * epr].clone().requires_grad_()
    gen = torch.Generator(device="cuda").manual_seed(77 + rank)
    prescale = os.environ.get("TEST_MOE_PRE_FC2_SCALE") == "1"
    ex = MoEExchange(cuda, rank, t, topk, h, experts)
    try:
        for turn in range(int(os.environ.get("TEST_ROUNDS", "8"))):
            x = torch.randn(t, h, generator=gen, device="cuda", dtype=torch.bfloat16).requires_grad_()
            ids = torch.randint(experts, (t, topk), generator=gen, device="cuda", dtype=torch.int32)
            if turn == 0:
                ids.fill_(0)
            route = torch.rand(t, topk, generator=gen, device="cuda").requires_grad_()
            gout = torch.randn(t, h, generator=gen, device="cuda", dtype=torch.bfloat16)
            y = ep_moe(x, ids, route, w1, w2, ex, scale_before_fc2=prescale)
            if turn == 0:
                try:
                    ep_moe(x, ids, route, w1, w2, ex)
                except RuntimeError as exc:
                    assert "finish backward" in str(exc)
                else:
                    raise AssertionError("reuse before backward must be rejected")
            grads = torch.autograd.grad(y, (x, route, w1, w2), gout, retain_graph=turn == 0)
            if turn == 0:
                try:
                    torch.autograd.grad(y, (x, route, w1, w2), gout)
                except RuntimeError as exc:
                    assert "repeated backward" in str(exc)
                else:
                    raise AssertionError("repeated backward must be rejected")
            gathered = []
            for value in (x.detach(), ids, route.detach(), gout):
                peers = [torch.empty_like(value) for _ in range(2)]
                dist.all_gather(peers, value)
                gathered.append(torch.cat(peers))
            all_x, all_ids, all_route, all_gout = gathered
            rx = all_x.requires_grad_()
            rw = all_route.requires_grad_()
            ra = weights1.detach().clone().requires_grad_()
            rb = weights2.detach().clone().requires_grad_()
            flat_ids = all_ids.reshape(-1).cpu().tolist()
            groups = [[i for i, expert in enumerate(flat_ids) if expert == e] for e in range(experts)]
            perm, pieces = [], []
            for e, indices in enumerate(groups):
                perm.extend(indices)
                indices_t = torch.tensor([i // topk for i in indices], device="cuda", dtype=torch.int64)
                a, b = (rx.index_select(0, indices_t) @ ra[e]).chunk(2, -1)
                activation = F.silu(a) * b
                if prescale:
                    flat_route = rw.reshape(-1)
                    activation = (activation * flat_route[indices].unsqueeze(-1)).to(activation.dtype)
                pieces.append(activation @ rb[e])
            inverse = torch.tensor(sorted(range(len(perm)), key=perm.__getitem__), device="cuda")
            per_tag = torch.cat(pieces).index_select(0, inverse).view(2 * t, topk, h)
            reference = (per_tag.float() if prescale else per_tag.float() * rw.unsqueeze(-1)).sum(1).to(torch.bfloat16)
            expected = torch.autograd.grad(reference, (rx, rw, ra, rb), all_gout)
            checks = ((y, reference[rank*t:(rank+1)*t]),
                      (grads[0], expected[0][rank*t:(rank+1)*t]),
                      (grads[1], expected[1][rank*t:(rank+1)*t]),
                      (grads[2], expected[2][rank*epr:(rank+1)*epr]),
                      (grads[3], expected[3][rank*epr:(rank+1)*epr]))
            failures = []
            for name, (actual, ref) in zip(("output", "input_grad", "router_grad", "w1_grad", "w2_grad"), checks):
                try:
                    torch.testing.assert_close(actual, ref, rtol=.04, atol=.02)
                except AssertionError as exc:
                    failures.append(f"rank={rank} turn={turn} tensor={name}\n{exc}")
            failed = torch.tensor([bool(failures)], dtype=torch.int32, device="cuda")
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item():
                raise AssertionError("\n".join(failures) if failures else "peer rank failed correctness")
        print(f"rank={rank} EP2 autograd PASS", flush=True)
    finally:
        ex.close()
        finalize()


if __name__ == "__main__":
    main()
