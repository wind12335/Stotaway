"""Two-rank MoE dispatch, expert compute and combine against a dense reference."""
import datetime
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from stowaway_nvidia.moe_exchange import MoEExchange
from stowaway_nvidia.moe_grouped import expert_linear
from stowaway_nvidia.runtime import finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    cuda = initialize()
    t = int(os.environ.get("MOE_T", "32"))
    topk, h, ff, experts = 2, int(os.environ.get("MOE_H", "32")), int(os.environ.get("MOE_FF", "48")), 4
    torch.manual_seed(18)
    w1 = torch.randn(experts, h, ff * 2, device="cuda", dtype=torch.bfloat16) * .1
    w2 = torch.randn(experts, ff, h, device="cuda", dtype=torch.bfloat16) * .1
    context = MoEExchange(cuda, rank, t, topk, h, experts)
    gen = torch.Generator(device="cuda").manual_seed(rank + 100)
    rounds = int(os.environ.get("TEST_ROUNDS", "64"))
    worst = 0.0
    samples = []
    try:
        other_stream = torch.cuda.Stream()
        with torch.cuda.stream(other_stream):
            try:
                context._check_stream()
            except RuntimeError as exc:
                assert "single caller stream" in str(exc)
            else:
                raise AssertionError("context must reject a different caller stream")
        for turn in range(rounds):
            x = torch.randn(t, h, generator=gen, device="cuda", dtype=torch.bfloat16)
            ids = torch.randint(experts, (t, topk), generator=gen, device="cuda", dtype=torch.int32)
            if turn % 4 == 0:
                ids.fill_((turn // 4) % experts)
            weights = torch.rand(t, topk, generator=gen, device="cuda")
            if turn % 13 == rank:
                torch.cuda._sleep(500_000)
            checking_sync = turn == 1 and os.environ.get("TEST_SYNC_DEBUG") == "1"
            if checking_sync:
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
            try:
                rows, offs, total, order = context.dispatch(x, ids)
                up = expert_linear(rows, w1[rank * 2:rank * 2 + 2], offs)
                gate, value = up.chunk(2, dim=-1)
                out = expert_linear((F.silu(gate) * value).contiguous(), w2[rank * 2:rank * 2 + 2], offs)
                result = context.combine(out, total, order, weights)
            finally:
                if checking_sync:
                    torch.cuda.set_sync_debug_mode("default")
            samples.append((x, ids, weights, result))
        # Do not inspect device results inside the burst: this exercises repeated
        # ticket/ACK reuse without a host synchronization between exchanges.
        for x, ids, weights, result in samples:
            reference_rows = torch.empty(t * topk, h, dtype=x.dtype, device=x.device)
            for expert in range(experts):
                selected = (ids.reshape(-1) == expert).nonzero().view(-1)
                inp = x.index_select(0, torch.div(selected, topk, rounding_mode="floor"))
                gate, value = (inp @ w1[expert]).chunk(2, dim=-1)
                reference_rows[selected] = (F.silu(gate) * value) @ w2[expert]
            expected = (reference_rows.view(t, topk, h).float() * weights.unsqueeze(-1)).sum(1).to(x.dtype)
            worst = max(worst, (result.float() - expected.float()).abs().max().item())
            torch.testing.assert_close(result, expected, rtol=.04, atol=.02)
        print(f"rank={rank} EP2 forward rounds={rounds} max_abs={worst:.6g} PASS", flush=True)
    finally:
        context.close()
        finalize()


if __name__ == "__main__":
    main()
