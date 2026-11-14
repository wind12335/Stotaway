"""Distributed correctness and output ownership for direct logical row layout."""
import datetime
import os

import torch
import torch.distributed as dist
from stowaway_nvidia.dx import DirectExchange
from stowaway_nvidia.runtime import initialize, finalize


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    cuda = initialize()
    gen = torch.Generator(device='cuda').manual_seed(9200 + rank)
    for dtype in (torch.float16, torch.bfloat16):
        for rows, inner, columns in ((64, 128, 96), (256, 256, 128)):
            for scratch in (False, True):
                ctx = DirectExchange(cuda, rank, 2, rows, inner, columns, dtype, reuse_rs_scratch=scratch)
                w = torch.randn((columns, inner), dtype=dtype, device='cuda', generator=gen) * .05
                retained = []
                for turn in range(8):
                    x = torch.randn((rows, inner), dtype=dtype, device='cuda', generator=gen) * .05
                    local = torch.empty((rows // 2, columns), dtype=dtype, device='cuda')
                    dist.reduce_scatter_tensor(local, x @ w.T)
                    parts = [torch.empty_like(local) for _ in range(2)]
                    dist.all_gather(parts, local)
                    expected = torch.cat(parts)[rank::2].contiguous()
                    result = ctx.gemm_reduce_scatter(x, w, output_layout='interleaved')
                    torch.testing.assert_close(result, expected, rtol=.02, atol=.02)
                    retained.append((result, result.clone()))
                for result, expected in retained:
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                try:
                    ctx.gemm_reduce_scatter(x, w, output_layout='unknown')
                    raise AssertionError('unknown layout accepted')
                except ValueError:
                    pass
                ctx.close()
    print(f'rank={rank} dtype=fp16,bf16 shapes=2 scratch=on,off iterations=8 PASS', flush=True)
    finalize()


if __name__ == '__main__':
    main()
