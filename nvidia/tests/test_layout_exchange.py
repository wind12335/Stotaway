"""Distributed row-layout conversion contract, including repeated inputs."""
import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.layout_exchange import convert_rs_rows


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
    if dist.get_world_size() != 2:
        raise ValueError('TP2 required')
    for dtype in (torch.float16, torch.bfloat16):
        for rows, columns in ((2, 7), (16, 35), (128, 256)):
            for turn in range(4):
                x = (torch.arange(rows * columns, device='cuda', dtype=torch.float32)
                     .reshape(rows, columns).remainder(31) + rank * 40 + turn).to(dtype)
                pieces = [torch.empty_like(x) for _ in range(2)]
                dist.all_gather(pieces, x)
                expected = torch.cat(pieces, dim=0)[rank::2].contiguous()
                converted = convert_rs_rows(x, 'contiguous_to_interleaved')
                torch.testing.assert_close(converted, expected, atol=0, rtol=0)
                restored = convert_rs_rows(converted, 'interleaved_to_contiguous')
                torch.testing.assert_close(restored, x, atol=0, rtol=0)
                if converted.data_ptr() == x.data_ptr() or restored.data_ptr() == converted.data_ptr():
                    raise AssertionError('conversion outputs must own independent storage')
    try:
        convert_rs_rows(torch.empty((3, 2), device='cuda', dtype=torch.float16), 'contiguous_to_interleaved')
        raise AssertionError('odd rank rows accepted')
    except ValueError:
        pass
    print(f'rank={rank} dtype=fp16,bf16 shapes=3 repetitions=4 PASS', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
