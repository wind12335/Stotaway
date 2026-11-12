"""A rank-divergent allocation must fail before touching the NVSHMEM heap."""
import datetime
import os

import torch
import torch.distributed as dist

from stowaway_nvidia.runtime import SymmetricBuffers, finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    initialize()
    try:
        SymmetricBuffers(rank, 2, [((8 + rank,), torch.int32)])
    except ValueError as error:
        if "layouts differ" not in str(error):
            raise
    else:
        raise AssertionError("divergent layouts were accepted")
    buffers = SymmetricBuffers(rank, 2, [((8,), torch.int32), ((2,), torch.bfloat16)])
    buffers.close()
    print(f"rank={rank} divergent layout rejected; matching allocation PASS", flush=True)
    finalize()


if __name__ == "__main__":
    main()
