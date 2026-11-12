"""Check fixed-size routing histogram, including a host synchronization guard."""
import torch

from stowaway_nvidia.moe_routing import expert_offsets


def main():
    torch.cuda.set_device(0)
    for experts, length in ((1, 0), (4, 257), (17, 10000), (128, 8192)):
        for dtype in (torch.int32, torch.int64):
            keys = torch.randint(-1, experts + 1, (length,), device="cuda", dtype=dtype)
            # Warm up compilation before testing host synchronization.
            expert_offsets(keys, experts)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                offsets, total = expert_offsets(keys, experts)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            valid = keys[(keys >= 0) & (keys < experts)]
            expected = torch.bincount(valid, minlength=experts).cumsum(0).to(torch.int32)
            torch.testing.assert_close(offsets, expected, rtol=0, atol=0)
            torch.testing.assert_close(total, expected[-1], rtol=0, atol=0)
            print(f"experts={experts} length={length} dtype={dtype} PASS", flush=True)


if __name__ == "__main__":
    main()
