import datetime
import json
import os
import time

import torch
import torch.distributed as dist

from stowaway_nvidia.runtime import IpcBuffer, SymmetricBuffers, finalize, initialize


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=600))
    cuda = initialize()
    cpu_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=600))
    size = 1 << 20
    operations = int(os.environ.get("F50_OPS", "40"))
    copies = 7
    repeats = int(os.environ.get("F50_REPEATS", "3"))
    delays = [int(delay) for delay in os.environ.get("F50_DELAYS_MS", "200,400,800").split(",")]
    if operations < 1 or repeats < 1 or any(delay < 1 for delay in delays):
        raise ValueError("operations, repeats and delays must be positive")
    allocation = operations * copies * size + operations * copies * 4
    ipc = IpcBuffer(cuda, allocation, cpu_group)
    heap = SymmetricBuffers(rank, 2, [((allocation,), torch.uint8)])
    local_torch = torch.empty(allocation, dtype=torch.uint8, device="cuda")
    source = torch.ones(size, dtype=torch.uint8, device="cuda")
    tickets = torch.arange(operations * copies, dtype=torch.int32, device="cuda")
    flags = torch.zeros(operations, dtype=torch.int32, device="cuda")
    compute, lane = torch.cuda.Stream(), torch.cuda.Stream()
    events = [torch.cuda.Event(enable_timing=False) for _ in range(operations)]
    compute.wait_stream(torch.cuda.current_stream())
    lane.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    results = []
    if rank == 0:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(10_000_000)
        end.record()
        end.synchronize()
        calibration_ms = start.elapsed_time(end)
        cycles_per_ms = 10_000_000 / calibration_ms
        destinations = {"ipc_peer": ipc.peer.value, "nvshmem_peer": heap.buffers[0][1],
                        "cuda_local": ipc.local.value, "torch_local": local_torch.data_ptr()}
        flag_views = [flags[operation] for operation in range(operations)]
        for repeat in range(-1, repeats):
            for sleep_ms in delays:
                for gating in ("event", "value"):
                    for name, destination in destinations.items():
                        torch.cuda.synchronize()
                        flags.zero_()
                        torch.cuda.synchronize()
                        sleep_start = torch.cuda.Event(enable_timing=True)
                        sleep_end = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(compute):
                            sleep_start.record(compute)
                            torch.cuda._sleep(int(cycles_per_ms * sleep_ms))
                            sleep_end.record(compute)
                        elapsed = {"fork_ms": 0.0, "data_ms": 0.0, "ticket_ms": 0.0}
                        begin = time.perf_counter()
                        for operation in range(operations):
                            with torch.cuda.stream(compute):
                                flag_views[operation].add_(1)
                            before = time.perf_counter()
                            if gating == "event":
                                events[operation].record(compute)
                                lane.wait_event(events[operation])
                            else:
                                cuda.write(flags.data_ptr() + operation * 4, operation + 1, compute)
                                cuda.wait(flags.data_ptr() + operation * 4, operation + 1, lane)
                            elapsed["fork_ms"] += (time.perf_counter() - before) * 1000
                            for index in range(copies):
                                offset = operation * copies + index
                                before = time.perf_counter()
                                cuda.copy(destination + offset * size, source.data_ptr(), size, lane)
                                elapsed["data_ms"] += (time.perf_counter() - before) * 1000
                                before = time.perf_counter()
                                cuda.copy(destination + operations * copies * size + 4 * offset,
                                          tickets.data_ptr() + 4 * offset, 4, lane)
                                elapsed["ticket_ms"] += (time.perf_counter() - before) * 1000
                        host_ms = (time.perf_counter() - begin) * 1000
                        torch.cuda.synchronize()
                        result = {"repeat": repeat, "sleep_ms": sleep_ms, "gate": gating,
                                  "actual_sleep_ms": sleep_start.elapsed_time(sleep_end),
                                  "destination": name, "host_ms": host_ms, **elapsed}
                        if repeat >= 0:
                            results.append(result)
                            print(json.dumps(result), flush=True)
        destination = os.environ.get("F50_JSON")
        if destination:
            with open(destination, "w", encoding="utf-8") as output:
                json.dump({"classification": "exploratory; requires review against preregistered criteria",
                           "operations": operations, "copies": copies, "copy_bytes": size,
                           "repeats": repeats, "delays_ms": delays,
                           "calibration_ms": calibration_ms,
                           "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset"),
                           "gpu": torch.cuda.get_device_name(), "samples": results}, output, indent=2)
    dist.barrier(group=cpu_group)
    ipc.close()
    heap.close()
    dist.destroy_process_group(cpu_group)
    finalize()


if __name__ == "__main__":
    main()
