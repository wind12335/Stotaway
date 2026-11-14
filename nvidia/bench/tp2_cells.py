"""Exploratory TP2 stock/DX vehicle-site measurements, not a fitted model."""
import datetime
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import triton

from stowaway_nvidia.dx import DirectExchange, ReduceScatter
from stowaway_nvidia.runtime import finalize, initialize


MODELS = {"27b_vehicle": (5120, 17408, 40, 8), "70b_vehicle": (8192, 28672, 64, 8)}


def configured_models():
    """Optional shape inventory; keeps the original two-vehicle default."""
    raw = os.environ.get("CELL_MODELS_JSON")
    if not raw:
        return MODELS
    entries = json.loads(Path(raw).read_text())
    if not isinstance(entries, dict) or not entries:
        raise ValueError("CELL_MODELS_JSON requires a nonempty object")
    for name, values in entries.items():
        if (not isinstance(name, str) or not name or not isinstance(values, list) or len(values) != 4
                or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values)
                or values[0] % values[2] or (values[2] + 2 * values[3]) * (values[0] // values[2]) % 2
                or values[0] % 2):
            raise ValueError(f"invalid model dimensions: {name}")
    return entries


def shapes(hidden, ffn, heads, kv, rows):
    dim = hidden // heads
    return (("qkv", "ag", rows, hidden, (heads + 2 * kv) * dim // 2),
            ("o_proj", "rs", rows, hidden // 2, hidden),
            ("fc1", "ag", rows, hidden, ffn),
            ("fc2", "rs", rows, ffn // 2, hidden))


def configured_sites(rows):
    """Yield model/site/direction/M/K/N from defaults or an explicit manifest."""
    manifest_path = os.environ.get('CELL_SHAPE_MANIFEST')
    if not manifest_path:
        model_filter=os.environ.get('CELL_MODEL')
        site_filter=os.environ.get('CELL_SITE')
        direction_filter=os.environ.get('CELL_DIRECTION')
        if site_filter and site_filter not in ('qkv','o_proj','fc1','fc2'):
            raise ValueError('unknown CELL_SITE')
        if direction_filter and direction_filter not in ('ag','rs'):
            raise ValueError('unknown CELL_DIRECTION')
        for model, dimensions in configured_models().items():
            if model_filter and model!=model_filter:continue
            for site, op, m, k, n in shapes(*dimensions, rows):
                if site_filter and site!=site_filter:continue
                if direction_filter and op!=direction_filter:continue
                yield model, site, op, m, k, n
        return
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['tp'] != 2 or manifest['M'] != rows or manifest['dtype'] != 'bfloat16':
        raise ValueError('shape manifest must match TP2, BF16 and M')
    for entry in manifest['sites']:
        m, k, n = (entry[x] for x in ('M', 'K', 'N'))
        if m != rows or min(m, k, n) <= 0 or m % 2:
            raise ValueError('invalid manifest shape')
        op = 'rs' if entry['op'] == 'rs' else 'ag' if entry['op'] in ('ag_up', 'ag_gu') else None
        if op is None:
            raise ValueError('unsupported manifest operation')
        yield manifest['model'], entry['site'], op, m, k, n


def measure(function, repeats, batch):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch):
            function()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([start.elapsed_time(end) / batch], device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
    cuda = initialize()
    rows = int(os.environ.get("CELL_M", "4096"))
    repeats = int(os.environ.get("CELL_REPEATS", "8"))
    batch = int(os.environ.get("CELL_BATCH", "16"))
    checks = int(os.environ.get("CELL_CHECKS", "3"))
    schedule_seed = int(os.environ.get("CELL_SCHEDULE_SEED", "0"))
    scratch = os.environ.get('CELL_RS_SCRATCH', '0') == '1'
    if rows <= 0 or rows % 2 or min(repeats, batch, checks) < 1:
        raise ValueError("M must be even and all counts positive")
    generator = torch.Generator(device="cuda").manual_seed(917 + rank)
    report = {"metadata": {"classification": "exploratory TP2 prototype timings, not fitted or publication results",
                           "rows": rows, "tp": 2, "dtype": "bfloat16", "checks": checks,
                           "p2p_bidirectional": torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0),
                           "repeats": repeats, "batch": batch, "gpu": torch.cuda.get_device_name(),
                           "warmup": 3, "rs_scratch": scratch,
                           "rs_actual_path": 'direct_exchange_rs_scratch' if scratch else 'direct_exchange',
                           "torch": torch.__version__, "triton": triton.__version__,
                           "python": sys.executable, "cuda_build": torch.version.cuda,
                           "schedule_seed": schedule_seed,
                           "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset")},
              "results": []}
    for model, site, op, m, k, n in configured_sites(rows):
            exchange = DirectExchange(cuda, rank, 2, m, k, n, torch.bfloat16,
                                      reuse_rs_scratch=scratch and op=='rs')
            reducer = ReduceScatter(cuda, rank, 2, m * n, torch.bfloat16) if op == "rs" else None
            weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.05
            full = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
            shard = torch.empty((m // 2, k), device="cuda", dtype=torch.bfloat16)
            partial = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            output = torch.empty((m // 2, n), device="cuda", dtype=torch.bfloat16)
            for turn in range(checks):
                full.normal_(generator=generator).mul_(0.05)
                shard.normal_(generator=generator).mul_(0.05)
                if op == "ag":
                    dist.all_gather_into_tensor(full, shard)
                    reference = full @ weight.t()
                    actual, gathered = exchange.all_gather_gemm(shard, weight, need_full=True)
                    torch.testing.assert_close(gathered, full, rtol=0, atol=0)
                else:
                    torch.matmul(full, weight.t(), out=partial)
                    dist.reduce_scatter_tensor(output, partial)
                    reference = output
                    actual = exchange.gemm_reduce_scatter(full, weight)
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
                if reducer is not None:
                    dx_reduced, _ = reducer(partial)
                    torch.testing.assert_close(dx_reduced.view_as(output), output, rtol=0.02, atol=0.02)
            def gemm():
                torch.matmul(full, weight.t(), out=partial)
            if op == "ag":
                def comm():
                    dist.all_gather_into_tensor(full, shard)
                def stock():
                    comm()
                    gemm()
                def dx():
                    return exchange.all_gather_gemm(shard, weight)
            else:
                def comm():
                    dist.reduce_scatter_tensor(output, partial)
                def stock():
                    gemm()
                    comm()
                def dx():
                    return exchange.gemm_reduce_scatter(full, weight)
            functions = [("gemm", gemm), ("comm", comm), ("stock", stock), ("dx", dx)]
            if reducer is not None:
                def dx_comm():
                    return reducer(partial, output)
                functions.append(("dx_comm", dx_comm))
            if schedule_seed:
                random.Random(f"{schedule_seed}:{model}:{site}:{m}").shuffle(functions)
            measurements = {name: measure(fn, repeats, batch) for name, fn in functions}
            record = {"model": model, "site": site, "op": op, "M": m, "K": k, "N": n,
                      "payload_MB": (m * k * 2 if op == "ag" else m * n * 2) / 1e6,
                      "checks": checks, "measurements": measurements}
            report["results"].append(record)
            if rank == 0:
                print(json.dumps({"model": model, "site": site, "M": m, "K": k, "N": n,
                                  **{name + "_ms": value["median_ms"]
                                     for name, value in measurements.items()}}), flush=True)
            if reducer is not None:
                reducer.close()
            exchange.close()
            del exchange, weight, full, shard, partial, output, reference, actual
            if op == "ag":
                del gathered
            torch.cuda.empty_cache()
    if rank == 0:
        destination = Path(os.environ["CELL_OUT"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    finalize()


if __name__ == "__main__":
    main()
