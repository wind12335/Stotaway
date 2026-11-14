"""Two-rank offline vLLM request latency with real weights and token gates.

Each process hosts an external-launcher rank. The measured region includes
LLM.generate scheduling, prefill, sampling, and decode. No tokenization,
loading, initialization, or warmup is included.
"""
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
if int(os.environ["WORLD_SIZE"]) != 2 or rank != int(os.environ["LOCAL_RANK"]):
    raise ValueError("exactly two local ranks required")
torch.cuda.set_device(rank)
arm = os.environ["E2E_ARM"]
if arm not in ("stock", "stock_sp", "dx", "plan"):
    raise ValueError("E2E_ARM must be stock, stock_sp, dx, or plan")
model = Path(os.environ["E2E_MODEL"])
arch = os.environ.get("E2E_ARCH", "llama")
rows = int(os.environ.get("E2E_M", "1024"))
generated = int(os.environ.get("E2E_OUTPUT", "32"))
requests = int(os.environ.get("E2E_REQUESTS", "6"))
warmups = int(os.environ.get("E2E_WARMUPS", "2"))
seed = int(os.environ.get("E2E_SEED", "20261110"))
if rows <= 0 or rows % 2 or min(requests, warmups, generated) < 1:
    raise ValueError("invalid benchmark dimensions")
adapter = None
if arm != "stock":
    from stowaway_nvidia.vllm_sp import LlamaSequenceParallel
    kwargs = ({'forward_plan':os.environ['E2E_FORWARD_PLAN'],
               'plan_workload':{'seq_length':rows,'micro_batch_size':1}} if arm=='plan' else {})
    adapter = LlamaSequenceParallel(None, rank, 'dx' if arm=='plan' else arm, [rows], architecture=arch, **kwargs)
    adapter.install()

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import get_world_group

llm = LLM(model=str(model), tensor_parallel_size=2, pipeline_parallel_size=1,
          distributed_executor_backend="external_launcher", seed=0, dtype="bfloat16",
          max_model_len=rows + generated + 64,
          gpu_memory_utilization=float(os.environ.get("E2E_GPU_UTIL", "0.65")),
          enforce_eager=True, enable_chunked_prefill=True, max_num_batched_tokens=rows,
          disable_log_stats=True, enable_prefix_caching=False)
tokenizer = llm.get_tokenizer()
text = "\n\n".join((model / name).read_text(errors="ignore")
                     for name in ("README.md", "LICENSE", "USE_POLICY.md") if (model / name).exists())
tokens = tokenizer.encode(text, add_special_tokens=False)
if len(tokens) < 2000:
    raise RuntimeError("model documentation must yield at least 2000 natural-text tokens")
reps = rows // len(tokens) + 2
stream = tokens * reps


def prompt(index):
    end = (reps - 1) * len(tokens) + (index + 1) * len(tokens) // (requests + 1)
    prefix = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    ids = prefix + stream[end - rows + len(prefix):end]
    assert len(ids) == rows
    return TokensPrompt(prompt_token_ids=ids)


params = {n: SamplingParams(temperature=0, max_tokens=n, ignore_eos=True) for n in (1, generated)}
for i in range(warmups):
    llm.generate([prompt(i % requests)], params[generated], use_tqdm=False)
    llm.generate([prompt(i % requests)], params[1], use_tqdm=False)
cpu_group = get_world_group().cpu_group
schedule = [(i, n) for i in range(requests) for n in (1, generated)]
random.Random(seed).shuffle(schedule)
results = []
hits_before = adapter.hits if adapter else None
for index, n in schedule:
    inp = prompt(index)
    dist.barrier(group=cpu_group)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = llm.generate([inp], params[n], use_tqdm=False)[0].outputs[0]
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    peer_times = [None] * 2
    dist.all_gather_object(peer_times, elapsed_ms, group=cpu_group)
    ids = list(output.token_ids)
    if len(ids) != n:
        raise RuntimeError("incomplete generation")
    results.append({"index": index, "generated": n,
                    "prompt_sha256": hashlib.sha256(json.dumps(inp["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
                    "ids": ids, "rank_ms": peer_times, "elapsed_ms": max(peer_times)})
if adapter:
    hits = adapter.hits - hits_before
    print(f"E2E_ADAPTER rank={rank} measured_hits={hits} total_hits={adapter.hits} fallbacks={adapter.fallbacks} "
          f"components={sorted(adapter.dx_components)} row_exact={adapter.row_exact}", flush=True)
    if hits < len(schedule):
        raise RuntimeError("measured requests did not all trigger the prefill adapter")
else:
    hits = 0
rank_plan_decisions = None
if adapter and adapter.lookup:
    local=[{'site':k[0],'M':k[1],'K':k[2],'N':k[3],'choice':k[4],'count':v}
           for k,v in adapter.plan_decisions.items()]
    rank_plan_decisions=[None]*2
    dist.all_gather_object(rank_plan_decisions,local,group=cpu_group)
    if rank_plan_decisions[0]!=rank_plan_decisions[1]:raise RuntimeError('plan decisions differ between ranks')
if rank == 0:
    destination = Path(os.environ["E2E_OUT"])
    summary = {str(n): {"median_ms": statistics.median(x["elapsed_ms"] for x in results if x["generated"] == n),
                        "samples_ms": [x["elapsed_ms"] for x in results if x["generated"] == n]}
               for n in (1, generated)}
    payload = {"arm": arm, "model": str(model), "arch": arch, "M": rows,
               "output": generated, "requests": requests, "warmups": warmups, "seed": seed,
               "gpu_util": float(os.environ.get("E2E_GPU_UTIL", "0.65")), "dtype": "bfloat16",
               "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
               "enforce_eager": True, "prefix_caching": False, "components": sorted(adapter.dx_components) if adapter else [],
               "row_exact": adapter.row_exact if adapter else None, "measured_hits_rank0": hits,
               "plan_sha256":adapter.lookup.digest if adapter and adapter.lookup else None,
               "plan_decisions_by_rank":rank_plan_decisions,
               "schedule": schedule, "summary": summary, "results": results}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as target:
        json.dump(payload, target, indent=2)
        target.write("\n")
    print(f"E2E_RESULT {destination} {summary}", flush=True)
dist.barrier(group=cpu_group)
if adapter:
    adapter.close()
    if adapter.cuda is not None:
        import nvshmem.core
        nvshmem.core.finalize()
