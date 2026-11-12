"""Two-rank offline batched generation benchmark with a strict token gate."""
import hashlib
import json
import os
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
if arm not in ("stock", "dx"):
    raise ValueError("invalid arm")
model = Path(os.environ["E2E_MODEL"])
arch = os.environ["E2E_ARCH"]
rows = int(os.environ.get("E2E_M", "1024"))
batch = int(os.environ.get("E2E_BATCH", "4"))
output_length = int(os.environ.get("E2E_OUTPUT", "32"))
repeats = int(os.environ.get("E2E_BATCH_REPEATS", "4"))
warmups = int(os.environ.get("E2E_WARMUPS", "2"))
if rows <= 0 or rows % 2 or min(batch, output_length, repeats, warmups) < 1:
    raise ValueError("invalid benchmark dimensions")
adapter = None
if arm == "dx":
    from stowaway_nvidia.vllm_sp import LlamaSequenceParallel
    adapter = LlamaSequenceParallel(None, rank, "dx", [rows], architecture=arch)
    adapter.install()

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import get_world_group

gpu_util = float(os.environ.get("E2E_GPU_UTIL", ".65"))
llm = LLM(model=str(model), tensor_parallel_size=2, pipeline_parallel_size=1,
          distributed_executor_backend="external_launcher", seed=0, dtype="bfloat16",
          max_model_len=rows + output_length + 64, gpu_memory_utilization=gpu_util,
          enforce_eager=True, enable_chunked_prefill=True, max_num_batched_tokens=rows,
          disable_log_stats=True, enable_prefix_caching=False)
tokenizer = llm.get_tokenizer()
natural = "\n\n".join((model / name).read_text(errors="ignore")
                     for name in ("README.md", "LICENSE", "USE_POLICY.md") if (model / name).exists())
tokens = tokenizer.encode(natural, add_special_tokens=False)
if len(tokens) < 2000:
    raise RuntimeError("natural text source too short")
copies = rows // len(tokens) + 2
stream = tokens * copies

def prompt(index):
    end = (copies - 1) * len(tokens) + (index + 1) * len(tokens) // (batch + 1)
    prefix = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    ids = prefix + stream[end - rows + len(prefix):end]
    if len(ids) != rows:
        raise RuntimeError("unexpected prompt length")
    return TokensPrompt(prompt_token_ids=ids)

inputs = [prompt(i) for i in range(batch)]
hashes = [hashlib.sha256(json.dumps(x["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest() for x in inputs]
sampling = SamplingParams(temperature=0, max_tokens=output_length, ignore_eos=True)
for _ in range(warmups):
    llm.generate(inputs, sampling, use_tqdm=False)
cpu_group = get_world_group().cpu_group
hits_before = adapter.hits if adapter else 0
measurements = []
for _ in range(repeats):
    dist.barrier(group=cpu_group)
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = llm.generate(inputs, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    rank_ms = [None] * 2
    dist.all_gather_object(rank_ms, elapsed, group=cpu_group)
    ids = [list(item.outputs[0].token_ids) for item in out]
    if len(ids) != batch or any(len(x) != output_length for x in ids):
        raise RuntimeError("incomplete batch output")
    measurements.append({"elapsed_ms": max(rank_ms), "rank_ms": rank_ms, "ids": ids})
hits = adapter.hits - hits_before if adapter else 0
if adapter:
    print(f"BATCH_ADAPTER rank={rank} measured_hits={hits} fallbacks={adapter.fallbacks}", flush=True)
    if hits < repeats:
        raise RuntimeError("prefill adapter insufficiently exercised")
if rank == 0:
    median = statistics.median(item["elapsed_ms"] for item in measurements)
    result = {"arm": arm, "model": str(model), "arch": arch, "M": rows, "batch": batch,
              "output": output_length, "repeats": repeats, "warmups": warmups, "gpu_util": gpu_util,
              "dtype": "bfloat16", "enforce_eager": True, "prefix_caching": False,
              "prompt_hashes": hashes, "measurements": measurements, "median_ms": median,
              "generated_tokens_per_second": batch * output_length * 1000 / median,
              "measured_hits_rank0": hits, "components": sorted(adapter.dx_components) if adapter else [],
              "row_exact": adapter.row_exact if adapter else None}
    dest = Path(os.environ["E2E_OUT"])
    with dest.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"BATCH_RESULT {dest} median_ms={median}", flush=True)
dist.barrier(group=cpu_group)
if adapter:
    adapter.close()
    if adapter.cuda is not None:
        import nvshmem.core
        nvshmem.core.finalize()
