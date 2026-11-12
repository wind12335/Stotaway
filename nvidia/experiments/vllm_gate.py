"""Experimental two-rank, real-text vLLM Llama greedy-token comparison.

Run stock, stock_sp, and dx as separate torchrun processes, same settings.
Requires the isolated vLLM environment and, for dx, NVSHMEM dependencies.
"""
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
if world != 2 or local_rank != rank:
    raise ValueError("requires exactly two local ranks")
torch.cuda.set_device(local_rank)
arm = os.environ.get("VS_ARM", "stock")
if arm not in ("stock", "stock_sp", "dx"):
    raise ValueError("VS_ARM must be stock, stock_sp or dx")
model = Path(os.environ.get("VS_MODEL", "/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-8B"))
rows = int(os.environ.get("VS_MNBT", "1024"))
count = int(os.environ.get("VS_GATE_NP", "4"))
length = int(os.environ.get("VS_GATE_OSL", "48"))
if rows <= 0 or rows % 2 or count <= 0 or length <= 0:
    raise ValueError("VS_MNBT must be even and all counts positive")
destination = Path(os.environ.get("VS_OUT", "/root/data/stowaway-nvidia/results/vllm"))
adapter = None
if arm != "stock":
    from stowaway_nvidia.vllm_sp import LlamaSequenceParallel

    adapter = LlamaSequenceParallel(None, rank, arm, [rows], architecture=os.environ.get("VS_ARCH", "llama"))
    adapter.install()

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import get_world_group

llm = LLM(model=str(model), tensor_parallel_size=2, pipeline_parallel_size=1,
          distributed_executor_backend="external_launcher", seed=0, dtype="bfloat16",
          max_model_len=rows + length + 64,
          gpu_memory_utilization=float(os.environ.get("VS_GPU_UTIL", "0.65")),
          enforce_eager=True, enable_chunked_prefill=True, max_num_batched_tokens=rows,
          disable_log_stats=True, enable_prefix_caching=False)
tokenizer = llm.get_tokenizer()
text = "\n\n".join((model / name).read_text(errors="ignore")
                     for name in ("README.md", "LICENSE", "USE_POLICY.md") if (model / name).exists())
tokens = tokenizer.encode(text, add_special_tokens=False)
if len(tokens) < 2000:
    raise RuntimeError("natural-text gate needs at least 2000 tokens of model documentation")
reps = rows // len(tokens) + 2
stream = tokens * reps


def prompt(index):
    end = (reps - 1) * len(tokens) + (index + 1) * len(tokens) // (count + 1)
    prefix = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    return TokensPrompt(prompt_token_ids=prefix + stream[end - rows + len(prefix):end])


llm.generate([TokensPrompt(prompt_token_ids=tokens[:256])], SamplingParams(max_tokens=2), use_tqdm=False)
llm.generate([prompt(0)], SamplingParams(max_tokens=1), use_tqdm=False)
results = []
for index in range(count):
    input_prompt = prompt(index)
    prompt_hash = hashlib.sha256(json.dumps(input_prompt["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()
    result = llm.generate([input_prompt], SamplingParams(temperature=0, max_tokens=length,
                          ignore_eos=True, logprobs=5), use_tqdm=False)[0].outputs[0]
    results.append({"prompt_sha256": prompt_hash, "ids": list(result.token_ids),
                    "top5": [{str(token): float(value.logprob) for token, value in step.items()}
                             for step in result.logprobs]})
if adapter is not None:
    print(f"VLLM_SP_COUNTS rank={rank} hits={adapter.hits} fallbacks={adapter.fallbacks} "
          f"contexts={len(adapter.contexts)} components={sorted(adapter.dx_components)} "
          f"row_exact={adapter.row_exact}", flush=True)
    if adapter.row_diagnostics:
        print(f"VLLM_ROW_DIAGNOSTIC rank={rank} calls={len(adapter.row_diagnostics)} "
              f"max_abs={max(x[0] for x in adapter.row_diagnostics)} "
              f"different_elements={sum(x[1] for x in adapter.row_diagnostics)} "
              f"elements={sum(x[2] for x in adapter.row_diagnostics)} "
              f"split_reference_max_abs={max(x[3] for x in adapter.row_diagnostics)} "
              f"relative_rms={(sum(x[4] for x in adapter.row_diagnostics) / max(sum(x[5] for x in adapter.row_diagnostics), 1e-30)) ** .5}", flush=True)
    if adapter.hits < count + 1:
        raise RuntimeError("SP path did not cover warmup and all prompt prefills; output gate is invalid")
if rank == 0:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"gatetext_{arm}_M{rows}.json"
    with output.open("x", encoding="utf-8") as target:
        json.dump({"arm": arm, "model": str(model), "M": rows, "np": count,
                   "osl": length, "dx_components": sorted(adapter.dx_components) if adapter else [],
                   "row_exact": adapter.row_exact if adapter else False,
                   "prompts": results}, target, indent=2)
    print(f"wrote {output}", flush=True)
torch.distributed.barrier(group=get_world_group().cpu_group)
if adapter is not None:
    adapter.close()
    if adapter.cuda is not None:
        import nvshmem.core
        nvshmem.core.finalize()
