"""vLLM output-consistency gate on NATURAL text (complements the random-token gate in vllm_sp.py).

Why: with random-token prompts the model's next-token distribution is nearly flat (top logprob around
-2..-5), so bf16 rounding differences (SP changes the sum order of TP reductions; DX GEMMs run in row
blocks) flip greedy picks within a few tokens, and a "first divergence" says nothing about correctness.
Here every prompt is exactly M tokens of real text (the model dir's README / LICENSE / USE_POLICY,
repeated), cut mid-sentence inside a repeated copy, so the continuation is a confident copy and any
real numerical fault shows up as a wrong copy.

Per arm, NP prompts (different cut points), greedy OSL tokens with top-5 logprobs. Output:
  gatetext_<arm>_M<M>.json   ids + top-5 logprobs per step
Compare offline with tools/vllm_gate_report.py <dir> <M>. One prefill of exactly M tokens per prompt =
one SP forward (the decode steps run the stock forward on top of the KV cache the SP prefill wrote).
Env: VS_ARM (stock|rccl|dx|plan), VS_MNBT, VS_GATE_NP (4), VS_GATE_OSL (48), VS_MODEL, VS_OUT.
"""
import ctypes, json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_vsp_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29879")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
ARM = os.environ.get("VS_ARM", "stock")
MODEL = os.environ.get("VS_MODEL", os.environ.get("SW_VLLM_MODEL", ""))
MNBT = int(os.environ.get("VS_MNBT", "16384"))
NP = int(os.environ.get("VS_GATE_NP", "4"))
OSL = int(os.environ.get("VS_GATE_OSL", "48"))
OUT = os.environ.get("VS_OUT", os.path.join(REPO, "results", "raw", "vllm"))
os.environ["VLLM_SP"] = "0" if ARM == "stock" else "1"
os.environ["VLLM_SP_MODE"] = {"stock": "rccl", "rccl": "rccl", "dx": "dx", "plan": "plan"}[ARM]
os.environ["VLLM_SP_BUCKETS"] = str(MNBT)
os.environ.setdefault("VLLM_SP_PLAN_MODEL", "70b")

import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
world = pHost.mpi_comm_size(mpi, comm)
hip = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so"))
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0
NNODES = int(os.environ.get("SW_NNODES", "1"))           # multi-node: one pipeline stage per node (vllm16_vcjob.sh)
GRANK = int(os.environ.get("SW_NODE_RANK", "0")) * world + rank
os.environ["RANK"] = str(GRANK); os.environ["LOCAL_RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world * NNODES)
import torch  # noqa: E402
torch.cuda.set_device(rank)
if ARM != "stock":
    from stowaway.kernels.common_ops import init_dcu  # noqa: E402
    host = pHost.DushmemHost(preload_bootstrap=False); host.init_mpi(comm)
    init_dcu(mpi, comm, rank, world, host=host)
    from stowaway.integrations import vllm_sp_patch  # noqa: E402
    vllm_sp_patch.install(rank, world)

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(model=MODEL, tensor_parallel_size=world, pipeline_parallel_size=NNODES,
          distributed_executor_backend="external_launcher", seed=0,
          dtype="bfloat16", max_model_len=MNBT + OSL + 64,
          gpu_memory_utilization=float(os.environ.get("VS_GPU_UTIL", "0.85")),
          enforce_eager=True, enable_chunked_prefill=True, max_num_batched_tokens=MNBT,
          disable_log_stats=True, enable_prefix_caching=False)
tok = llm.get_tokenizer()
text = "\n\n".join(open(os.path.join(MODEL, f), errors="ignore").read()
                   for f in ("README.md", "LICENSE", "USE_POLICY.md") if os.path.exists(os.path.join(MODEL, f)))
base = tok.encode(text, add_special_tokens=False)
assert len(base) > 2000, "no text found in the model dir"
reps = MNBT // len(base) + 2
stream = (base * reps)


def prompt(i):
    # all prompts end inside the LAST repeated copy (the model has seen the text before -> confident copy);
    # different cut points per prompt, each exactly MNBT tokens long
    end = (reps - 1) * len(base) + (i + 1) * len(base) // (NP + 1)
    return TokensPrompt(prompt_token_ids=[tok.bos_token_id] + stream[end - MNBT + 1:end])


llm.generate([TokensPrompt(prompt_token_ids=base[:256])], SamplingParams(max_tokens=2), use_tqdm=False)
llm.generate([prompt(0)], SamplingParams(max_tokens=1), use_tqdm=False)        # SP path warm-up
res = []
for i in range(NP):
    o = llm.generate([prompt(i)], SamplingParams(temperature=0, max_tokens=OSL, ignore_eos=True, logprobs=5),
                     use_tqdm=False)[0].outputs[0]
    steps = [{str(t): float(v.logprob) for t, v in lp.items()} for lp in o.logprobs]
    res.append({"ids": list(o.token_ids), "top5": steps})
if GRANK == 0:
    os.makedirs(OUT, exist_ok=True)
    json.dump({"arm": ARM, "M": MNBT, "np": NP, "osl": OSL, "prompts": res},
              open(os.path.join(OUT, f"gatetext_{ARM}_M{MNBT}.json"), "w"))
    print(f"[vgate] {ARM}: wrote {NP} prompts x {OSL} tokens; first ids {res[0]['ids'][:10]}", flush=True)
    if ARM != "stock":
        print(vllm_sp_patch.stats_line(), flush=True)
# Final barrier on vLLM's CPU (gloo) world group, NOT the default NCCL group: the default group has never
# run a collective, so barrier() would build a fresh NCCL world communicator here, and with vLLM holding
# 85% of HBM plus the 4 GB heap it fails with "Failed to CUDA calloc" (11-02, 16-card rccl arm, rank 8-15).
from vllm.distributed.parallel_state import get_world_group  # noqa: E402
torch.distributed.barrier(group=get_world_group().cpu_group)
try:
    mpi.MPI_Finalize()
except Exception:
    pass
