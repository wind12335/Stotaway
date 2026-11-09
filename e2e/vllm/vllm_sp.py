"""vLLM llama3-70B TP8 prefill: original vs sequence-parallel form with the overlap method and the
predictor (innovations 1 and 2). One arm per process (mpirun -np 8 = vLLM external launcher).

  VS_ARM=stock   stock vLLM (TP AllReduce)                          VLLM_SP=0
  VS_ARM=rccl    SP form, all 4 sites original (RCCL RS / AG)       isolates the SP rewrite itself
  VS_ARM=dx      SP form, all 4 sites overlap method                innovation 1 everywhere
  VS_ARM=plan    SP form, predictor's per-site choice (stage 1)     innovation 2
Launch: e2e/vllm/run.sh (runs the arms ABAB and writes results/raw/vllm/).

Cells (VS_MNBT = max_num_batched_tokens = SP bucket M, default 16384):
  gate   1 prompt of exactly M tokens (one prefill chunk of M) + 32 greedy tokens with logprobs;
         token ids and per-token logprobs saved per arm -> compared offline against stock
  ttft   1 x M tokens, 1 output token (pure prefill of one M chunk), REPS reps
  ttft2  1 x 2M tokens, 1 output token (two M chunks; the second attends to the first)
  thru   16 x M tokens, 1 output token (16 back-to-back M chunks)
Numerics: SP changes the sum order of TP reductions (AR -> RS), so ids may flip on rare near-ties;
the gate reports the first divergence position and the max |logprob diff| over the common prefix.
"""
import csv, ctypes, json, os, sys, time

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
REPS = int(os.environ.get("VS_REPS", "3"))
OUT = os.environ.get("VS_OUT", os.path.join(REPO, "results", "raw", "vllm"))
os.environ["VLLM_SP"] = "0" if ARM == "stock" else "1"
os.environ["VLLM_SP_MODE"] = {"stock": "rccl", "rccl": "rccl", "dx": "dx", "plan": "plan"}[ARM]
os.environ["VLLM_SP_BUCKETS"] = str(MNBT)
os.environ.setdefault("VLLM_SP_PLAN_MODEL", "70b")      # llama3-70B = the planner's 70b shape

# ---- prelude: MPI + dushmem before torch (interposition-order law) ----
import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
world = pHost.mpi_comm_size(mpi, comm)
hip = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so"))
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0
# multi-node (e2e/dual16/vllm16_vcjob.sh): one mpirun of TP ranks per node = one pipeline stage per node;
# the engine world (MPI, symmetric heap) is the node-local TP group, torch.distributed spans all nodes.
NNODES = int(os.environ.get("SW_NNODES", "1"))
NODE = int(os.environ.get("SW_NODE_RANK", "0"))
GRANK = NODE * world + rank
os.environ["RANK"] = str(GRANK); os.environ["LOCAL_RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world * NNODES)
import torch  # noqa: E402
torch.cuda.set_device(rank)
if ARM != "stock":
    from stowaway.kernels.common_ops import init_dcu  # noqa: E402
    host = pHost.DushmemHost(preload_bootstrap=False); host.init_mpi(comm)
    init_dcu(mpi, comm, rank, world, host=host)
    from stowaway.integrations import vllm_sp_patch  # noqa: E402
    vllm_sp_patch.install(rank, world)
if GRANK == 0:
    print(f"[vsp] arm={ARM} TP={world} PP={NNODES} M-bucket={MNBT} model={MODEL}", flush=True)

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(model=MODEL, tensor_parallel_size=world, pipeline_parallel_size=NNODES,
          distributed_executor_backend="external_launcher", seed=0,
          load_format=os.environ.get("VS_LOAD_FORMAT", "auto"),
          dtype="bfloat16", max_model_len=int(os.environ.get("VS_MAX_LEN", max(32768, 2 * MNBT + 64))),
          # 0.85 for EVERY arm: the overlap arms allocate the 4 GB symmetric heap (+ engine state) before
          # vLLM starts, leaving ~56.6 of 64 GiB free; vLLM refuses to start when free < util x total
          # (0.90 -> 57.6 GiB). Same value for stock keeps the KV-cache budget identical across arms.
          gpu_memory_utilization=float(os.environ.get("VS_GPU_UTIL", "0.85")),
          enforce_eager=True, enable_chunked_prefill=True, max_num_batched_tokens=MNBT,
          disable_log_stats=True, enable_prefix_caching=False)
torch.cuda.synchronize()
_tok = llm.get_tokenizer()
_vocab = getattr(_tok, "vocab_size", 128256)


def prompts(nreq, isl, seed):
    out = []
    for r in range(nreq):
        g = torch.Generator().manual_seed(seed * 100003 + r)
        out.append(TokensPrompt(prompt_token_ids=torch.randint(1000, _vocab - 1, (isl,), generator=g).tolist()))
    return out


def run(ps, osl, logprobs=None):
    sp = SamplingParams(temperature=0, max_tokens=osl, ignore_eos=True, logprobs=logprobs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(ps, sp, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, outs


run(prompts(2, 256, 999), 4)                       # warm-up (stock path: M is not a bucket)
run(prompts(1, MNBT, 998), 1)                      # warm-up of the SP path (builds ctxs, JIT)
rows = []
os.makedirs(OUT, exist_ok=True)

# gate
wall, outs = run(prompts(1, MNBT, 777), 32, logprobs=1)
o = outs[0].outputs[0]
ids = list(o.token_ids)
lps = [float(lp[t].logprob) if lp and t in lp else None for t, lp in zip(ids, o.logprobs or [])]
if GRANK == 0:
    json.dump({"arm": ARM, "M": MNBT, "tp": world, "pp": NNODES, "ids": ids, "logprobs": lps},
              open(os.path.join(OUT, f"gate_{ARM}_M{MNBT}.json"), "w"))
    print(f"[vsp] gate {ARM}: first ids {ids[:8]}", flush=True)

for cell, nreq, isl in (("ttft", 1, MNBT), ("ttft2", 1, 2 * MNBT), ("thru", 16, MNBT)):
    reps = REPS if cell != "thru" else max(1, REPS - 1)
    for rep in range(reps):
        wall, _ = run(prompts(nreq, isl, 100 + rep), 1)
        rows.append(dict(ts=time.strftime("%F %T"), arm=ARM, M=MNBT, tp=world, pp=NNODES, cell=cell, rep=rep, nreq=nreq, isl=isl,
                         wall_s=round(wall, 4), tok_s=round(nreq * isl / wall, 1)))
        if GRANK == 0:
            print(f"[vsp] {ARM} {cell} r{rep} n{nreq} isl{isl}: {wall:.3f}s {nreq * isl / wall:.0f} tok/s", flush=True)

if GRANK == 0:
    note = ""
    if ARM != "stock":
        from stowaway.integrations import vllm_sp_patch
        note = vllm_sp_patch.stats_line()
        print(note, flush=True)
    path = os.path.join(OUT, "vsp.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()) + ["note"])
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(dict(r, note=note))
    print(f"[vsp] done -> {path}", flush=True)
    if os.environ.get("VS_DUMP_MODS"):    # used by e2e/dual16/pack_vllm_env.sh: site-packages closure of a run
        import site
        sp = [os.path.realpath(p) for p in site.getsitepackages()]
        tops = set()
        for m in list(sys.modules.values()):
            f = os.path.realpath(getattr(m, "__file__", None) or "")
            for d in sp:
                if f.startswith(d + os.sep):
                    tops.add(f[len(d) + 1:].split(os.sep)[0])
        open(os.environ["VS_DUMP_MODS"], "w").write("\n".join(sorted(tops)) + "\n")
# Final barrier on vLLM's CPU (gloo) world group, NOT the default NCCL group: the default group has never
# run a collective, so barrier() would build a fresh NCCL world communicator here, and with vLLM holding
# 85% of HBM plus the 4 GB heap it fails with "Failed to CUDA calloc" (11-02, 16-card rccl arm, rank 8-15).
from vllm.distributed.parallel_state import get_world_group  # noqa: E402
torch.distributed.barrier(group=get_world_group().cpu_group)
try:
    mpi.MPI_Finalize()
except Exception:
    pass
