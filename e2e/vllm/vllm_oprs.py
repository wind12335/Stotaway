"""vLLM prefill end-to-end A/B with the OPRS runtime (src/stowaway/oprs): every tensor-parallel call
site of every layer is dispatched by a plan. Hygon port of the measurement side of the NVIDIA vLLM
suite (nvidia/experiments/run_vllm_e2e_suite.sh); one arm = one mpirun process (the vLLM external
launcher), launched by e2e/vllm/run_oprs.sh.

What the four call sites (qkv / proj / fc1 / fc2) do in each arm (VS_ARM = name:kind):
  stock          original vLLM (TP AllReduce): no hook, no heap, no plan
  sp_rccl        sequence-parallel (SP) form, every site serial: RCCL AllGather -> the layer's own GEMM,
                 or the layer's own GEMM -> RCCL ReduceScatter. This IS the Null arm (null.json): the
                 hook, the heap and the plan's contexts are there, nothing is replaced -- the reference
                 that separates the SP rewrite itself from the overlap methods
  all_deo        SP form, DEO (Hygon's own overlap method) at every site          (always_deo.json)
  all_<B>        SP form, the external backend B (async_tp | triton_dist) everywhere (always_<B>.json)
  oprs           SP form, per site serial / DEO / B as the plan says              (oprs_<B>.json)
  oprs_deo_only  two-way plan, serial / DEO                                      (oprs_deo_only.json)
  name:<path>    an explicit plan file, run under the name `name`
The SP form runs only for batches of exactly VS_MNBT tokens (the plan's rows); anything else (decode,
short tails) takes the stock forward.

Timing cells (VS_MNBT = M = max_num_batched_tokens; every cell reports the prompt tokens vLLM really
processed):
  init   LLM construction (model load, profiling run in which the dispatcher is built)
  gate   1 prompt of exactly M random tokens + 32 greedy tokens with logprobs -> gate_<arm>_M<M>.json
  ttft   1 x M tokens, 1 output token (one prefill chunk of M), REPS reps
  ttft2  1 x 2M tokens (two M chunks, the second attends to the first)
  thru   16 x M tokens (16 back-to-back M chunks), REPS-1 reps
VS_GATE=1 replaces them by the natural-text gate of e2e/vllm/vllm_gate.py (M tokens of real text, cut
mid-sentence inside a repeated copy -> confident continuations), plus the logprob of every prompt
token (prompt_logprobs): a comparison that does not depend on near-ties.
VS_SEEDED_WEIGHTS=1 (set by the driver for VS_TINY=1): the tiny vehicle has no weights and the fork's
dummy loader fills every parameter, the norms included, with values around 1e-3 -- the model's
next-token distribution is then uniform to 1e-5 and every arm prints the same tokens, so the gate
shows nothing. With seeded weights of a sane scale each arm holds the SAME model with a non-flat
distribution, and token / logprob differences between arms are the numerics of the overlap paths.
Still a random model: this checks the plumbing and the arithmetic, not any real model's behaviour.
Rows -> $VS_OUT/voprs.csv (rank 0 appends); tables -> tools/oprs_tables.py inference.
"""
import csv
import ctypes
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_voprs_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29880")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# the old predictor's SP patch (src/stowaway/integrations/vllm_sp_patch.py, switch VLLM_SP) is another
# system: it is never installed in an OPRS process
os.environ["VLLM_SP"] = "0"


from stowaway.oprs.arms import resolve_arm  # noqa: E402


def _resolve(spec):
    """(name, kind, plan file | None) of this process's arm `name:kind`; PLANS is the plan set and its
    manifest names the external backend B of the deployment (stowaway.oprs.arms)."""
    kind = spec.partition(":")[2] or spec
    try:
        if kind == "stock" or kind.endswith(".json"):
            return resolve_arm(spec, "", "")
        plans = os.environ["PLANS"]
        backend = json.load(open(os.path.join(plans, "manifest.json")))["backend"]
        return resolve_arm(spec, plans, backend)
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"[voprs] bad arm {spec!r} (PLANS={os.environ.get('PLANS', '')!r}): "
                         f"{type(exc).__name__}: {exc}") from None


NAME, KIND, PLAN = _resolve(os.environ.get("VS_ARM", "stock"))
MODEL = os.environ.get("VS_MODEL", os.environ.get("SW_VLLM_MODEL", ""))
WORKLOAD = os.environ.get("VS_WORKLOAD", "llama70_infer")
MNBT = int(os.environ.get("VS_MNBT", "16384"))
REPS = int(os.environ.get("VS_REPS", "3"))
GATE = os.environ.get("VS_GATE", "") == "1"
NP = int(os.environ.get("VS_GATE_NP", "4"))
OSL = int(os.environ.get("VS_GATE_OSL", "48"))
OUT = os.environ.get("VS_OUT", os.path.join(REPO, "results", "raw", "vllm"))
MAX_LEN = int(os.environ.get("VS_MAX_LEN", MNBT + OSL + 64 if GATE else max(32768, 2 * MNBT + 64)))
if KIND != "stock" and not os.path.isfile(PLAN or ""):
    raise SystemExit(f"[voprs] arm {NAME}: plan not found: {PLAN}")

# ---- prelude: MPI + symmetric heap before torch (interposition-order law of this stack) ----
import pydushmem.host as pHost  # noqa: E402
mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
world = pHost.mpi_comm_size(mpi, comm)
hip = ctypes.CDLL(os.environ.get("SW_HIP_LIB", "/opt/dtk-26.04/lib/libamdhip64.so"))
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0
os.environ["RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world)
import torch  # noqa: E402
torch.cuda.set_device(rank)

hook = None
if KIND != "stock":
    from stowaway.oprs import vllm_hook  # noqa: E402
    host = pHost.DushmemHost(preload_bootstrap=False)      # the heap was reserved from DUSHMEM_SYMMETRIC_SIZE
    host.init_mpi(comm)
    methods = [m for m in os.environ.get("VS_METHODS", "").split(",") if m] or None
    vllm_hook.install(mpi, comm, rank, world, host, PLAN, WORKLOAD, methods=methods, buckets=(MNBT,))
    hook = vllm_hook

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

if rank == 0:
    print(f"[voprs] arm={NAME} kind={KIND} plan={PLAN or '<none: stock>'} workload={WORKLOAD} "
          f"M={MNBT} TP={world} model={MODEL} gate={GATE}", flush=True)

t0 = time.perf_counter()
llm = LLM(model=MODEL, tensor_parallel_size=world, distributed_executor_backend="external_launcher", seed=0,
          load_format=os.environ.get("VS_LOAD_FORMAT", "auto"),
          dtype="bfloat16", max_model_len=MAX_LEN,
          # 0.85 for EVERY arm (handoff section 5): the overlap arms hold the symmetric heap besides
          # vLLM's budget; the same value for stock keeps the KV-cache budget identical across arms
          gpu_memory_utilization=float(os.environ.get("VS_GPU_UTIL", "0.85")),
          enforce_eager=True, enable_chunked_prefill=True, max_num_batched_tokens=MNBT,
          disable_log_stats=True, enable_prefix_caching=False)
torch.cuda.synchronize()
init_s = time.perf_counter() - t0
tok = llm.get_tokenizer()
_vocab = getattr(tok, "vocab_size", 128256)

COLUMNS = ["ts", "arm", "kind", "plan_digest", "backend", "M", "tp", "cell", "rep", "nreq", "isl", "wall_s",
           "tok_s", "prefilled_tokens", "cached_tokens", "sp_forwards", "stock_forwards", "hits", "fallbacks",
           "hit_detail", "load_ms", "prepare_ms", "nonserial_entries", "status", "note"]


def reseed(llm, seed=20261126):
    """Weights of a sane scale for the weight-less tiny vehicle (see the module docstring). Seeded on
    the CPU: every arm of a comparison holds the same model. Norms become 1, every matrix gets
    N(0, 1 / fan_in), and the output projection is sharpened x4 so that the distribution is not flat."""
    model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model
    gen, changed = torch.Generator().manual_seed(seed), 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            elif param.dim() == 2:
                scale = (4.0 if name == "lm_head.weight" else 1.0) / math.sqrt(min(param.shape))
                param.copy_((torch.randn(param.shape, generator=gen) * scale).to(param.dtype))
                changed += 1
    if rank == 0:
        print(f"[voprs] seeded weights of {changed} matrices (seed {seed})", flush=True)


if os.environ.get("VS_SEEDED_WEIGHTS", "") == "1":
    reseed(llm)                                     # before the first request is processed


def counts():
    """Hits / fallbacks of this rank (forward-only runs of the dispatcher) and the dispatcher's figures."""
    if hook is None or hook.stats()["disp"] is None:
        return {}
    s = hook.stats()
    hit = {f"{site}={choice}": n for (site, _, _, _, _, choice, _), n in s["runs"].items() if choice != "serial"}
    return {"sp_forwards": s["sp_forwards"], "stock_forwards": s["stock_forwards"],
            "hits": sum(hit.values()), "fallbacks": sum(n for (*_, choice, _) , n in s["runs"].items()
                                                         if choice == "serial"),
            "hit_detail": ",".join(f"{k}:{v}" for k, v in sorted(hit.items())),
            "load_ms": round(s["load_ms"], 1), "prepare_ms": round(s["prepare_ms"], 1),
            "plan_digest": s["plan"][:12], "backend": s["backend"]}


def write(rows):
    if rank != 0:
        return
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "voprs.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


base = {"ts": time.strftime("%F %T"), "arm": NAME, "kind": KIND, "plan_digest": "", "backend": "", "M": MNBT,
        "tp": world, "nreq": "", "isl": "", "tok_s": "", "prefilled_tokens": "", "cached_tokens": "",
        "sp_forwards": 0, "stock_forwards": 0, "hits": 0, "fallbacks": 0, "hit_detail": "", "load_ms": "",
        "prepare_ms": "", "nonserial_entries": "", "status": "", "note": ""}
if KIND != "stock":
    base["nonserial_entries"] = sum(e["choice"] != "serial" for e in json.load(open(PLAN))["entries"])
write([dict(base, cell="init", rep=0, wall_s=round(init_s, 3), **counts())])


def prompts(nreq, isl, seed):
    out = []
    for r in range(nreq):
        g = torch.Generator().manual_seed(seed * 100003 + r)
        out.append(TokensPrompt(prompt_token_ids=torch.randint(1000, _vocab - 1, (isl,), generator=g).tolist()))
    return out


def run(ps, params):
    torch.cuda.synchronize()
    t = time.perf_counter()
    outs = llm.generate(ps, params, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    prefill = sum(len(o.prompt_token_ids) for o in outs)
    cached = sum(int(getattr(o, "num_cached_tokens", 0) or 0) for o in outs)
    return wall, outs, prefill, cached


def status_of(c):
    """ok | NO-SP-FORWARD (the bucket forward never ran) | NO-HITS (the plan replaces, nothing ran)."""
    if KIND == "stock":
        return "ok"
    if not c.get("sp_forwards"):
        return "NO-SP-FORWARD"
    if int(base["nonserial_entries"]) and not c.get("hits"):
        return "NO-HITS"
    return "ok"


rows = []
if GATE:
    text = "\n\n".join(open(os.path.join(MODEL, f), errors="ignore").read()
                       for f in ("README.md", "LICENSE", "USE_POLICY.md") if os.path.exists(os.path.join(MODEL, f)))
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) > 2000, "no text found in the model dir"
    reps = MNBT // len(ids) + 2
    stream = ids * reps

    def prompt(i):
        # every prompt ends inside the LAST repeated copy (seen before -> confident copy) and is
        # exactly MNBT tokens long; the cut point differs per prompt
        end = (reps - 1) * len(ids) + (i + 1) * len(ids) // (NP + 1)
        return TokensPrompt(prompt_token_ids=[tok.bos_token_id] + stream[end - MNBT + 1:end])

    run([TokensPrompt(prompt_token_ids=ids[:256])], SamplingParams(max_tokens=2))       # warm-up (stock path)
    run([prompt(0)], SamplingParams(max_tokens=1))                                        # SP path warm-up
    res = []
    for i in range(NP):
        wall, outs, prefill, cached = run([prompt(i)], SamplingParams(temperature=0, max_tokens=OSL,
                                                                     ignore_eos=True, logprobs=5,
                                                                     prompt_logprobs=1))
        rows.append(dict(base, cell="gate_text", rep=i, nreq=1, isl=MNBT, wall_s=round(wall, 4),
                         prefilled_tokens=prefill, cached_tokens=cached))
        o = outs[0]
        steps = [{str(t): float(v.logprob) for t, v in lp.items()} for lp in o.outputs[0].logprobs]
        # logprob of each prompt token under the model (None at the first position): a tie-free numerics
        # comparison between arms
        plp = [None if lp is None else float(lp[t].logprob) for t, lp in
               zip(o.prompt_token_ids, o.prompt_logprobs or [])]
        res.append({"ids": list(o.outputs[0].token_ids), "top5": steps, "prompt_lp": plp})
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        json.dump({"arm": NAME, "kind": KIND, "M": MNBT, "tp": world, "np": NP, "osl": OSL, "prompts": res},
                  open(os.path.join(OUT, f"gatetext_{NAME}_M{MNBT}.json"), "w"))
        print(f"[vgate] {NAME}: wrote {NP} prompts x {OSL} tokens; first ids {res[0]['ids'][:10]}", flush=True)
else:
    def timed(cell, rep, nreq, isl, seed, record=True):
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        wall, outs, prefill, cached = run(prompts(nreq, isl, seed), params)
        if record:
            rows.append(dict(base, cell=cell, rep=rep, nreq=nreq, isl=isl, wall_s=round(wall, 4),
                             tok_s=round(nreq * isl / wall, 1), prefilled_tokens=prefill, cached_tokens=cached))
        if rank == 0:
            print(f"[voprs] {NAME} {cell} r{rep} n{nreq} isl{isl}: {wall:.3f}s {nreq * isl / wall:.0f} tok/s "
                  f"(prefilled {prefill}){'' if record else ' [warm-up, not a sample]'}", flush=True)

    run(prompts(2, 256, 999), SamplingParams(max_tokens=4))       # warm-up (stock path: M is not a bucket)
    timed("warmup", 0, 1, MNBT, 998)                              # SP path warm-up: builds the dispatcher, JIT
    # random-token gate: 1 x M + 32 greedy tokens with logprobs
    wall, outs, prefill, cached = run(prompts(1, MNBT, 777),
                                       SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, logprobs=1))
    o = outs[0].outputs[0]
    gate_ids = list(o.token_ids)
    gate_lp = [float(lp[t].logprob) if lp and t in lp else None for t, lp in zip(gate_ids, o.logprobs or [])]
    if rank == 0:
        json.dump({"arm": NAME, "M": MNBT, "tp": world, "ids": gate_ids, "logprobs": gate_lp},
                  open(os.path.join(OUT, f"gate_{NAME}_M{MNBT}.json"), "w"))
    rows.append(dict(base, cell="gate", rep=0, nreq=1, isl=MNBT, wall_s=round(wall, 4),
                     prefilled_tokens=prefill, cached_tokens=cached))
    for cell, nreq, isl in (("ttft", 1, MNBT), ("ttft2", 1, 2 * MNBT), ("thru", 16, MNBT)):
        # one unrecorded run of the cell right before its reps: the first use of a path after another
        # one ran (decode steps, a second chunk) can pay a one-time cost that is not the steady state.
        # Seen on one card: the first ttft of a triton_dist arm after the gate took 1.04 s, the next
        # one 0.012 s.
        timed(f"warmup_{cell}", 0, nreq, isl, 50 + len(cell), record=False)
        for rep in range(REPS if cell != "thru" else max(1, REPS - 1)):
            timed(cell, rep, nreq, isl, 100 + rep)

final = counts()
if hook is not None:
    hook.report()                                   # OPRS_SUMMARY / OPRS_RUN ... + stats line, every rank
status = status_of(final)
if rows:
    write([dict(r, **final, status=status) for r in rows])
if rank == 0:
    note = hook.stats_line() if hook is not None else "stock vLLM: no hook, no heap"
    print(f"[voprs] {NAME} done status={status} -> {os.path.join(OUT, 'voprs.csv')}\n[voprs] {note}", flush=True)
if status != "ok":
    # a broken arm must not pass for a slow one: fail the process after everything was written
    raise SystemExit(f"[voprs] {NAME}: {status} (M={MNBT}, bucket {MNBT}, TP={world})")

# Final barrier on vLLM's CPU (gloo) world group, NOT the default NCCL group: the default group has never
# run a collective, so barrier() would build a fresh NCCL world communicator here, and with vLLM holding
# 85% of HBM plus the symmetric heap it fails with "Failed to CUDA calloc".
from vllm.distributed.parallel_state import get_world_group  # noqa: E402
torch.distributed.barrier(group=get_world_group().cpu_group)
try:
    mpi.MPI_Finalize()
except Exception:                                               # noqa: BLE001
    pass
