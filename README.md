# Stowaway

**Predict where tensor-parallel communication can hide, then hide it there.**

Stowaway speeds up Megatron TP + sequence-parallel training (and MoE expert parallelism) on
Hygon DCU (BW1000, 8 cards per node). It never touches the vendor GEMM; communication rides along
("stows away") behind the compute that is already there. Two parts:

1. **Planner (predictor)** — `planner/`. Offline and CPU-only. For each of the 4 TP call sites
   (QKV, o_proj, fc1, fc2) it decides whether overlap pays off for *this* model shape and
   micro-batch size. Stage 1 is an operator-level cost model. Stage 2 is a step-level
   control-plane term. Stage 2 is needed because a fused AG can make the GPU wait on the host,
   and that wait can cancel an operator-level win.
2. **Overlap runtime** — `src/stowaway/`. It executes the plan inside Megatron:
   - **DX overlap** for AG+GEMM and GEMM+RS: block-wise GEMM, the copy engine pushes each block
     straight to its owner in permutation order, ticket-gated, no barrier.
   - **Host-sync-free MoE layer**: SM-store dispatch/combine plus a grouped expert GEMM with
     device-side counts. The CPU never waits for the GPU.

```
 model shape, MBS ──► planner/planner.py ──► per-site plan  (e.g. fbag: o_proj+fc2 overlapped, QKV+fc1 original)
   (+ one short           stage 1: operator cost model          │
    baseline run)         stage 2: step-level host-wait term    ▼
                                                     e2e/megatron/run.sh  kind=plan
                                                     └─ prelude installs the hooks for the chosen sites only
```

Headline results (details in `results/`):

| | Result |
|---|---|
| Operator level, 8 cards | GEMM+RS 1.30× geomean (30/32 cells faster, up to 1.85×); AG+GEMM 1.13× at M = 32768 |
| Megatron end to end, 8 cards (vs null control) | 27B 1.057×, 70B 1.063–1.077× |
| Megatron end to end, 16 cards (2 nodes, RDMA) | 70B 1.079×, 27B 1.054×; the predictor's pick beats the alternative by 3.46% / 2.77% |
| Predictor | stage 1 leave-one-model-out: 0/97 cells slower than original; stage 2: 5/5 in-sample, 4/5 leave-one-out, 2/2 out-of-sample on 16 cards |
| vLLM LLaMA3-70B prefill, 8 cards (full 80 layers, M 16384) | all four sites overlapped 1.14–1.17× vs original vLLM (1.11–1.13× vs the SP rewrite alone); output 192/192 tokens identical on the natural-text gate; the predictor's stage-1 pick (o_proj+fc2 only) gets 1.06× — conservative here |
| vLLM LLaMA3-70B prefill, 16 cards (TP8 x PP2, RDMA) | all four sites overlapped 1.19–1.22× (1.18–1.19× vs the SP rewrite alone); predictor pick 1.16–1.18×; output 192/192 identical |
| MoE layer (Qwen3-30B-A3B, EP8) | operator 1.17–1.57× fwd+bwd; end to end 1.116× (8 cards), 1.188× expert MLP (1 card) |

---

## Repository layout

```
env/env.sh            all paths and env vars (source this first)
src/stowaway/         the runtime (python package)
  kernels/              DX overlap ops (gemm_rs_dx.py), rings/relays, MoE sync-free path (moe_*.py)
  integrations/         Megatron / vLLM hooks; every hook default OFF, switched by MGD_* / OPUS_* env
src/pydushmem/        ctypes bindings for DTK dushmem (symmetric heap, bootstrap)
src/csrc/             hipBLASLt grouped-GEMM shim for MoE (build.sh)
planner/              the predictor (CPU): planner.py, costmodel.py, step_planner_v3.py, moe_plan.py
  data/                 measured cells, GEMM probes, fitted step points
bench/                operator-level benches + drivers (AG+GEMM, GEMM+RS, AG+dgrad, MoE layer, smoke)
probes/               micro-probes behind the laws the planner uses (host-block, permutation bandwidth, graph tax)
e2e/megatron/         Megatron vehicles: dense TP8+SP (run.sh) and MoE EP8 (run_moe.sh)
e2e/dual16/           2-node / 16-card job (TP8 x DP2, RDMA)
e2e/vllm/             vLLM LLaMA3-70B prefill in SP form (original / overlap / predictor)
e2e/paper.sh          one entry point per paper table
tools/                env check, run summaries, report generators
results/              paper tables (*.md), findings log (findings.md), raw logs (raw/)
docs/                 predictor principle, glossary, paper narrative, diagrams
3rdparty/             upstream commits + patches (Megatron-LM-das), DCU shims for vLLM
```

Naming: the arm kinds (`base`, `fnull`, `fbag`, `fbc`, ...) and the env switches (`MGD_*`, `OPUS_*`)
keep their development names so that `results/raw/` logs and `results/findings.md` stay traceable.
The arm table is at the top of `e2e/megatron/run.sh`, and terms are explained in `docs/术语表.md`.

---

## Reproduce step by step

### Step 0 — prerequisites
You need a Hygon DCU node with DTK 26.04, Python 3.11, and the DTK builds of torch 2.9 and
triton 3.3. You also need Open MPI 5 (`mpirun`). Exact versions are in `3rdparty/README.md` §1.

### Step 1 — get the third-party pieces
```bash
git clone <this repo> stowaway && cd stowaway
# Megatron-LM-das at the tested commit + our 3 small patches (3rdparty/README.md §2)
cd 3rdparty && git clone https://github.com/HYGON-AI/Megatron-LM-das.git && cd Megatron-LM-das
git checkout 52b4450c3ea3031615aa1e324334c5268a2ee6a7 && git submodule update --init 3rdparty/Megatron-LM
(cd 3rdparty/Megatron-LM && git checkout 571370c829ca768fe37244f4e2e7f28d8accc4ab && git apply ../../../patches/megatron-lm-571370c.patch)
git apply ../patches/megatron-das-52b4450.patch && cd ../..
# tokenizer / model dirs (or export SW_TOKENIZER / SW_MOE_MODEL / SW_VLLM_MODEL)
ln -s <Qwen3.6-27B tokenizer dir>  3rdparty/tokenizers/Qwen3.6-27B
ln -s <Qwen3-30B-A3B dir>          3rdparty/models/Qwen3-30B-A3B      # MoE only
ln -s <LLaMA3-70B HF dir>          3rdparty/models/LLAMA3-70B         # vLLM only
```

### Step 2 — environment and build
```bash
source env/env.sh            # DTK, PYTHONPATH=src, MPI lib for the bootstrap, 8 GB symmetric heap
bash src/csrc/build.sh       # MoE grouped-GEMM shim (needed for MoE only)
python3 tools/check_env.py   # every line OK (WARN lines only matter for the part they name)
```

### Step 3 — smoke test (8 cards, ~10 min)
```bash
bash bench/smoke.sh
```
This runs one small cell of every operator plus the MoE layer. Every `correctness` line must show
`fails=0` / `PASS`. Speedups at this small M are only for orientation.

### Step 4 — the predictor alone (CPU, seconds)
```bash
python3 -I planner/planner.py --selftest                          # reproduces 5/5 choices on the measured points
python3 -I planner/planner.py --model 70b --mbs 8                 # plan for a built-in shape
python3 -I planner/planner.py --hidden 8192 --ffn 28672 --heads 64 --kv 8 --layers 80 --mbs 4 --step-ms 9655
python3 -I planner/validate.py                                    # stage-1 leave-one-model-out -> planner/data/planner_validation.md
```
The planner prints a per-site table and ends with `**可部署配置：<kind>…**` (the deployable config).
Give it `--step-ms` (the step time of a ≤3-iteration original run) for the stage-2 term. Without it,
it estimates the step time from GEMM times. For new shapes with no probe file, it times the 4 GEMMs
on one card.

### Step 5 — operator-level tables (8 cards, ~40 min ≈ 5 card-hours)
```bash
bash e2e/paper.sh ops        # 11 models x M 8192/16384/32768, AG+GEMM and GEMM+RS
                             # -> results/raw/ops_8card/, then planner/extract.py + gen_table.py
                             # -> results/算子级结果表.md
```

### Step 6 — end-to-end Megatron, predictor + overlap (8 cards)
```bash
# the predictor decides (plan arm), compared with the original and the null control, ABAB:
SW_MODEL=70b MBS=8 ARMS="b1:base n1:fnull p1:plan b2:base n2:fnull p2:plan" bash e2e/megatron/run.sh
# -> results/raw/e2e_8card/70b_mbs8_<date>/{driver.log,summary.md,plan_p1.md}
```
`plan` arms read the median step time of a `base` arm already finished in the same run.
`summary.md` gives speedup vs `base` (original) and vs `fnull` (null control: same heap and hooks,
nothing taken over = pure overlap gain). The paper's tables are groups of `e2e/paper.sh`:

| `e2e/paper.sh <group>` | Table | Cost |
|---|---|---|
| `main` | 27B / 70B MBS8: base, fnull, fbag | ~16 card-h |
| `ablation` | 27B: base → fdk → fallt → fbag | ~13 card-h |
| `agchoice` | fbag vs fbc at 5 shapes (stage-2 fit points) | ~18 card-h |
| `mbs` | 70B base vs fbag at MBS 1/2/4/8 | ~8 card-h |
| `predictor` | base / fnull / plan at 5 shapes | ~26 card-h |
| `moe-op`, `moe-e2e`, `moe-1gpu` | MoE layer / vehicle | 0.3 / 1.5 / 0.6 card-h |

Rules for reading the numbers: compare only within one run. The absolute step time drifts 5–7%
between days. Each arm is its own process, and the median is taken from iteration 3 on.

### Step 7 — 16 cards (2 nodes) and vLLM
- 16 cards: `e2e/dual16/README.md` covers it. Submit the same command on both pods; about 42 card-hours.
- vLLM, 8 cards: `VS_ARMS="stock dx" VS_REPS=1 bash e2e/vllm/run.sh` (smoke), then the default 8 arms
  (~6 card-hours), then `bash e2e/vllm/run_gate.sh` (natural-text output check, ~1.5 card-hours).
  Results: `results/vLLM结果表.md`. This needs the shims in `3rdparty/README.md` §3.
- vLLM, 16 cards (TP8 x PP2): `e2e/dual16/vllm16_vcjob.sh`, see `e2e/dual16/README.md` (~24 card-hours).

---

## How it works (one paragraph each)

**DX overlap (`src/stowaway/kernels/gemm_rs_dx.py`).**
- **GEMM+RS:** the local GEMM runs in R row blocks. As soon as a block is done, the copy engine
  pushes it directly into its owner's symmetric heap, in permutation order (rank r sends to r+j at
  step j). Every link carries one flow, at ~141 GB/s/rank, while the next block computes. A 4-byte
  ticket per block tells the owner the data landed; there is no barrier. The owner sums the R
  blocks in fp32.
- **AG+GEMM:** the mirror image. Blocks of the input arrive in permutation order, and each one is
  multiplied by the vendor GEMM as soon as its ticket is set.
- **Backward:** a fused AG+dgrad, so all four TP call sites have a fused form.

**Why a predictor (`planner/`, `docs/预测器原理.md`).**
- The best choice differs per site and per shape. On 70B, overlapping QKV/fc1 wins at MBS8 but
  loses at MBS4 and MBS2. Overlapping o_proj/fc2 loses at 70B MBS2.
- **Stage 1** predicts each site's operator time from its shape and the measured single-card GEMM
  time. A guard requires the pessimistic estimate to beat the original by 2%.
- **Stage 2** adds what operator benchmarks cannot see. On this platform, an engine copy into
  another process's memory pulls the host back to the GPU's progress. So every fused AG call costs
  GPU idle time e = 0.0096 × (GPU time per layer cycle) when the cycle is shorter than ~205 ms.
  The RS group falls back to the original below ~111 ms per cycle.
- The planner picks the subset of sites with the best predicted step time. It only switches on
  sites predicted to be at least 1% faster.

**Host-sync-free MoE (`moe_sf.py`, `moe_expert_mlp.py`, `moe_ggemm.py`).**
- Megatron's MoE path reads token counts on the host twice per layer (`.tolist()`). Each read
  stalls the CPU behind the GPU.
- Stowaway dispatches with SM stores, using device atomics to claim slots in the expert rank's
  heap. Ranks wait on 4-byte flags with `hipStreamWaitValue32`, and ACK planes protect buffer
  reuse.
- Rows are sorted by local expert on the device, and hipBLASLt grouped GEMM runs on device-written
  arguments. Virtual groups of B rows keep the grid fixed, and `planner/moe_plan.py` picks B per
  shape.
- Results return to their source rows by tag. No host read and no barrier remain.

---

## License

MIT (`LICENSE`). The DCU kernel layer derives from ByteDance triton-distributed (MIT); see `NOTICE`
and `LICENSE.triton-dist`. Megatron-LM is not redistributed: `3rdparty/patches/` holds diffs only.
