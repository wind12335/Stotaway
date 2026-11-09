# Third-party dependencies

Nothing here is vendored except small patches and stubs. `env/env.sh` looks for the checkouts and
model directories at the paths below (all overridable by env vars); symlinks are fine.

## 1. Platform (tested versions)

| Component | Version | Notes |
|---|---|---|
| Hardware | Hygon BW1000 (gfx936) × 8 per node, 64 GB each | PCIe P2P full mesh inside the node |
| DTK | 26.04 (`/opt/dtk-26.04`) | hipcc, hipBLAS / hipBLASLt, RCCL, **dushmem** (symmetric heap) |
| Python | 3.11.9 | |
| torch | 2.9.0 (DTK build) | ships `torch.libs/libmpi-*.so`, used for the engine bootstrap |
| triton | 3.3.0 (DTK build, gfx936 backend) | |
| Open MPI | 5.0.3 (`mpirun`) | launcher only; one process per card |
| vLLM | 0.11.0+das.opt1.dtk2604.torch290 (das fork wheel) | only for `e2e/vllm` |

## 2. Megatron-LM (training vehicles, `e2e/megatron`, `e2e/dual16`)

```bash
cd 3rdparty
git clone https://github.com/HYGON-AI/Megatron-LM-das.git
cd Megatron-LM-das
git checkout 52b4450c3ea3031615aa1e324334c5268a2ee6a7
git submodule update --init 3rdparty/Megatron-LM          # NVIDIA Megatron-LM core_r0.18.0
(cd 3rdparty/Megatron-LM && git checkout 571370c829ca768fe37244f4e2e7f28d8accc4ab)
git apply ../patches/megatron-das-52b4450.patch
(cd 3rdparty/Megatron-LM && git apply ../../../patches/megatron-lm-571370c.patch)
```

| Patch | What it changes | Effect on the paper's runs |
|---|---|---|
| `megatron-das-52b4450.patch`, `recompute.py` | defines `micro_sp_idx_` before the checkpoint branch (stock raises UnboundLocalError under block recompute) | bug fix, needed for `--recompute-granularity full` |
| `megatron-das-52b4450.patch`, `param_and_grad_buffer.py` | optional `MGD_COMM_STREAM=1`: grad reduce on a side stream for a single distributed-optimizer instance | **off by default**; no paper arm sets it |
| `megatron-lm-571370c.patch`, `hybrid.py` | `typing.override` → `typing_extensions.override` | Python 3.11 compatibility |

Megatron itself is otherwise unmodified: every Stowaway change is a monkey patch installed by
`e2e/megatron/prelude.py`, and only when an arm's `MGD_*` / `OPUS_*` switches are set.

## 3. vLLM on DCU (`e2e/vllm`)

The das vLLM wheel imports three packages that are absent on DTK. `dcu_shims/` provides them:

| Shim | What it does |
|---|---|
| `amdsmi` | bridges the few `amdsmi_*` calls vLLM's ROCm platform makes to DTK's ROCm-SMI C API |
| `lightop` | stub; `reshape_and_cache_cuda` is a **lazy proxy** to `vllm._custom_ops` (the das fork turns `VLLM_USE_OPT_RESHAPE_AND_CACHE=1` on for Qwen3, so this call is on the hot path); everything else raises when called |
| `lmslim` | stub that raises when called (quantization kernels, unused for bf16) |

Install: copy the three directories into site-packages, or `export SW_PYSITE=<repo>/3rdparty/dcu_shims`.
A real flash-attn wheel for DTK is also required (vendor-provided).

## 4. Models and tokenizers

Training uses `--mock-data`; the tokenizer only fixes the vocabulary size.

| Env var | Default path | Used by |
|---|---|---|
| `SW_TOKENIZER` | `3rdparty/tokenizers/Qwen3.6-27B` | dense 27B / 70B vehicles (vocab 248320) |
| `SW_MOE_MODEL` | `3rdparty/models/Qwen3-30B-A3B` | MoE vehicle (config + tokenizer, vocab 151936) |
| `SW_VLLM_MODEL` | `3rdparty/models/LLAMA3-70B` | vLLM prefill bench (full HF weights) |

## 5. Cluster pods (16-card job)

Pod images may miss triton / einops / transformers / flash_attn. Put a copy of the needed
site-packages on shared storage as `3rdparty/pysite` (symlink; here -> `/root/private_data/pysite_addons`).
`e2e/dual16/vcjob.sh` (Megatron) uses it by default (`SW_PYSITE`); `e2e/dual16/arm_inner_16.sh` prepends it.
The vLLM job instead unpacks `3rdparty/pyenv_vllm16.tar.gz` (`e2e/dual16/pack_vllm_env.sh`), which carries
vllm, flash_attn, the shims, **triton** and every other vLLM runtime dependency.
torch and numpy stay the pod's own (11-02: the cluster image has torch / numpy but no triton).
