#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MEGATRON=${MEGATRON:-/root/data/stowaway-nvidia/third_party/Megatron-LM}
OUT=${OUT:-/root/data/stowaway-nvidia/results/raw/megatron_sp_diagnostic.log}
if [[ -z "${NVSHMEM_HOME:-}" ]]; then
  NVSHMEM_HOME="$(python -c 'import os, site; print(next(os.path.join(p, "nvidia", "nvshmem") for p in site.getsitepackages() if os.path.exists(os.path.join(p, "nvidia", "nvshmem", "lib", "libnvshmem_host.so.3"))))')"
fi
export NVSHMEM_HOME LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-256M}"
export PYTHONPATH="$PWD:$MEGATRON${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export STOWAWAY_DIAGNOSTIC_TORCH_NORM="${STOWAWAY_DIAGNOSTIC_TORCH_NORM:-1}"
export STOWAWAY_DX="${STOWAWAY_DX:-0}"
mkdir -p "$(dirname "$OUT")"
recompute_args=()
if [[ "${STOWAWAY_RECOMPUTE:-0}" == 1 ]]; then
  recompute_args=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi
model_args=()
if [[ "${STOWAWAY_VEHICLE_27B:-0}" == 1 ]]; then
  model_args=(--num-query-groups 8 --group-query-attention --swiglu --qk-layernorm
              --normalization RMSNorm --no-bias-swiglu-fusion --position-embedding-type rope
              --no-rope-fusion --untie-embeddings-and-output-weights)
fi
if [[ "${STOWAWAY_LLAMA70:-0}" == 1 ]]; then
  model_args=(--num-query-groups 8 --group-query-attention --swiglu
              --normalization RMSNorm --no-bias-swiglu-fusion --position-embedding-type rope
              --rotary-base 500000 --no-rope-fusion --untie-embeddings-and-output-weights
              --attention-dropout 0 --hidden-dropout 0)
fi
if [[ "${STOWAWAY_QWEN36:-0}" == 1 ]]; then
  model_args=(--num-query-groups 4 --group-query-attention --swiglu --qk-layernorm
              --normalization RMSNorm --norm-epsilon 1e-6 --no-bias-swiglu-fusion
              --position-embedding-type rope --rotary-base 10000000 --rotary-percent 0.25
              --no-rope-fusion --untie-embeddings-and-output-weights
              --attention-dropout 0 --hidden-dropout 0 --kv-channels 256
              --experimental-attention-variant gated_delta_net --linear-attention-freq 4
              --linear-num-value-heads 48 --attention-output-gate --padded-vocab-size 248320)
fi
optimizer_args=()
if [[ "${STOWAWAY_DISTRIBUTED_OPTIMIZER:-0}" == 1 ]]; then
  optimizer_args=(--use-distributed-optimizer)
  if [[ "${STOWAWAY_OVERLAP_GRAD_REDUCE:-1}" == 1 ]]; then
    optimizer_args+=(--overlap-grad-reduce)
  fi
fi
parallel_args=(--tensor-model-parallel-size "${STOWAWAY_TP:-2}")
if [[ "${STOWAWAY_TP:-2}" != 1 ]]; then
  parallel_args+=(--sequence-parallel)
fi
if [[ "${STOWAWAY_GRAD_REDUCE_BF16:-0}" == 1 ]]; then
  optimizer_args+=(--grad-reduce-in-bf16)
fi
if [[ "${STOWAWAY_OPTIMIZER_CPU_OFFLOAD:-0}" == 1 ]]; then
  optimizer_args+=(--optimizer-cpu-offload --use-precision-aware-optimizer --use-torch-optimizer-for-cpu-offload)
fi
te_args=()
if [[ "${STOWAWAY_TE_OVERLAP:-0}" == 1 ]]; then
  te_args=(--tp-comm-overlap --tp-comm-bootstrap-backend gloo)
fi
moe_args=()
if [[ "${STOWAWAY_MOE:-0}" == 1 ]]; then
  moe_args=(--num-experts "${STOWAWAY_MOE_EXPERTS:-4}" --expert-model-parallel-size 2 --moe-router-topk "${STOWAWAY_MOE_TOPK:-2}"
            --moe-ffn-hidden-size "${STOWAWAY_MOE_FFN:-128}" --moe-token-dispatcher-type alltoall
            --swiglu --no-bias-swiglu-fusion)
  if [[ -n "${STOWAWAY_MOE_ROUTER_DTYPE:-}" ]]; then
    moe_args+=(--moe-router-dtype "$STOWAWAY_MOE_ROUTER_DTYPE")
  fi
fi
bw_args=()
if [[ "${STOWAWAY_BWMATCH:-0}" == 1 ]]; then
  bw_args=(--attention-dropout 0 --hidden-dropout 0 --weight-decay 0.1
           --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.02
           --clip-grad 1.0 --ddp-average-in-collective --lr-warmup-iters 1
           --padded-vocab-size 248320)
fi
timeout "${TEST_TIMEOUT:-120}" python -m torch.distributed.run --standalone --nproc-per-node="${STOWAWAY_WORLD_SIZE:-2}" experiments/megatron_prelude.py "$MEGATRON/pretrain_gpt.py" \
  --num-layers "${STOWAWAY_LAYERS:-2}" --hidden-size "${STOWAWAY_HIDDEN:-128}" \
  --ffn-hidden-size "${STOWAWAY_FFN:-512}" --num-attention-heads "${STOWAWAY_HEADS:-4}" \
  --seq-length "${STOWAWAY_SEQ:-128}" --max-position-embeddings "${STOWAWAY_MAX_POSITION:-${STOWAWAY_SEQ:-128}}" \
  --micro-batch-size "${STOWAWAY_MBS:-1}" --global-batch-size "${STOWAWAY_GBS:-2}" \
  --train-iters "${STOWAWAY_ITERS:-2}" --eval-interval 100 --eval-iters 0 --lr "${STOWAWAY_LR:-0.0001}" --min-lr "${STOWAWAY_MIN_LR:-0.00001}" \
  --lr-decay-style "${STOWAWAY_LR_STYLE:-constant}" "${parallel_args[@]}" --bf16 \
  --no-gradient-accumulation-fusion --no-persist-layer-norm --no-masked-softmax-fusion \
  --no-bias-gelu-fusion --no-bias-dropout-fusion --disable-bias-linear --transformer-impl "${STOWAWAY_TRANSFORMER_IMPL:-local}" \
  --mock-data --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${STOWAWAY_TOKENIZER:-/root/eb-public/huggingface-models/Qwen/Qwen3.6-27B}" \
  --num-workers 0 --log-interval 1 --no-save-optim --no-save-rng "${recompute_args[@]}" "${model_args[@]}" "${optimizer_args[@]}" "${te_args[@]}" "${moe_args[@]}" "${bw_args[@]}" > "$OUT" 2>&1
tail -35 "$OUT"
