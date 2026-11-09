#!/bin/bash
# ONE mpirun rank of the dense Megatron vehicle (TP=8 + SP, full recompute, bf16, mock data).
# Shape from env (set by run.sh): VH_HIDDEN VH_FFN VH_HEADS VH_KVG NL MBS; defaults = 27B vehicle
#   27B: hidden 5120 / ffn 17408 / 40 heads / 8 KV groups / 32 layers
#   70B: hidden 8192 / ffn 28672 / 64 heads / 8 KV groups / 8 layers (memory limit at TP8 x 64 GB)
# Attention is 40H:8KV head_dim 128 for 27B (mcore GPT forces head_dim = hidden / heads); the
# TP communication only sees hidden / ffn / vocab, so the comm geometry matches the real model.
# Arm switches (MGD_* / OPUS_*) arrive exported from run.sh.
set -u
export PYTHONPATH=$SW_ROOT/src:$MEGATRON_DAS:$MEGATRON_LM${SW_PYSITE:+:$SW_PYSITE}
[ -n "${OPUS_HEAP:-}" ] && export DUSHMEM_SYMMETRIC_SIZE=$OPUS_HEAP
if [ -n "${MGD_MXOFF:-}" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export GPU_MAX_HW_QUEUES=${MGD_HWQ:-4}
else
    export GPU_MAX_HW_QUEUES=4
    export CUDA_DEVICE_MAX_CONNECTIONS=1
fi
export TOKENIZERS_PARALLELISM=false
if [ -n "${MGD_CGRAPH:-}" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    export NCCL_GRAPH_REGISTER=1
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
export TRITON_CACHE_DIR=/tmp/tc_swveh_${OMPI_COMM_WORLD_RANK:-0}
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MGD_STATS_EARLY=1

export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${DIST_PORT:-29711}

mkdir -p "${LOGDIR:-/tmp/sw_mgd}"
exec > "${LOGDIR:-/tmp/sw_mgd}/out_r${RANK}.log" 2>&1

GBS=$((8 * ${MBS:-8}))

python -u ${OPUS_PRELUDE:-$SW_ROOT/e2e/megatron/prelude.py} \
    $MEGATRON_DAS/pretrain_gpt.py \
    --seq-length 4096 \
    --num-layers ${NL:-32} \
    --hidden-size ${VH_HIDDEN:-5120} \
    --ffn-hidden-size ${VH_FFN:-17408} \
    --num-attention-heads ${VH_HEADS:-40} \
    --max-position-embeddings 40960 \
    --num-query-groups ${VH_KVG:-8} \
    --group-query-attention \
    --swiglu \
    --qk-layernorm \
    --normalization RMSNorm \
    --no-persist-layer-norm \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    $([ "${VEH_RECOMP:-full}" = full ] && echo "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1") \
    $([ "${VEH_RECOMP:-full}" = selective ] && echo "--recompute-granularity selective") \
    --no-bias-dropout-fusion \
    --no-bias-swiglu-fusion \
    --position-embedding-type rope \
    --no-rope-fusion \
    --untie-embeddings-and-output-weights \
    --transformer-impl local \
    --use-mcore-models \
    --micro-batch-size ${MBS:-8} \
    --global-batch-size $GBS \
    --train-iters ${TRAIN_ITERS:-25} \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --clip-grad 1.0 \
    --bf16 \
    --disable-bias-linear \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --lr 3.0e-5 \
    --lr-decay-style cosine \
    --min-lr 3.0e-6 \
    --lr-warmup-iters 1 \
    --ckpt-format torch \
    --ddp-average-in-collective \
    --overlap-grad-reduce \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --mock-data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $SW_TOKENIZER \
    --vocab-size 248320 \
    $([ -z "${MGD_QUIET:-}" ] && echo --log-throughput) \
    $([ -n "${MGD_NOCHECK:-}" ] && echo --no-check-for-nan-in-loss-and-grad) \
    $([ -n "${MGD_RERUNOFF:-}" ] && echo --rerun-mode disabled) \
    $([ -n "${MGD_CGRAPH:-}" ] && echo --cuda-graph-impl full_iteration --no-overlap-p2p-communication) \
    --eval-iters 2 \
    --log-interval $([ -n "${MGD_QUIET:-}" ] && echo 1000 || echo 1) \
    --save-interval 1000 \
    --eval-interval 1000 \
    $([ -z "${MGD_QUIET:-}" ] && echo "--tensorboard-dir ${LOGDIR}/tb") \
    --rank ${RANK} \
    --world-size ${WORLD_SIZE} \
    --local-rank ${LOCAL_RANK} \
    --dist-url tcp://localhost:${DIST_PORT:-29711}
echo "EXIT $?"
