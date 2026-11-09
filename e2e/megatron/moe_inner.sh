#!/bin/bash
# ONE mpirun rank of the MoE vehicle: Qwen3-30B-A3B shape (hidden 2048, 128 experts top-8,
# expert ffn 768, 12 layers by default), TP=1, EP = MOE_EP (8 on 8 cards; 1 on a single card with
# MOE_E=16 experts = the per-rank expert work of the EP=8 run). Arm switches arrive from run_moe.sh.
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
export TRITON_CACHE_DIR=/tmp/tc_swmoe_${OMPI_COMM_WORLD_RANK:-0}
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
    --num-layers ${NL:-12} \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --num-experts ${MOE_E:-128} \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 768 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type ${MOE_DISPATCHER:-alltoall} \
    $([ -n "${MOE_CAPF:-}" ] && echo --moe-expert-capacity-factor ${MOE_CAPF} --moe-pad-expert-input-to-capacity) \
    --expert-model-parallel-size ${MOE_EP:-8} \
    --max-position-embeddings 40960 \
    --num-query-groups 4 \
    --group-query-attention \
    --swiglu \
    --qk-layernorm \
    --normalization RMSNorm \
    --no-persist-layer-norm \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
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
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --use-distributed-optimizer \
    --mock-data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $SW_MOE_MODEL \
    --vocab-size 151936 \
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
