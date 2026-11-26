#!/bin/bash
# ONE mpirun rank of the dense Megatron vehicle with the OPRS runtime (TP + SP, bf16, mock data).
# Copy of arm_inner.sh with the tensor-parallel size as a parameter and WITHOUT any old control-plane
# switch (MGD_* / OPUS_*): what runs here is e2e/megatron/prelude.py + src/stowaway/oprs/megatron_hook.py.
# Shape from env (set by run_oprs.sh): VH_HIDDEN VH_FFN VH_HEADS VH_KVG NL MBS SEQ ACC VOCAB
# OPRS_TOKENIZER; defaults = 27B vehicle (ACC = micro-batches per iteration, default 8)
#   27B: hidden 5120 / ffn 17408 / 40 heads / 8 KV groups / 32 layers
#   70B: hidden 8192 / ffn 28672 / 64 heads / 8 KV groups / 8 layers (memory limit at TP8 x 64 GB)
# Attention is 40H:8KV head_dim 128 for 27B (mcore GPT forces head_dim = hidden / heads); the TP
# communication only sees hidden / ffn / vocab, so the comm geometry matches the real model.
# OPRS switches arrive exported from run_oprs.sh:
#   TP               cards = tensor-model-parallel size (pipeline and context parallel stay 1)
#   OPRS_PLAN        plan file (e2e/megatron/run_oprs.sh); unset = the serial arm: no hook, no heap
#   OPRS_WORKLOAD    workload name the plan was generated for (bw27b_train, bw70b_train, custom_train)
#   OPRS_METHODS     overlap methods this process may run (comma list; default all, deo included)
#   OPRS_DEO_BWD     ag (default: fused AG + dgrad backward at the row sites) | stock
#   OPRS_HEAP        symmetric-heap bytes (the arms that allocate contexts)
set -u
export PYTHONPATH=$SW_ROOT/src:$MEGATRON_DAS:$MEGATRON_LM${SW_PYSITE:+:$SW_PYSITE}
[ -n "${OPRS_HEAP:-}" ] && export DUSHMEM_SYMMETRIC_SIZE=$OPRS_HEAP
export GPU_MAX_HW_QUEUES=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/tc_swveh_${OMPI_COMM_WORLD_RANK:-0}
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OPRS_TRITON_DIST_ROOT=${OPRS_TRITON_DIST_ROOT:-/root/private_data/tri-dist-hygon}

export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${DIST_PORT:-29711}

mkdir -p "${LOGDIR:-/tmp/sw_mgd}"
exec > "${LOGDIR:-/tmp/sw_mgd}/out_r${RANK}.log" 2>&1

TP=${TP:-8}
# the vehicle's GBS = 8 x MBS (arm_inner.sh): data parallel size is always 1 here, so an iteration is
# ACC = 8 micro-batches at any TP. The call-site row count M = SEQ x MBS per micro-batch (the plan's M).
GBS=$(( ${ACC:-8} * ${MBS:-8} ))
SEQ=${SEQ:-4096}
echo "[oprs-inner] rank=$RANK tp=$TP plan=${OPRS_PLAN:-<none: serial arm>} workload=${OPRS_WORKLOAD:-} heap=${DUSHMEM_SYMMETRIC_SIZE:-<default>} seq=$SEQ mbs=${MBS:-8} gbs=$GBS hidden=${VH_HIDDEN:-5120} ffn=${VH_FFN:-17408} heads=${VH_HEADS:-40} kv=${VH_KVG:-8} layers=${NL:-32} recomp=${VEH_RECOMP:-full} iters=${TRAIN_ITERS:-25}"

python -u $SW_ROOT/e2e/megatron/prelude.py \
    $MEGATRON_DAS/pretrain_gpt.py \
    --seq-length $SEQ \
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
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --mock-data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${OPRS_TOKENIZER:-$SW_TOKENIZER} \
    --vocab-size ${VOCAB:-248320} \
    $([ -z "${OPRS_QUIET:-}" ] && echo --log-throughput) \
    --eval-iters 2 \
    --log-interval $([ -n "${OPRS_QUIET:-}" ] && echo 1000 || echo 1) \
    --save-interval 1000 \
    --eval-interval 1000 \
    --rank ${RANK} \
    --world-size ${WORLD_SIZE} \
    --local-rank ${LOCAL_RANK} \
    --dist-url tcp://localhost:${DIST_PORT:-29711}
rc=$?
echo "EXIT $rc"
exit $rc                                  # the process exit code must reach mpirun and the driver
