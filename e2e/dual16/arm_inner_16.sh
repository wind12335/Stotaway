#!/bin/bash
# e2e/dual16/arm_inner_16.sh -- ONE mpirun rank of the 16-card vehicle (2 nodes x 8 cards, TP8 x DP2).
# Same model args as e2e/megatron/arm_inner.sh (shape via VH_*, default 27B), three 16-card changes:
#   1) global RANK = local OMPI rank + NODE_RANK * 8; megatron --dist-url points at node 0
#      (torch TCPStore rendezvous across pods, no SSH);
#   2) DP=2 follows from world 16 / (TP8 x PP1); GBS = 16 x MBS keeps 8 micro-batches per DP group;
#   3) MGD_BRS_MODS=mappings,layers: DP=2 activates the cross-node RS sites of param_and_grad_buffer;
#      the symmetric heap is node-local, so those must stay on RCCL (forced here, idempotent).
# MPI is only the engine-world bootstrap (mpirun of the 8 local ranks).
set -u
export PYTHONPATH=$SW_ROOT/src:$MEGATRON_DAS:$MEGATRON_LM
[ -n "${OPUS_HEAP:-}" ] && export DUSHMEM_SYMMETRIC_SIZE=$OPUS_HEAP
# Pod images may lack triton / einops / transformers / flash_attn ...: put the shared-storage
# site-packages copy first (SW_PYSITE, see 3rdparty/README.md). torch / numpy stay the pod's own.
[ -n "${SW_PYSITE:-}" ] && export PYTHONPATH=$SW_PYSITE:${PYTHONPATH}
export GPU_MAX_HW_QUEUES=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/tc_sw16_${OMPI_COMM_WORLD_RANK:-0}
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MGD_STATS_EARLY=1
# 跨节点通信面（RCCL/TCPStore）网络绑定与日志
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

NODE_RANK=${NODE_RANK:-0}
NODE_BASE=$(( NODE_RANK * 8 ))
export RANK=$(( ${OMPI_COMM_WORLD_RANK:-0} + NODE_BASE ))
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=16
export MASTER_ADDR=${T23_MASTER_IP:?inner needs T23_MASTER_IP}
export MASTER_PORT=${DIST_PORT:-29911}

mkdir -p "${LOGDIR}"
exec > "${LOGDIR}/out_r${OMPI_COMM_WORLD_RANK:-0}.log" 2>&1

GBS=$((16 * ${MBS:-12}))

python -u $SW_ROOT/e2e/megatron/prelude.py \
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
    --micro-batch-size ${MBS:-12} \
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
    --log-throughput \
    --eval-iters 2 \
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --rank ${RANK} \
    --world-size ${WORLD_SIZE} \
    --local-rank ${LOCAL_RANK} \
    --dist-url tcp://${T23_MASTER_IP}:${DIST_PORT:-29911}
rc=$?
echo "EXIT $rc"
exit $rc   # rc 透传: 秒败不再被 echo 掩盖成 mpirun rc=0 (18:40 首跑教训)
