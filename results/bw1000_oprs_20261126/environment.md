# 环境：BW1000 上的 OPRS 移植（单卡阶段，2026-11-26）

本目录里的每个数字都产生于下面这一台机器。换机器、换卡数、换软件栈都必须重测：计划文件里带着平台块，
运行时加载器会拒绝与当前平台不符的计划（`src/stowaway/oprs/auto_plan.py`）。

OPRS = 先固定一个外部重叠后端 B，再对每个调用点预测"串行 / DEO / B"哪个更快，输出一份 JSON 计划，
由运行时派发器在 Megatron 或 vLLM 里照计划执行。DEO 是海光这边自己的重叠方法（NVIDIA 侧叫 DX）。

| 项 | 值 |
|---|---|
| 加速卡 | Hygon BW1000（DCU），`hy-smi` 设备名 `BW`，架构 gfx936，单卡显存 64 GB。**本阶段只有 1 张卡**（拓扑 BW1） |
| 主机 | 容器 `crdnotebook-2110675621851545601-liqsh2-74795`，内核 5.15.0-25-generic，128 核，503 GB 内存 |
| 软件栈 | DTK 26.04（`/opt/dtk-26.04`）、hip 6.3.26093、torch 2.9.0（HIP 版）、triton 3.3.0、Python 3.11.9 |
| 集合通信 | RCCL（`torch.distributed` 的 `nccl` 后端） |
| 对称堆 | DUSHMEM（环境变量 `DUSHMEM_SYMMETRIC_SIZE`），进程由 `mpirun` 启动，`pydushmem.host.mpi_init()` 建 MPI 世界 |
| Megatron | Megatron-LM-das @ `52b4450`（海光适配层）+ Megatron core `core_r0.18.0`（`571370c82`）。`3rdparty/Megatron-LM-das` 是只读链接，指向 `/root/private_data/lyc/Megatron-LM-das` |
| vLLM | `vllm-0.11.0+das.opt1.dtk2604.torch290`（site-packages 里的海光适配版） |
| Triton-distributed（DCU 移植，外部后端 B 之一） | `/root/private_data/tri-dist-hygon`，只读导入。工作树在 `40f48db`，另有 861 项未提交改动（研究树）：实际运行的是运行时工作树里的 `triton_dist/kernels/dcu/{all_gather_gemm,gemm_reduce_scatter,common_ops}.py` |
| Async-TP（外部后端 B 之二） | torch 2.9.0 自带的 `torch.ops.symm_mem` 融合算子（`torch.distributed._symmetric_memory`），无额外源码 |
| 仓库 | `/root/private_data/stowaway`，分支 `hygon`（自 `origin/nvidia` 的 `b63d7e0` 切出）加本次迁移的改动。本目录所有记录产生于迁移提交之前：记录里 `commit=b63d7e00f1…`、`src_dirty=True` |
| 训练载具 | 随机初始化的稠密 GPT 小车（`SW_MODEL=custom`：hidden 1024 / ffn 2048 / heads 16 / kv 4 / 4 层 / vocab 248320，seq 4096，MBS 1，每步 2 个微批）。分词器 `3rdparty/tokenizers/Qwen3.6-27B`，数据是 Megatron 的 mock 数据 |
| 推理载具 | `tools/oprs_tiny_model.py` 生成的 Llama 结构小模型（hidden 1024 / ffn 3584 / 8 头 / 4 层 / vocab 128256），没有权重文件：`load_format=dummy` 载入后再换成带种子的随机权重（`VS_SEEDED_WEIGHTS=1`）。分词器与自然文本来自 `3rdparty/models/LLAMA3-70B` |
| 本阶段没有用到的 | LLAMA3-70B 真实权重（140 GB bf16，装不进一张 64 GB 卡）；Qwen3.6-27B（海光 Megatron 没有接入该结构的载具） |

每条采样记录里的版本号由 `stowaway.oprs.platform.detect_versions()` 在采样的进程里现场读取
（`dtk 26.04 / hip 6.3.26093 / torch 2.9.0 / triton 3.3.0`），设备状态由 `hy-smi` 随点记录（`gpu_state_at_point`）。

端口约定（同一时间只能有一个 `mpirun`）：算子矩阵 29991，运行时冒烟 29987，位级检查 29989，Megatron 端到端 35000 起（每臂 +7），
vLLM 端到端 29880 起、自然文本检查 29950 起（每臂 +3）。
