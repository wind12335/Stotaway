# Stowaway NVIDIA 移植原型

此目录是 `/root/Stotaway`（提交 `cf70c6f5f84ca823d1d8c6e2bcae7e922ae37158`）的独立 NVIDIA 迁移原型，不修改原仓库。它借鉴原项目 DX 的分块直接交换和 ticket/ACK 思路，不是完整移植或性能结果。

## 当前范围

- 单机、每进程一张 GPU、双卡：NVSHMEM Python API 初始化对称内存并映射到本机 peer；初始化方式参考 Triton-Distributed，但原型不再强制导入其 Triton 编译器扩展。
- CUDA Driver 流内存等待与 Runtime 异步拷贝；数据及 ticket 在同一流上依次复制，Triton system-scope release atomic 发布 ACK。
- GEMM+ReduceScatter、AllGather+GEMM 前向及 AG+dgrad 的 BF16/FP16 原型；GEMM 通过 `torch.matmul` 调用 PyTorch 后端。
- 独立 ReduceScatter 双缓冲原型，以及 Dense 行/列并行的一阶自动求导实验封装；行并行 backward 可选 AG+dgrad 原型，列并行 backward 使用 NCCL。
- 连续调用并检查结果的双进程测试，含两个精度、多个形状、非默认调用流及不对称设备延迟。
- PyTorch 2.7 对称内存融合算子及 Inductor micro-pipeline TP 图改写的独立双卡检查；尚未接入 Megatron 或规划器。

不支持：多机、非双卡、二阶自动求导、生产 Megatron/vLLM/MoE、CUDA Graph、动态形状重用和通用规划器。已有诊断性双卡查表预测器与受限 E2E 性能数据，不能视为完整生产迁移或通用性能保证；最新结果见 `results/a800/双卡E2E实验_20261110.md`。独立 RS 支持输出别名于本 rank 输入分片和 `k_slots` 多 lane；`chunk_ev` 非空时拒绝，因为原 DCU 的 `dxv_op` 虽有该参数但未使用它，不能把它当作实现了 chunk 就绪语义。任何尚未验证的行为都不能用于生产或论文结论。

`planner/platform.py` 仅提供形状检查和未标定时的原版回退，**不是已完成的 A800 规划器**；显式传入的配置如标注其他平台或 TP 不一致会被拒绝，合法配置也不会在尚未实现校准和验证时启用 DX。

`planner/gemm_probe.py --rows 8192 --tp 2 --output planner/data_a800/gemm_probe_tp2_m8192.json` 可采集单卡 GEMM 原始样本；输出路径要求不存在，避免误覆盖。已有 M=8192/16384/32768 的 TP2 和 TP8 载具维度单卡结果；这些文件不包含集合通信或规划器拟合数据。

`bench/tp2_cells.py`、`planner/extract_tp2_cells.py` 和 `planner/evaluate_tp2_costmodel.py` 用于采集并离线评估 TP2 原型代价数据。当前留一载具预测 19/32 命中事后最优，样本小且未进行完整实验设计；探索性配置不与 `planner/platform.py` 的部署回退对接。

## 环境与运行

需要 NVIDIA CUDA、PyTorch NCCL、Triton-Distributed、nvshmem4py 和 NVSHMEM。当前机器的 NVSHMEM 动态库不在默认搜索路径。使用脚本自动发现 pip 包路径，或提前设置 `NVSHMEM_HOME`：

```bash
cd /root/Stowaway-NVIDIA
bash run_tests.sh
bash run_tests.sh tests/test_protocol.py
bash run_tests.sh tests/test_dgrad.py
bash run_tests.sh tests/test_reduce_scatter.py
bash run_tests.sh tests/test_autograd.py
bash run_tests.sh tests/test_reuse.py
bash run_tests.sh bench/operator_smoke.py
bash run_tests.sh experiments/async_tp_smoke.py
bash run_tests.sh experiments/async_tp_compile_smoke.py
```

运行会占用两张 GPU，失败时 `timeout` 限制挂起时间。默认对称堆每进程 256 MiB；需要更大测试时可设置 `NVSHMEM_SYMMETRIC_SIZE`。`TEST_ROUNDS` 控制连续调用次数，默认 32。实际使用时必须所有 rank 以相同顺序创建、调用和关闭 context；使用者保证输入在调用前后不被其他流修改。每个 context 只能在同一调用流上串行使用，不支持多线程并发调用。

已观察到 NVSHMEM4Py 的 `get_peer_tensor` 在释放并复用不同大小的分配时可能返回旧大小的映射。本原型保留本地对称 tensor 的所有权，通过 `nvshmem_ptr` 获取直接可访问的 peer 指针；若不可访问则拒绝使用。该路径只针对同机 P2P，不是跨节点传输。

来源：设计基于上游 Stowaway 仓库，原项目 MIT 许可和 Triton-Distributed 的归属说明见 `/root/Stotaway/LICENSE` 与 `/root/Stotaway/NOTICE`。

BW1000 侧建议和本机阶段性检查分别见 `A800迁移注意.md`、`results/a800/ENV.md`。目前代码是独立原型，而非原仓库生产 DX 的逐行平台垫片；在完成生产路径差异核对、独立 RS 和原 benchmark 门之前，不应把它作为冻结机制的最终移植。

`experiments/run_megatron_smoke.sh` 是双卡小模型诊断载具，`STOWAWAY_DX=0` 为参考、`STOWAWAY_DX=1` 为 DX，`STOWAWAY_RECOMPUTE=1` 打开全重计算，`STOWAWAY_ITERS` 指定训练步数。默认禁用 Torch 编译并使用实验性 TorchNorm/SP 兼容层；设置 `TORCH_COMPILE_DISABLE=0 STOWAWAY_DIAGNOSTIC_TORCH_NORM=0 STOWAWAY_TE_LOCAL_NORM=1` 可尝试本地 linear + TE TENorm 组合。当前隔离 TE 环境的 wheel 元数据仍有兼容性警告，不能作为正式论文基线。该适配器只支持 world=TP=2、单机、指定接口和无偏置路径，按形状惰性集体创建 context；只有所有 rank 执行相同路径和形状时才能安全使用。若需比较每步参数，分别为两次运行设置不同 `STOWAWAY_CAPTURE_DIR`，之后调用 `experiments/compare_captures.py`。采集会同步并写磁盘，不能计时。

`STOWAWAY_BRS=1` 可试验 Megatron TP 的 `layers` / `mappings` reduce-scatter 接入，并可与 `STOWAWAY_DX=1` 组合；组合时 column DX 的 backward 也使用所选择的 RS 接口。`STOWAWAY_BRS_MIN_BYTES` 设置输入字节阈值，`STOWAWAY_BRS_SLOTS` 设置 lane 数。该 TP 诊断适配器不会替换 distributed optimizer 的 bucket collective。它按形状惰性集体分配，要求双 rank 以相同顺序走相同形状与分支；不支持多 TP 组、CUDA Graph 或并发线程。

`STOWAWAY_DX=1 STOWAWAY_TP2_LOOKUP=$PWD/planner/data_a800/tp2_forward_lookup_20261110.json` 可试用双卡逐调用点的保守选择：只匹配已测的完整形状，未知形状回退 stock，按 CUDA/torch/Triton/GPU/连接配置拒绝不匹配的 profile，双 rank 检查 profile 哈希一致。该表使用前向算子计时，尚未建模训练反向、TE 或 Async-TP，属于诊断原型；采集和独立回归见 `results/a800/双卡预测器适配_20261110.md`。

训练诊断可改用 `planner/data_a800/tp2_training_lookup_20261110.json`；其标定包括简化的前向与反向，但尚未包含完整 Megatron layer/optimizer 开销，也不是生产调度器。对应采样及验证见 `results/a800/双卡预测器适配_20261110.md`。

额外设置 `STOWAWAY_TP2_INTERPOLATE=1` 可试验同调用点、相同 K/N、两个已校准 M 之间的保守插值；只在两侧均选 DX 时选 DX，范围外回退 stock。它默认关闭，当前只做了有限的未见 M 检查。

`stowaway_nvidia/moe_grouped.py` 提供正确性优先的 Triton 专家线性计算原型，直接在设备读取专家累计偏移，支持 FP16/BF16 的前向和一阶反向；空专家和空行均有定义。调用方必须保证偏移单调、非负、不超过输入行数，内核不做设备到主机同步来验证它们。固定网格计算开销较大，尚未调优，也未实现原项目的跨卡 MoE dispatch、Megatron 接入或规划器。

`stowaway_nvidia/moe_routing.py` 提供设备侧固定大小的专家计数和前缀和；只产生偏移，不负责排序或分发。IDs 超出范围会作为 padding 忽略。当前限制最多 1024 个专家和 int32 计数。

`stowaway_nvidia/moe_local.py` 将本机 top-k 路由、分组专家线性、SwiGLU 和加权合并串联起来，用于单 GPU 数值诊断；调用方必须提供有效专家 ID。它未实现跨卡 EP，也未接入 Megatron 或性能调优。

`stowaway_nvidia/moe_exchange.py` 是独立的单机 EP2 前向通信原型，用 NVSHMEM 对称堆和 CUDA stream wait 交换数据，固定预留最坏路由容量；不支持反向、动态 token 数、并发或无效 expert ID，也未接入 Megatron。每 rank 必须按相同顺序发起并结束调用。

`stowaway_nvidia/moe_ep.py` 在上述通信原型上增加有限的训练反向：单机 EP2、BF16、固定形状、每次前向必须完成一次反向后才能复用 context，所有输入和本地专家权重必须参与梯度计算。多个前向尚未反向的调度、重计算、重复 backward 和多线程不支持；所有 rank 必须以相同顺序进入正反向。

`STOWAWAY_MOE=1 STOWAWAY_MOE_GG=1 STOWAWAY_TP=1` 可运行小型 Megatron EP2 的专家 GEMM 诊断适配器，仍使用上游 dispatcher。当前 dispatcher 传入 CPU 计数，适配器再复制回设备；每次调用还会堆叠复制各专家权重，因此不能用于声称完整路径无 host 同步或报告性能。只有满足无偏置、SwiGLU 等条件的 `SequentialMLP` 命中，否则回退上游。

`STOWAWAY_MOE=1 STOWAWAY_MOE_EP=1 STOWAWAY_TP=1` 可诊断完整 Megatron MoE 层的 NVSHMEM EP2 前向/反向，限制无重计算、无共享专家、固定大小和单次反向等条件，其他配置回退上游。该路径还未覆盖真实模型和不同微批调度，不能用于性能结论；它与其他诊断适配器必须分开运行。

Megatron EP 的默认乘权顺序已对齐上游：SwiGLU 后、FC2 前乘 router 概率，当前需要可微 NCCL all-gather 获取概率；`STOWAWAY_MOE_PRE_FC2_SCALE=0` 仅用于复现旧的输出端乘权实验。两种有限模型形状的两步参数/梯度与多轮时延对照见 `results/a800/双卡E2E实验_20261110.md`。这不是对长训练、任意路由或多卡规模的验收；通用 `ep_moe` 默认仍采用输出端乘权，调用方可显式传 `scale_before_fc2=True`。`STOWAWAY_MOE_REFERENCE_GEMM=1` 是 CPU 偏移读取的诊断路径，不可用于性能基准。

vLLM Qwen3 诊断设置 `VS_ARCH=qwen3`：Qwen3-14B 真权重在 TP2、M512/1024 的少量提示上已与原版逐 token 对照通过；配置和局限同 Llama 的默认 NVSHMEM 行通信路径。

`STOWAWAY_DX_SITES` 可按 `qkv,proj,fc1,fc2` 指定启用 DX 的调用点；`STOWAWAY_COLUMN_MIN_N`、`STOWAWAY_COLUMN_MAX_N` 或逗号分隔的 `STOWAWAY_COLUMN_WIDTHS` 可进一步限定 column 输出宽度。这些均为诊断选择，不是规划器；必须所有 rank 使用同一配置。

`STOWAWAY_OPT_BRS=1 STOWAWAY_TP=1 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 STOWAWAY_GRAD_REDUCE_BF16=1` 可单独诊断两卡 DP 优化器 bucket；该适配器限制 world=DP=2，插入 NCCL marker，不能用于性能或混合 TP/DP 结论，也不能与其他诊断适配器同时启用。

完整 TE 层诊断可设 `STOWAWAY_TRANSFORMER_IMPL=transformer_engine STOWAWAY_TE_OVERLAP=1 STOWAWAY_DIAGNOSTIC_TORCH_NORM=0 STOWAWAY_TE_LOCAL_NORM=0 UB_SKIPMC=1`；`STOWAWAY_TE_INSPECT=1` 打印每个 TP 层的 userbuffer 重叠配置。此环境的 TE wheel 平台元数据仍有警告，不能直接当正式基线。

`stowaway_nvidia/vllm_sp.py` 是 vLLM 0.11 Llama 适配器的未完成原型，仅双卡、无量化无偏置、eager、单 PP，指定 prefill token bucket，其余形状回退上游。真实 Llama-3-8B 试验中，分块行投影和部分尺寸的分块列投影都出现 token 分歧，不能视为通过严格生成门；默认列投影使用原有 NCCL 路径，行投影使用整矩阵 GEMM + NVSHMEM RS，仅在有限样本通过对照，不重叠行 GEMM 与通信。`VS_DX_COMPONENTS=col,row`、`VS_ROW_EXACT=0` 可显式启用未经严格门的重叠路径；这些开关不是自动规划器。更多细节见 `results/a800/ENV.md`。

这些测试只能增加对当前设备和形状的信心，不能证明完整 CUDA 内存模型、跨节点行为或所有调度条件下无死锁。未实现的阶段见原仓库分析：跨进程事件/内存模型扩展、性能调优、框架及规划器适配。
