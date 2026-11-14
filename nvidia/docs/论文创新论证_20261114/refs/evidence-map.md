# 证据地图

范围：2026-11-14 工作区，本地论文全文、NVIDIA 代码及已保存结果；不是穷尽文献检索。模型 E2E 指既定载具的完整运行，具体训练与推理配置见 E6。相关资料以所核查版本为准。

| ID | 来源 | 类型与可核实事实 | 支持的说法 | 风险 |
|---|---|---|---|---|
| R1 | [Flux](https://arxiv.org/abs/2406.06858)；本地 `research/flux.txt` | 论文，GEMM 与通信细粒度融合，讨论 Megatron 与 vLLM | 属于算子融合重叠类相关工作 | 没有相同配置的本次八卡 Flux 比较；不能断言其没有系统级规划 |
| R2 | [FlashOverlap](https://arxiv.org/abs/2504.19519)；`docs/research_sources/flashoverlap.txt`，`/root/FlashOverlap/tune/search.py` | 论文与源码，通过 signaling/reordering 和 predictive search 选分组 | 预测分组和流水线 max 递推均有先例 | 仅对比已阅读内容，不宣称其无法整模型选择 |
| R3 | [Triton-distributed](https://arxiv.org/abs/2504.19442)；`docs/research_sources/triton-distributed.txt` | 论文，扩展 Triton 以表达分布式通信和重叠 | 分布式 kernel 编程与优化平台 | 本次 A800 路径需与论文硬件/实现区分 |
| R4 | [Syncopate](https://arxiv.org/abs/2601.20595)；`docs/research_sources/syncopate.txt` | 论文，chunk 抽象、Triton 源码变换、自动调参，可组合外部通信计划 | 强相关的生成与规划工作 | A800 实验为适配路径，非默认等同论文最强实现 |
| R5 | [FiCCO](https://arxiv.org/abs/2512.10236)；`research/ficco.txt` | 论文，DMA 细粒度重叠并据计算/通信特征选择 schedule | 特征驱动重叠选择、标准 GEMM+DMA 均有先例 | 论文的 81% 评估任务不同，不能直接横向对比 |
| R6 | [TokenWeave](https://arxiv.org/abs/2505.11329)；`research/tokenweave.txt` | 论文，推理 token 分块、wave-aware 调度等 | 推理重叠及 wave 调度并非空白 | 平台/路径不同，未做同环境实测 |
| R7 | [Centauri DOI](https://doi.org/10.1145/3620666.3651379)；`docs/相关工作撞车审查_20261112.md` | 依据 FlashOverlap 的二手引用：面向训练做通信分区与分层调度 | 有必要进一步核查其规划目标与实现 | 未获取全文，不能以二手材料断言差异或不撞车 |
| E1 | `planner/costmodel.py`（原 BW1000） | 代码，G/通信成本与分位数保守门 | 原版已经有预测和回退，增量是候选/路由/执行的拓展 | 不能称本工作首次引入预测 |
| E2 | `planner/original_a800/orthogonal_predictor.py`、`route_local_residual.py` | route 类型、log(M,N,K) 岭回归、按模型族 OOF 残差、邻居加权修正 | 实际评估算法与路由隔离 | 岭回归及最近邻不是新的机器学习基础算法；route 元数据不等于全部被运行时强制校验 |
| E3 | `planner/original_a800/export_tp8_auto_plan.py`、`experiments/build_tp8_{sync,flash}_model_plans_20261114.py` | 固定一个外部方法；shape/方向每点最大预测收益；JSON 导出 | 具体 B/DX/stock 选择机制 | Sync/Flash 模型计划用 FP16 外部数据估计 BF16，非同类型严格模型 |
| E4 | `stowaway_nvidia/multi_backend8.py`、`auto_plan8.py`、`external_tp8.py` | 跨 rank digest 检查、exact shape lookup、未知 shape stock fallback、后端实际调用与布局转换 | 真实部署与布局接入 | 未自动解析任意库、未实现端到端全局联合搜索 |
| E5 | `stowaway_nvidia/dx8.py` | peer 分块、厂商 matmul、CUDA copy 与 stream wait、NVSHMEM 对称缓冲和 Triton 归约/信号 | DX 具体执行机制 | 不应写成纯 Triton GEMM 或全部通信是 NVSHMEM put；单个机制有先例 |
| E6 | `results/单机8卡端到端与预测器结果_20261113.md` 与其原始日志/JSON | 四后端 Qwen 训练及 Llama 推理真实计划和耗时 | 系统闭环与具体观察到的模型结果 | 各方案独立一轮；Qwen 随机初始化/mock、Llama 真实权重；Sync/Flash 自动计划只用 stock/DX |
| E7 | `results/单机8卡多后端算子结果_20261113.md`、`results/八卡DX算子重测对照_20261114.md` | 99/99 DX BF16 重测；Flash 33/33 RS 含 23 原生+10 派发计时 | 覆盖和方向互补现象 | 跨批、dtype 和布局口径不同；DX 展示择高有选择偏差 |
| E8 | `/root/data/stowaway-nvidia/results/tp8_{auto_threeway,td_dx_threeway}_lomo_20261113.json`、`tp8_{syncopate,flashoverlap}_dx_fp16_lomo_20261114.json` | 每点选路、近似 oracle 和实际归一化值 | 96/99、85/99、89/99、27/33；按模型留出开发评估 | 未见模型并不等于完全陌生的形状族；数据已参与开发，标签跨批；Flash 口径混合 |
| E9 | `experiments/tp8_auto_{td,flash,sync}_*20261114.json` 与 `20261113.json`，`results/单机8卡端到端与预测器结果_20261113.md` | 本地计划和实际命中，TD/Async 有 B+DX 混合，Sync/Flash 为 stock+DX | 区分通用候选机制和具体被选组合 | 部分文件命名不同，以实际文件和日志为准 |
