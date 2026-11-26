# 状态：每个后端能不能跑、覆盖到哪里、哪些失败了（单卡阶段，2026-11-26）

一句话：OPRS 的整条链路（采样 → 评估 → 生成计划 → Megatron / vLLM 里真实派发 → 汇总表）在**一张卡**上全部打通，
三种重叠方法都能按计划文件在两个框架里被派发；**所有多卡（TP≥2）的路径都还没有在新协议下测过**，
本目录里没有任何可以用作性能结论的数字。

术语：DEO = 海光自己的重叠方法（NVIDIA 侧叫 DX）；Async-TP、Triton-distributed = 两个外部重叠后端（交接文档里的"B"）；
"串行" = 原来的路径（先通信、再做 GEMM，或反过来），是所有比值的分母。

## 1. 后端可用性与覆盖

| 后端 | 接入方式 | 单卡（TP=1）实测 | TP=8 | 覆盖 |
|---|---|---|---|---|
| 串行（参照） | Megatron 原生前向 / vLLM 原生前向 | 通（两个框架） | **未测**（新协议） | 全部 |
| DEO | `src/stowaway/oprs/backends/deo.py`，走 DUSHMEM 引擎世界 | 通：算子级 12/12 点正确；Megatron 四个站点全部命中；vLLM 四个站点全部命中 | **未测** | 按静态形状规则：算子表 99/99 点；bw27b、bw70b、llama70 车辆都是 4/4 站点 |
| Async-TP | `torch.ops.symm_mem` 融合算子（torch 2.9.0 自带） | 通：算子级 12/12 点正确；Megatron、vLLM 四个站点全部命中 | **未测** | 算子表 99/99 点；三个车辆 4/4 站点 |
| Triton-distributed | `backends/triton_dist.py` → `/root/private_data/tri-dist-hygon` 的 DCU 移植（只读） | 通：算子级 12/12 点正确；Megatron、vLLM 全部站点命中（vLLM 里每个 AG 权重做一份 K 连续拷贝） | **未测** | 受内核硬约束限制（AG：M、N 是 128 的倍数、K 是 16 的倍数、M/TP 是 1024 的倍数；RS：M、N、K 都是 256 的倍数）：算子表 87/99 点——gemma3-27b 的 rs（K=2688）和 qwen25-72b 的全部 9 点跑不了；首批 21 个形状里 18 个；bw27b 只有 2/4 站点（proj 的 K=640、fc2 的 K=2176），bw70b 与 llama70 是 4/4 |

来源与许可证：Async-TP 是 PyTorch 2.9.0 自带的算子（BSD-3），不需要额外编译；Triton-distributed 用的是本机已有的 DCU 移植
（/root/private_data/tri-dist-hygon，只读导入，不编译任何东西），其内核层源自 ByteDance triton-distributed（MIT，见仓库的
`NOTICE` 与 `LICENSE.triton-dist`）。DEO 调的是海光栈上已有的内核（`src/stowaway/kernels`）。三个后端都只支持 16 位 dtype
（`bfloat16`、`float16`）；bf16 与 fp16 在单卡冒烟里都通过（TP=8 两个 dtype 都未测），其余记录都是 bf16。

"静态支持"只说明形状满足内核的硬约束；一条路径到底接通了没有，只认有正确性记录的实测（`stowaway.oprs.data.usable`）。
| FlashOverlap、Syncopate | — | 按要求不纳入（对照只有串行、Async-TP、Triton-distributed） | — | — |

关于 Async-TP 的一条更正：此前记为"BW 上不可用"。本次实测它在 DTK 26.04 / torch 2.9.0 上**单卡**可用
（`torch.ops.symm_mem`，bf16），算子级、Megatron、vLLM 三条路都通过。多卡仍未测，"可用"的结论只到单卡为止。

Qwen3.6-27B：`stowaway.oprs.workloads.qwen36_sites` 移植了它的静态站点形状（与 NVIDIA 侧逐项一致，见单元测试），
但海光的 Megatron 没有这个结构的载具，**未接入**，不能派发。`SW_MODEL=27b` 的车辆是一个稠密 GPT 形状
（hidden 5120 / ffn 17408），不是 Qwen3.6-27B。

## 2. 做过的事与结果（都是通路检查，不是性能测量）

| 事 | 位置 | 结果 |
|---|---|---|
| 单元测试（CPU） | `tests/`，`python3 -m pytest tests -q` | 96 项通过 |
| 运行时冒烟 | `raw/smoke/oprs_smoke_tp1_final_bf16.log`、`oprs_smoke_tp1_fp16.log` | 算子级，不计时：DEO、Async-TP、Triton-distributed × AG、RS，前向、对输入的梯度、对权重的梯度与解析解一致，关闭后堆可复用。bf16 与 fp16 都是全部 PASS |
| 路由决策的宿主开销 | 同上（`host overhead per call` 行） | 派发器每次决定走哪条路径花 1.5–1.7 µs（命中与未命中相同），两个 dtype 一致 |
| 位级对照 | `raw/smoke/oprs_bitwise_tp1.{log,jsonl}` | DEO 与 Async-TP 和 `matmul(x, w.t())` 逐位相同；Triton-distributed 和 `matmul(x, w_kn)` 逐位相同；这两种厂商 GEMM 写法本身在约 0.02% 的元素上差一个 bf16 步长 |
| 算子级采样（新协议 b2b16） | `raw/operators/tp1_smoke_q4_m8192/` | 4 个模型族 × 3 个算子 × M=8192 × 3 个候选 = 36 条，全部 `ok`，正确性全部通过 |
| 计划的实际派发路径 | `raw/operators/tp1_plan_dispatch_*` | 每个计划里非串行的站点经真实派发器各跑 132 次，有耗时、有命中数、有正确性 |
| 旧表的回放评估 | `tables/predictor_*.csv` | 标签 `historical_replay_legacy`：预测器选对 84/99（去掉两个旧异常点后 81/97）；固定用 DEO 73/99；只用岭回归 79/99。这是把旧表喂给移植后的预测器，**不是**新协议下的留族验证 |
| Megatron 端到端 | `raw/e2e_training/tp1_smoke_oprs_async_tp_v4/`、`tp1_smoke_oprs_triton_dist/`，以及用最终代码复核的 `tp1_smoke_final_check/` | 小车 4 层、8 步，退出码都是 0（`_v4` 那一批 mpirun 自身返回 1，见第 3 节；后面的批次驱动已记下 `py_rc`）。Null 臂 0 命中、全部回退（原因 `plan`）；OPRS 臂只命中计划选中的站点；Always 臂四个站点全命中。triton_dist 那一批每个站点 160 次（训练 128 + 训练结束后的评估 32），`_v4` 那一批早于增量统计，记的是 128。loss：DEO、Async-TP 各臂打印出的 7 位数字与串行完全相同（1.182455E+01）；Triton-distributed 臂是 1.182467 / 1.182465，从第 5 位有效数字起不同（它用的是自己的 triton GEMM，见位级对照） |
| vLLM 端到端 | `raw/e2e_inference/tp1_plumbing_vllm_triton_dist_tp1_M2048_1126_2157/` | 两轮各 5 个臂，全部 `ok`；每个臂实际预填 2048 / 4096 / 32768 个 token |
| vLLM 自然文本检查 | `…_gate_1126_2144/`、`…_async_tp_…_gate_1126_2207/` | 带种子的随机权重小模型：每个臂 4 个提示 × 48 个贪心 token 与原版完全相同，8188 个提示位置上的 logprob 差为 0（与位级对照的结论一致） |

### 开销记录（单卡；8 卡要重记）

| 环节 | 数值 | 出处 |
|---|---|---|
| 离线拟合 | 0.01–0.28 秒（36 条样本，按方法 × 方向各拟合一个模型） | 各计划集 `manifest.json` 的 `overheads.fit_s` |
| 生成计划 JSON | 1–5 毫秒 | 同上，`export_s` |
| 运行时读计划、各 rank 核对摘要 | Megatron 123–147 毫秒；vLLM 180–197 毫秒 | 日志里的 `OPRS_READY … load_ms` |
| 建立上下文（分配对称堆、初始化后端） | Async-TP 3–19 毫秒；DEO 17–19 毫秒；Triton-distributed 26–80 毫秒（按站点数 1–4 个） | `OPRS_READY … prepare_ms` |
| 每次路由决策 | 1.5–1.7 微秒 | 冒烟日志 |

读计划的那一百多毫秒里包含一次跨 rank 的摘要核对和平台版本读取，每次运行只付一次（不是每个调用点付）。
8 卡上这些数都会变，尤其是 Triton-distributed 的上下文（vLLM 里每个 AG 权重一次 K 连续拷贝）。

### 计划的实际派发路径上，预测与实测的对照（单卡，TP=1）

| 计划 | 站点 | 选择 | 计划预测的加速 | 实测加速 | 命中次数 | 正确性（相对误差） |
|---|---|---|---|---|---|---|
| `oprs_async_tp.json` | fc2 | DEO | 1.007 | 0.663 | 132 | 0 |
| `oprs_triton_dist.json` | qkv | Triton-distributed | 1.076 | 0.330 | 132 | 9.8e-4 |
| `oprs_triton_dist.json` | fc1 | Triton-distributed | 1.223 | 0.411 | 132 | 2.0e-3 |
| `oprs_triton_dist.json` | fc2 | DEO | 1.007 | 0.659 | 132 | 0 |

读法：预测与实测方向相反。单卡上本来就没有可重叠的通信，实测小于 1 不奇怪；值得注意的是预测为什么会大于 1——
这些计划是用"M=8192、形状更大"的 4 族样本拟合的，小车站点的形状落在样本范围之外（N、K 都更小），
对数线性模型在那里是外推。这说明：**调用点的形状落在拟合样本范围之外时，预测值不能当真**。
为此计划文件现在会记录越界维度（`provenance.out_of_range`，`tables/target_sites.csv` 的 `out_of_range` 列）。
真实车辆里 qkv、proj 两个站点天然离表，8 卡实验要把这一列和端到端结果对照着看。

## 3. 失败与中止的启动（都保留在原目录，没有删除也没有覆盖）

| 目录 | 发生了什么 | 处理 |
|---|---|---|
| `raw/e2e_training/tp1_smoke_stock_custom/` | 驱动脚本没有 export `TP`，内层按 8 卡启动，Megatron 断言 `world size (1) is not divisible by total_model_size (8)` | 已修：`export TP` |
| 同上（第一次）与 `…_oprs_async_tp/` | 内层脚本用 `echo` 盖掉了 python 的退出码（日志里 `EXIT 1`，驱动记 rc=0） | 已修：`exit $rc`；另记 `py_rc` |
| `raw/e2e_training/tp1_smoke_oprs_async_tp/` | 试过"在 TP=1 强制开序列并行"的单卡开关，被 Megatron 的模型配置拒绝（`Cannot use sequence parallelism without tensor parallelism`） | 放弃该开关。TP=1 时序列并行布局与普通布局是同一组张量，派发条件改成"序列并行形态，或 TP=1" |
| `raw/e2e_training/tp1_smoke_oprs_async_tp_v2/` | Null 臂：`invalid plan entry`——计划条目里多了一个 `origin` 字段，加载器的字段集是严格的 | 已修：`origin` 移到 `provenance.origins` |
| `raw/e2e_training/tp1_smoke_oprs_async_tp_v3/` | 两个问题：默认允许的方法里漏了 DEO（`plan needs ['deo'] but backends enabled here are ['async_tp','triton_dist']`）；Null 计划没有带上来源计划的后端（`invalid plan allocation`） | 已修：选项改名 `methods`，默认是全部已接入方法；Null 计划继承来源计划的后端。这一批随后被我手动终止 |
| `raw/e2e_training/tp1_smoke_oprs_async_tp_v4/` | 完整跑完 7 个臂。两个遗留：(a) mpirun 自身的返回值是 1（进程不做 `MPI_Finalize` 就离开，prterun 记为异常终止），真实退出码要看日志里的 `EXIT n`；(b) 命中数只统计到 `train()` 返回，训练结束后的评估没有计入 | (a) 驱动与表格改读 `EXIT n`/`py_rc`；(b) 退出时补打增量行（`after_report`），下一批 `tp1_smoke_oprs_triton_dist` 已包含 |
| `raw/e2e_inference/tp1_plumbing_vllm_async_tp_tp1_M2048_1126_2114/` | `sp_rccl`、`all_deo` 两个臂启动即退：臂名到计划文件名的映射写错（找了 `sp_rccl.json`） | 已修：映射集中在 `stowaway.oprs.arms` |
| `raw/e2e_inference/…_1126_2118/` | 跑完，但 `always_triton_dist` 的第一个 ttft 是 1.04 s、第二个 0.012 s：换了一条路径之后的首次使用有一次性开销 | 已修：每个计时格之前先做一次不记录的预热（`…_1126_2157/` 是修后的） |
| `raw/e2e_inference/…_gate_1126_2131/` | 自然文本检查"全对"，但其实是空结论：`load_format=dummy` 把所有参数（包括归一化）填成 1e-3 量级，模型的下一 token 分布均匀到 1e-5，每个臂输出都一样 | 已修：`VS_SEEDED_WEIGHTS=1` 换成正常量级的随机权重，分布不再平坦（提示 logprob 在 −36 到 −4 之间） |
| `raw/smoke/oprs_smoke_tp1_run1_rc1_no_finalize.log` | 冒烟全部 PASS 但 mpirun 返回 1（同 v4 的 (a)） | 已修：冒烟结束前 `MPI_Finalize`；`oprs_smoke_tp1.log` 是修后的 |

## 4. 没有做的（都是"未测"，不是 0）

- TP=8 下的一切：算子表（21 个首批形状、99 点全表）、负载站点形状的独立测量、新协议下的留族验证、bw27b / llama70 的计划、
  Megatron 与 vLLM 的端到端各臂、Llama3-70B 真实权重上的自然文本检查。
- 真实 Qwen3.6-27B 的训练载具（未接入）。
- 双机（按要求不启动）。
- 旧预测器的遗留没有动：RS 截止的拟合、`results/raw/vllm/vllm_attr_1112/` 的归因结果、`e2e/vllm/vllm_sp.py` 与 `tools/vllm_report.py` 里
  未提交的单站点臂改动。它们不属于这次迁移，也没有混进本次提交。
