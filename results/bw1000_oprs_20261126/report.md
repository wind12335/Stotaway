# 报告：BW1000 上的 OPRS 移植（单卡阶段，2026-11-26）

## 结论

1. **迁移完成，链路全通。** NVIDIA 侧的 OPRS 框架（按模型族留出的收益预测器、计划导出与加载、运行时派发器、
   Megatron 与 vLLM 适配器、海光自己的重叠方法 DEO，以及两个外部后端 Async-TP、Triton-distributed）已移植到海光栈
   （DUSHMEM 对称堆、torch-HIP、RCCL），代码在分支 `hygon`。交接文档要求的输出结构（`environment.md`、`status.md`、
   `raw/operators`、`raw/e2e_training`、`raw/e2e_inference`、`plans`、`tables`、本文件）已建立。
2. **单卡上每一段都实际跑过。** 算子级采样 36/36 条正常；三个候选在 Megatron 训练和 vLLM 推理里都按计划文件被派发；
   带随机权重的小模型上，vLLM 的每个重叠臂与原版输出逐位一致；96 项 CPU 单元测试通过。
3. **没有性能结论。** 一张卡上没有可重叠的通信，所有端到端时间只说明"跑通了、账算对了"。TP=8 下的算子表、计划、
   端到端各臂、真实权重的数值检查都**未测**，命令已备好（见 `docs/BW1000_OPRS移植说明_20261126.md` 第 4 节）。

## 已完成的（文件与口径）

测量口径只有两种：**算子级**（单独测一个投影算子，不在框架里）和**端到端**（整个训练步或整个预填充请求）。

| 内容 | 文件 | 口径与说明 |
|---|---|---|
| 单元测试 | `tests/`（在仓库里，不在这个目录） | CPU，不需要卡；覆盖预测器、计划导出与加载、评估、派发器的 CPU 可见部分、三个表格工具、计划集工具 |
| 运行时冒烟 | `raw/smoke/oprs_smoke_tp1_final_bf16.log`、`oprs_smoke_tp1_fp16.log` | 算子级，不计时：三个候选 × 两个方向，前向与梯度对解析解，bf16 与 fp16 全部 PASS；另记每次路由决策的宿主开销 |
| 位级对照 | `raw/smoke/oprs_bitwise_tp1.{log,jsonl}` | 算子级，不计时：见下面第 3 节 |
| 算子级采样 | `raw/operators/tp1_smoke_q4_m8192/samples_tp1_bfloat16.jsonl` | 新协议 b2b16：16 次连调成一组、8 组、组间轮换臂序；柱 = 串行中位 / 候选中位。4 个模型族、M=8192，36 条 |
| 计划的实际派发 | `raw/operators/tp1_plan_dispatch_{async_tp,triton_dist}/` | 算子级：计划的非串行站点经真实派发器各 132 次 |
| 汇总表 | `tables/operator_samples.csv` | 上面三批的 40 条（含 `plan_dispatch` 行），每条带状态；跑不了的点是带状态的行，不是 0 |
| 旧表回放评估 | `tables/predictor_choices.csv`、`predictor_summary.csv`、`predictor_family_holdout.json` | 标签 `historical_replay_legacy`：旧表 99 点，DEO 与串行二选一 |
| 计划集 | `plans/tp1_plumbing_{async_tp,triton_dist}/`、`plans/tp1_plumbing_vllm_{…}/` | 单卡通路用的计划：由上面的采样拟合，`note` 里写明不是测量。每个目录 5 个计划加 `manifest.json` |
| 站点形状表 | `tables/target_sites.csv` | 四个计划集的 80 条站点：预测得分、静态支持、是否在算子表上 |
| Megatron 端到端 | `raw/e2e_training/*/`（每个目录有 `summary.md`、`e2e_training.csv`），合并表 `tables/e2e_training.csv` | 端到端：每步耗时（自第 3 步起取中位）、loss、显存、各站点命中与回退 |
| vLLM 端到端 | `raw/e2e_inference/*/`（同上），合并表 `tables/e2e_inference.csv` | 端到端：ttft（1×M）、ttft2（1×2M）、thru（16×M）各遍耗时，实际预填 token 数 |
| vLLM 自然文本检查 | `raw/e2e_inference/*_gate_*/gatetext_*.json` | 端到端：每臂 4 个提示 × 48 个贪心 token 对原版，外加每个提示位置的 logprob |

## 三个值得记住的观察

### 1. 计划在实际派发路径上的预测与实测方向相反（单卡）

见 `status.md` 第 2 节的表：预测 1.007–1.223，实测 0.330–0.663。原因是这些站点的形状落在拟合样本的范围之外，
对数线性模型在那里是外推。单卡的实测值本身没有意义，但外推这件事与卡数无关：真实车辆的 qkv、proj 两个站点
天然不在算子表上。计划文件现在会标出越界的维度（`provenance.out_of_range`），8 卡实验要把它和端到端结果对照着看。

### 2. "与厂商 GEMM 逐位相同"取决于框架怎么存权重

`raw/smoke/oprs_bitwise_tp1.log`：同一个算子，`matmul(x, w.t())` 与 `matmul(x, w_kn)`（权重按 `[K, N]` 存）这两种写法
在约 0.02% 的元素上差一个 bf16 步长。DEO 和 Async-TP 与前一种逐位相同，Triton-distributed 与后一种逐位相同。
vLLM 的海光适配版把权重存成 `[K, N]`，所以它的三个重叠臂都与原版逐位一致；Megatron 存成 `[N, K]`，
所以 Triton-distributed 的 loss 与串行略有不同，DEO 与 Async-TP 相同。做数值一致性检查时要先弄清这一层，
否则"完全一致"或"略有差异"都会被误读。

### 3. 占位权重会让数值检查变成空结论

vLLM 的 `load_format=dummy` 在这个适配版里把所有参数（包括归一化层）填成 1e-3 量级，模型的下一 token 分布均匀到 1e-5，
任何臂的输出都一样。第一遍自然文本检查"全对"就是这个原因（目录 `…_gate_1126_2131/` 保留作证）。
现在单卡检查用 `VS_SEEDED_WEIGHTS=1` 换成正常量级的随机权重。8 卡上用真实权重，不涉及这个问题。

## 待完成（换到 8 卡之后）

按 `docs/BW1000_OPRS移植说明_20261126.md` 第 4 节的顺序：

1. 算子表：首批 21 个形状，再补到 99 点；负载站点形状各测一批；两个旧异常点重测到新目录。
2. 新协议下的留族验证（`tables/predictor_choices.csv` 里出现 `family_holdout_b2b16` 的行），再生成 bw27b、llama70 的计划集。
3. 计划的实际派发路径复测（8 卡），与离线预测对照。
4. Megatron 端到端五个臂两轮；vLLM 端到端五个臂两轮加自然文本检查（Llama3-70B 真实权重）。
5. 消融：只用岭回归对加上残差修正；预测计划对事后最优；生成与加载的开销。

## 没有纳入的

- FlashOverlap、Syncopate（按要求对照只有串行、Async-TP、Triton-distributed）。
- 真实 Qwen3.6-27B 的训练载具（未接入；`SW_MODEL=27b` 是稠密 GPT 形状车辆）。
- 双机实验。
- 旧预测器的遗留结果与未提交改动（不属于这次迁移）。

本目录由迁移会话产生，提交前的 `commit` 字段都是 `b63d7e0`、`src_dirty=True`。
