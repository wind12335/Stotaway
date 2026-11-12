# A800 移植环境与阶段记录

记录时间：2026-11-09，系统报告时区 UTC+08:00。以下为当次观察；共享资源和软件环境可能变化。

| 项目 | 版本或状态 |
| --- | --- |
| GPU | 2 × NVIDIA A800-SXM4-80GB，GPU 间 `NV8` |
| Driver | 580.65.06 |
| CUDA Toolkit / PyTorch 构建 CUDA | 12.9.86 / 12.6 |
| Python / PyTorch / NCCL | 3.12.9 / 2.7.0+cu126 / 2.26.2 |
| Triton / Triton-Distributed | Triton 3.4.0；`/root/Triton-distributed` 提交 `552fedcb04d0caf909f6ac165594d2afbe301b88` |
| NVSHMEM / nvshmem4py | 3.3.9 / 0.1.2，动态库路径由 `run_tests.sh` 设置 |
| 原 DCU 仓库 | `/root/Stotaway`，`cf70c6f5f84ca823d1d8c6e2bcae7e922ae37158`，master；原工作区未修改 |
| 训练 / vLLM | 已浅克隆上游 Megatron-LM `core_v0.18.2`（`571370c829ca768fe37244f4e2e7f28d8accc4ab`）和 vLLM `v0.11.0`（`b8b302cde434df8c9289a2b465406b47ebab1c2d`）至 `/root/data/stowaway-nvidia/third_party/`；均未安装或验证运行 |
| 存储 | `/root/data` 是 256 GiB 可写 Lustre 挂载；`/root/eb-public` 是只读共享模型挂载 |

版本选择：Megatron 提交与 BW1000 原载具核心版本一致，其元数据要求 Python ≥3.12 和 PyTorch ≥2.6，当前 Python 3.12.9 / torch 2.7.0 满足这两条元数据约束，不等于其他依赖或训练已验证。vLLM 0.11.0 与原 DCU vLLM 同大版本，源码中 `LlamaModel` 的相关字段与 import 名称初步匹配，但原 patch 运行兼容性未测；其 CUDA requirements 固定 torch 2.8.0，因此不要安装到依赖 torch 2.7 的现有 base 环境，未来应建立独立 vLLM 环境并验证 Triton-Distributed 兼容性。仓库为浅克隆，签出 tag 的 detached HEAD，不包含完整历史或自动安装的依赖。

两份现有模型路径为 `/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B` 和 `/root/eb-public/huggingface-models/Qwen/Qwen3.6-27B`。索引列出的权重文件分别约 131.42 GiB 和 51.75 GiB，合计约 183.17 GiB，不含其他文件；复制到 256 GiB 数据盘在当前空间下预计可放，但会挤占环境、构建、缓存和结果空间。共享模型可直接读取，尚未复制或下载；文件内容与实际加载仍需验证。

## 已执行的功能检查

- CUDA IPC 协议：1 MiB、逐轮变异，默认连接数与 `CUDA_DEVICE_MAX_CONNECTIONS=1` 各 10,000 轮；数据逐元素比特比较，均无不匹配。另测 64 MiB、16 轮、默认连接数，通过。未覆盖 1–64 MiB 的全部大小及每个大小的 10,000 轮。
- NVSHMEM DX 前向：默认连接数，FP16/BF16、3 个小形状、两种调用流、各 64 轮通过；小形状在默认连接数与 `CUDA_DEVICE_MAX_CONNECTIONS=1` 下各进行了两种 dtype 的 2,000 轮浸泡。原生产基准及其全部形态尚未完成迁移，因此 **G2 未通过**。
- NVSHMEM AG+dgrad：单个小形状，FP16/BF16，默认连接数及 `CUDA_DEVICE_MAX_CONNECTIONS=1` 各 128 轮；gather 与输入精确一致，梯度与 PyTorch 参考比较通过。
- 独立 ReduceScatter 原型：8192 元素，FP16/BF16，同步和延迟等待、scale 1.0/0.5；默认连接数 64 轮、`CUDA_DEVICE_MAX_CONNECTIONS=1` 256 轮通过，并确认输入未修改。原版支持的别名输出、多 lane 和 chunk event 尚未实现，因此不是完整 `dxv_op` 移植。
- Dense 行/列并行一阶 autograd 实验封装：FP16/BF16，默认连接数和单连接各 8 轮前向、输入梯度和权重梯度对照通过；backward 使用 NCCL，不是原版融合反向训练路径。
- 本地上游 Megatron-LM 核心可通过临时 `PYTHONPATH` 导入；提示缺少 TE、Apex 并回退部分实现，`sentencepiece` 未安装。导入成功不代表训练程序可运行，也没有修改或安装依赖。

本轮单连接 RS/autograd 和 DX 回归的终端输出保存在 `/root/data/stowaway-nvidia/results/raw/functional_20261109.log`。早先小规模检查未留存原始日志。

上述测试是小规模功能检查，不是性能测量。IPC 测试的 ACK 与 DX 算子采用同类 ticket/ACK 顺序，但两者使用不同的堆实现；不能把 IPC 测试替代 NVSHMEM 压力门。当前 DX 是独立原型，尚未证明与原仓库生产路径逐项等价。

## 当前未完成

G1 尚未完成全部指定大小及轮数；G2 的独立 RS 生产功能、原生产 benchmark 及完整反向/训练尚未完成。G3 的判据见 `F50_verdict.md`，尚未运行。G4、Megatron/vLLM/MoE、规划器重标定和论文性能实验均未开始。复现命令在项目 README 中；后续正式实验应保存完整原始日志。

## 2026-11-09 算子和 Megatron 进一步检查

- `bench/operator_smoke.py` 对四个双卡算子进行了逐轮变异正确性与简短计时；M=8192、K=2560、N=5120 的 GEMM+RS 对应 27B TP=2 的 o_proj 形状。默认及 `CUDA_DEVICE_MAX_CONNECTIONS=1` 的小形状，以及单连接较大形状均运行通过。AG 的 gathered tensor 与分块 GEMM 参考逐位一致；其他操作按容差检查。
- 单连接大形状，3 轮正确性、5 次计时、每批 4 次：AG+GEMM 原版/原型 1.0915/1.0065 ms；GEMM+RS 1.2524/0.9488 ms；AG+dgrad+wgrad 2.1219/1.7791 ms；独立 RS 0.4170/0.3488 ms。前两列不是同一层的统一 M/N/K 定义；此脚本用相同张量尺寸验证各路径，只有 GEMM+RS 明确对应该 o_proj 形状。**这是功能冒烟、短时粗测，不是正式论文数字。** JSON 和完整日志在 `/root/data/stowaway-nvidia/results/raw/operator_27b_oproj_tp2_strict.json` 和同名 `.log`。
- 单连接模式下独立 RS 现在支持输出别名于自身输入分片，FP16/BF16 各 64 轮通过；尚不支持多 lane 和 chunk event。
- Megatron TP=2 两层小模型在关闭 Torch 编译且不启用 SP 时运行了两步（`megatron_baseline_no_sp_eval.log`）。启用 SP 时缺失 Apex/TE 回退到 Torch LayerNorm，触发其 `sequence parallel not supported` 断言（`megatron_baseline_eager_nopersist.log`）。默认编译则遇到 PyTorch 2.7 与当前 Triton 3.4 `CompiledKernel.launch_enter_hook` 不兼容（`megatron_baseline_smoke.log`）。因此 SP baseline、DX hook 和训练正确性尚未通过；不得把非 SP baseline 当作训练迁移完成。
- 在 `/root/data/stowaway-nvidia/envs/megatron-cu126` 创建了继承 base site-packages 的 venv，尝试通过指定代理安装 torch 2.7 对应的 Triton 3.3.0；下载速率约 0.5 MB/s，两次中止，未完成安装。该 venv 尚不可视为隔离或验证完成的训练环境；不能覆盖 base 的 Triton-Distributed。

## 2026-11-09 双卡训练诊断

- `experiments/megatron_prelude.py` 提供仅供诊断的 Torch LayerNorm SP 兼容层，给 norm 参数设置 `sequence_parallel` 以使 Megatron 归约参数梯度；独立测试将两卡分片的 LayerNorm 和梯度 all-reduce 与全量参考比较，FP32/BF16 通过（`megatron_torch_norm_shard_check.log`）。这不是上游支持配置，也不替代 TE/Apex 正式环境。
- 以相同诊断兼容层、关闭 Torch 编译的设置，Megatron TP2/SP 小模型 baseline 与 DX 各跑两步；DX 行/列路径各命中 16 次，每 rank 四次列路径回退是宽度超过 8192 的输出层。行并行 backward 使用 DX AG+dgrad。训练退出前集体释放 NVSHMEM contexts；见 `megatron_sp_baseline_capture.log`、`megatron_sp_dx_capture.log`。
- `experiments/compare_captures.py` 比较两 rank、两 step 的每个参数（每个文件 20 个张量），`rtol=0.02, atol=0.002` 下通过，最大绝对差为 0.00036621094；完整结果为 `megatron_sp_parameter_comparison.log`。这只能验证该小模型、两步和诊断条件，不能证明真实目标训练数值或性能。
- 单连接模式的独立 RS 在 FP16/BF16 下各运行 2,000 轮同步、延迟等待和自身分片别名测试，结果正确，日志 `rs_2000_conn1.log`。完整 G1/G2、原生产语义（多 lane、chunk event）、SP 正式环境、vLLM/MoE、候选池和 8 卡仍未验证，不能宣布整体迁移完成。

## 2026-11-09 双卡继续验证

- 在相同诊断 TorchNorm 和禁用编译条件下，baseline 与 DX 各运行 5 步 full recompute，DX 每 rank 行、列各命中 80 次，输出层回退 10 次；两 rank、五步所有 20 个参数按 `rtol=0.02, atol=0.002` 对照通过，最大绝对差 0.00073242188。日志 `megatron_recompute_base.log`、`megatron_recompute_dx.log` 和 `megatron_recompute_parameters.log`。该对照不等价于目标模型、真实数据或正式软件栈验证。
- 单向生产者/消费者 ticket/ACK 协议：NVSHMEM 对称堆在 1、4、16、64 MiB，默认和单连接模式分别各 10,000 轮，逐元素比特比较无不匹配；日志 `nvshmem_protocol_sizes.log`。CUDA IPC 同样大小单连接 10,000 轮通过，日志 `ipc_protocol_conn1_sizes.log`；默认模式的 4、16、64 MiB 各 10,000 轮亦通过，但本次默认模式未保存完整终端日志。仍未覆盖所有中间大小、双向并发及故障恢复。
- AG+dgrad 在默认和单连接下，FP16/BF16、单形状各连续 2,000 轮通过（`ag_dgrad_2000_modes.log`）。AG+GEMM 重用同一输入缓冲、逐轮更换内容，默认和单连接模式各 256 轮，gather 和分块 GEMM 均逐位相同（`ag_reused_buffer.log`）。
- 尚未执行 8 卡实验。F50、正式 SP 软件环境、原生产算子完整契约、目标模型及端到端验证仍未完成；不能使用原 DCU 标定数据作为 A800 规划依据。

## 2026-11-09 环境修复与模型身份核对

- 数据盘独立 venv 安装标准 Triton 3.3.0 后，PyTorch 2.7 的 `torch.compile` 小例子可运行，但当前 Triton-Distributed 3.4 的扩展需要 `triton._C.libtriton.distributed`，无法与标准 Triton 3.3 共存。NVIDIA DX 原型不使用该扩展，因此将原型的 NVSHMEM 初始化从 `triton_dist.utils` 解耦，使用 NVSHMEM Python API。**这不意味着论文 1 Triton-Distributed 内核可在此环境运行**；它仍需要对应的 Triton-Distributed 环境。
- **更正：**先前名为 `triton33_*` 和 `megatron_triton33_*` 的双卡日志使用了 base 环境的 `torchrun` 脚本，实际子进程不是独立 venv，不能作为 Triton 3.3 的验证。已将运行入口改为 `python -m torch.distributed.run`，新日志才使用 PATH 指定解释器。单进程 Triton 3.3 `torch.compile` 测试不受影响。
- 共享目录 `Qwen/Qwen3.6-27B/config.json` 的实际 `model_type` 是 `qwen3_5`；其 text_config 为 hidden=5120、FFN=17408、64 层、24 attention heads、4 KV heads、head_dim=256，包含 48 个 linear-attention 与 16 个 full-attention 层。DCU 仓库 27B 载具的 heads=40、KV=8、32 层和普通 Transformer 设置仅复用部分维度，**不能表述为完整 Qwen3.6-27B 架构或真实权重训练**。模型目录命名及版本来源应由项目方确认。

## 2026-11-10 TE 与 F50

- 隔离 venv 中安装 Triton 3.3.0、Transformer Engine 2.16.1 的 CUDA12 核心与基于 PyTorch 2.7 本地编译的 PyTorch 扩展。源码包下载时的元数据默认要求 CUDA13 核心，本机 CUDA12 环境以 `--no-deps` 构建并安装。实际安装的扩展元数据要求 CUDA12 核心；但该核心 wheel 的内部 `WHEEL` 标记为 cp310，而当前 Python 为 3.12，`pip check` 因此仍报告核心不受本平台支持，尚非完整受支持的软件组合。一个已发现的 ONNX/ml-dtypes 版本冲突已在 venv 中修正；未改动 base 环境。
- 修正 launcher 后，使用该 venv 的 PyTorch/Triton/TE，在 local linear + TE TENorm（通过本地 spec 适配，非上游原样配置）、启用编译和 full recompute 下，Megatron TP2/SP 小模型的 baseline 与 DX 各运行 5 步；每卡 row/column 各命中 80 次，保存的每步参数均在 `rtol=0.02, atol=0.002` 内，最大绝对差 0.00061035156。日志为 `megatron_te_local_base_v1.log`、`megatron_te_local_dx_v1.log` 和 `megatron_te_local_parameter_comparison.log`。尚未验证目标模型、完整原始生产配置、全部梯度和 8 卡。
- 双卡 F50 初步结果见 `F50_verdict.md`。中位数未观察到 DCU 的 host 拉回；预注册没有明确异常值和中位数处理，仍需复核，不能宣称跨平台规律或将规划器系数直接归零。
- 修正 launcher 后，真实 venv 里的 Triton 3.3/NVSHMEM 在单连接模式下完成 1 MiB、10,000 轮协议和小形状 DX/autograd 测试，日志 `triton33_corrected_tests.log`。27B TP2 的 o_proj 形状 `M=8192 K=2560 N=5120` 冒烟通过，日志及采样为 `operator_te_triton33_27b_shape.log` / `.json`，仅用于检查功能。
- 同一 Triton 3.3 环境进一步完成 NVSHMEM 64 MiB 逐轮变异 10,000 轮单连接协议测试，见 `triton33_nvshmem_64m_10000.log`；不表示全部 G1/G2 验证完成。
- 中等模型（hidden 1024、FFN 4096、2 层、seq 512、mock data）在同一 TE TENorm 组合下 baseline 与 DX 各运行 3 步并完成 full recompute；DX 每 rank row/column 各命中 24 次、输出层回退 3 次。三步 loss 差小于 0.0003，但未做逐参数比较，见 `megatron_te_medium_base.log` 和 `megatron_te_medium_dx.log`。
- 在 base 环境的 Triton-Distributed 3.4 中，执行原仓库的 `bench_3rdv3_windowed_panel_gemmrs.py`（M=1024,N=512,K=1024，TP2、BF16、chunk=512、window=1、band=1）；两卡与参考的 `assert_allclose` 均通过，日志 `anchor_v3_gemmrs_smoke.log`。这只是论文 1 相关 v3 路径的小形状功能冒烟，不代表论文所有内核、TP8 或性能已验证。另一原生 Triton-Distributed GEMM-RS 冒烟见 `triton_dist_original_gemmrs_smoke.log`。
- 27B **载具维度**单层 TP2/SP、seq 1024、MBS 1、full recompute、mock data，参考臂和 DX 各执行一步；包含 H=5120、FFN=17408、heads=40、KV=8、RMSNorm、SwiGLU、RoPE 和 untied embedding。DX 将 `STOWAWAY_COLUMN_MAX_N=20000` 后每 rank 行/列均命中 4 次、输出层回退 1 次；两臂打印 loss 分别 13.45733 / 13.45707，grad norm 均 10.159。日志 `megatron_27bvehicle_1layer_base.log`、`megatron_27bvehicle_1layer_dx_all.log`。仅一步且未保存参数/梯度，不能作为训练正确性门或完整 Qwen 模型；此诊断脚本没有开启原 DCU 载具的 distributed optimizer，也没有使用原 4096 序列长度与对应 GBS。

## 2026-11-10 生产语义与载具数值继续验证

- 原 DCU `dxv_op` 虽在签名中接受 `chunk_ev`，实际上并未使用它；真正按 chunk 等待的是另一条 ring/relay 路径。NVIDIA 原型对非空 `chunk_ev` 明确报不支持，避免无声忽略产生错误；不能宣称实现了 chunk-ready 功能。独立 RS 新增 `k_slots` 多 lane、buffer plane 和 ACK 复用，默认及单连接模式、FP16/BF16、lane 1/2/4、连续 256 轮，含多个延迟句柄倒序等待及本 rank 输入别名测试通过；日志 `rs_multilane_handles.log`。尚未扩展到 8 rank。
- 27B 载具维度单层、seq=1024、TP2/SP、full recompute、mock-data 下，baseline 和 DX 各训练 3 步；每 rank 行列路径各命中 12 次、输出头回退 3 次。每个 rank/step 的 11 个参数在 `rtol=0.02, atol=0.002` 下通过，最大绝对差 0.00073242188，见 `megatron_27bvehicle_parameter_comparison.log`。另针对第 1 步收集各参数 `main_grad`，两卡各 11 个梯度在相同容差内通过，最大绝对差 0.00030517578，见 `megatron_27bvehicle_grad_comparison.log`；采集包含完整 main_grad、不是性能运行。并非完整模型、完整序列或多步梯度门。
- 小模型 TP2/SP 另在 `--use-distributed-optimizer --overlap-grad-reduce` 下 baseline 与 DX 各完成两步；两 rank 每步 20 个参数在相同容差内通过，最大绝对差 0.00036621094，日志 `megatron_distopt_parameter_compare.log`。TP2 时 DP=1，该测试不验证多 DP rank 的梯度同步。
- 完整的原生产 benchmark、符合版本元数据的 TE 环境、F50 正式判据复核、目标 27B/70B 层数与 4096 序列、多 rank 和 vLLM 等仍未完成；无 8 卡结果。
- 新增 `planner/platform.py` 的未标定安全回退和模型形状校验，`tests/test_planner.py` 验证拒绝非 A800 配置、拒绝不一致的 head 维数、无校准时所有调用点选 stock。该文件不是完整规划器，不拟合或使用 DCU 数据。

## 2026-11-10 两卡继续迁移

- 独立 RS 在 Triton 3.3 venv、`CUDA_DEVICE_MAX_CONNECTIONS=1` 下，FP16/BF16、每种组合 2000 轮：同步、延迟句柄、lane 1/2/4、输入分片别名等配置通过参考对照。日志 `rs_multilane_2000_conn1.log`。它仍限同机 TP2，不能外推到 TP8。
- PyTorch 2.7 的私有 `symm_mem` AG+GEMM 和 GEMM+RS 算子分别在双卡 FP16/BF16 各 64 轮对照通过，`pytorch_symm_mem_smoke_conn1.log`。独立编译示例显式启用 Inductor 私有 `_micro_pipeline_tp` 并启用对称内存，两种图改写生成代码均包含对应融合算子且数值对照通过，日志 `pytorch_async_tp_compile_smoke.log`；有 FX erase_node 警告。两者均非 Megatron 端到端集成或性能结果，使用的是私有接口。
- Megatron TP2/SP 27B 载具维度单层、seq 4096、MBS 1、GBS 2、full recompute、mock data、distributed optimizer（DP=1）的参考和 DX 各执行两步；每卡行列各命中 16 次、输出头等列路径回退 4 次。两个 rank 两步各 11 个保存参数逐元素一致，日志 `megatron_27bvehicle_seq4096_parameter_compare.log`，运行日志 `megatron_27bvehicle_seq4096_capture_base.log` 和 `megatron_27bvehicle_seq4096_capture_dx.log`。仅保存 BF16 参数，并不说明 FP32 优化器状态或全部梯度逐位一致；模型不是 Qwen3.6 完整架构或真实权重。
- 新增 `stowaway_nvidia/megatron_brs.py`，试验接入 TP 的 layers/mappings RS；不接入 optimizer bucket，后者涉及多 handle 合并及不同通信组。两层小模型、seq 128、TP2/SP、full recompute、3 步，lane=2；每卡 layers 命中 30、mappings 命中 24 且回退 24。两 rank 每步 20 个保存参数逐元素一致，日志 `megatron_brs_small_parameter_compare.log`。存在惰性集体创建 context 的顺序约束，仅供诊断。
- 同一 BRS 接入进一步用于上述 27B 载具维度、seq 4096、两步配置；每卡 layers 命中 12、mappings 命中 8、mappings 回退 12，两 rank 两步各 11 个保存 BF16 参数逐元素一致；日志 `megatron_27bvehicle_seq4096_brs.log` 和 `megatron_27bvehicle_seq4096_brs_parameter_compare.log`。未替换 optimizer bucket，未验证 DP>1 或完整模型。
- 27B 载具维度增至 4 层、seq 4096，TP2/SP、MBS 1、GBS 2、mock data、full recompute 和 distributed optimizer（DP=1），参考与 DX 各运行 3 步。DX 每卡 row/column 各命中 96 次、column 回退 6 次，三步打印 loss/grad norm 与参考一致；日志 `megatron_27bvehicle_4layer_seq4096_base.log` / `_dx.log`。未采集此项参数或梯度，仅属于训练冒烟，非数值验收或性能结果。

## 2026-11-10 单卡 GEMM 标定输入

- 新增 `planner/gemm_probe.py`，以 BF16 测试 27B 和 70B **载具维度**四个 TP 调用点的单卡 GEMM；TP=2、M=8192/16384/32768，5 次预热、25 次单次采样、8 组每组 16 次连发。每组包括原始采样、运行参数和 GPU/软件信息；结果分别为 `planner/data_a800/gemm_probe_tp2_m8192.json`、`gemm_probe_tp2_m16384.json`、`gemm_probe_tp2_m32768.json`，日志 `gemm_probe_tp2_retry.log`。三份文件各 8 个调用点，数据结构检查通过。
- 这些是单卡 GEMM 原始测量，TP 表示矩阵分片维度，并不代表双卡集合通信或 TP8 性能。未测通信时间、DX 数据面和完整 cells，也没有拟合 A800 模型；规划器继续使用 stock 安全回退。
- 同样在单张 A800 上采集 TP8 分片形状的三组 M=8192/16384/32768：`planner/data_a800/gemm_probe_tp8_m8192.json`、`gemm_probe_tp8_m16384.json`、`gemm_probe_tp8_m32768.json`，各含 8 个调用点及原始样本，日志 `gemm_probe_tp8_single_gpu.log`。这是 G4 的单卡 GEMM 采样，不是八卡实验或八卡通信标定。
- 新增 `bench/tp2_cells.py`，按 27B/70B 载具的四个实际 TP2 调用点形状分别检查 3 轮逐轮变异正确性，并采集 GEMM、NCCL 通信、stock、DX 的 8 组 × 16 连发样本，两卡取每组较慢值。M=4096、8192、单连接各 8 个格子均完成，原始 JSON 和日志 `tp2_cells_m4096_conn1.*` / `tp2_cells_m8192_conn1.*`。调用顺序固定且仅少量重复，DX 是简化原型，数据未进入拟合或规划器，不是论文性能结果。
- 同样流程完成 M=16384 的 8 个调用点，原始结果 `tp2_cells_m16384_conn1.json`、日志同名 `.log`；仍是探索性采样，不满足完整标定或端到端实验门。
- M=8192 又在不设置 `CUDA_DEVICE_MAX_CONNECTIONS` 时完成相同 8 个调用点和检查，结果 `tp2_cells_m8192_default.json`。不同连接配置下部分 DX 计时变化明显，不能将单连接和默认连接的测量混为同一组校准数据。
- M=32768 单连接也完成两个载具各四个调用点，结果 `tp2_cells_m32768_conn1.json`。目前这四个 M 值按形状各一次探索性采样，尚未随机顺序、跨时段重复，也未涵盖其他候选或校验代价模型泛化。
- 默认连接模式下 M=4096、16384、32768 的对应采样也完成（`tp2_cells_m4096_default.*`、`tp2_cells_m16384_default.*`、`tp2_cells_m32768_default.*`），和默认模式 M=8192 一起为上述 32 个形状提供第二种连接条件；并不等于正式 G2 原 benchmark 门通过。
- 为原一级代价模型的 RS 数据面项补采 `tp2_cells_m{4096,8192,16384,32768}_dxcomm_checked_conn1.json`：每个 RS 形状先做 3 轮 DX 独立 RS 与 NCCL 结果对照，再以同样 8×16 的采样记录 DX RS。早先 `tp2_cells_m4096_dxcomm_conn1.json` 未在该脚本内检查独立 RS，后缀 `checked` 的文件是有效的这轮版本。仍需独立验证拟合、数据范围和候选可比性，不能自动启用规划器。
- `planner/extract_tp2_cells.py` 将该单连接数据转换成原代价模型字段，生成明确标记未验证的 `planner/data_a800/cells_tp2_conn1_exploratory.json`。`planner/evaluate_tp2_costmodel.py` 读取原 DCU costmodel 结构并留一载具离线试验：32 点选中两候选事后较快者 19 点，所选慢于 stock 0 点；stock/选择/oracle 的逐点中位时间直接相加分别 313.80/305.97/302.29 ms。详见 `costmodel_tp2_conn1_loo_exploratory.json`。只有两种载具、固定顺序一次采样、没有其他候选和整步验证，该比例不构成正式验证，模型不加载到部署规划器。
- PyTorch 2.7 `_coalescing_manager` 负责聚合 optimizer bucket work；直接替换单个 RS 会使该管理器无法等待自定义句柄。当前 world=TP=2、DP=1，不能以现有 TP 训练证明跨 DP 的 optimizer bucket 通信正确；该路径尚未接入。

## 2026-11-10 vLLM 迁移开始

- 新增 `stowaway_nvidia/vllm_sp.py`：按原仓库 vLLM 序列并行前向实现 stock-SP 与 DX 两种诊断路径，限制单机 TP2、无量化且无偏置 Llama、eager 执行、无 PP。尚未支持规划器或多机，context 集体创建仍要求各 rank 路径和顺序一致。
- `tests/test_vllm_projections.py` 在 Megatron Triton 3.3 环境以模拟线性层接口检查适配器的投影而非完整 vLLM：双卡、单连接、FP16/BF16、stock-SP 和 DX 各 256 轮，随机变化输入与权重，对 NCCL/GEMM 参考通过，日志 `vllm_adapter_projection_conn1_retry.log`。首次运行清理时因非必要 vLLM import 失败，已修复；实际上游 layer、attention、KV cache、完整生成未验证。
- 相同模拟投影测试在未设置 `CUDA_DEVICE_MAX_CONNECTIONS` 时也通过，日志 `vllm_adapter_projection_default.log`。
- 两层合成 Llama 形状前向进一步串联残差、归一化和四个投影，与每层 all-reduce 的全量参考对照；stock-SP / DX、FP16/BF16、64 轮，在单连接及默认连接模式均通过。日志 `vllm_synthetic_forward_conn1.log` / `vllm_synthetic_forward_default.log`。其中 attention 和 vLLM Python 模块均为模拟件，不能证明真实推理路径。
- `experiments/vllm_gate.py` 和 `compare_vllm_gates.py` 是尚未运行的自然文本贪心 token 对比门，禁止将其存在视为推理结果。vLLM 0.11.0 独立 venv 已开始安装，Transformers 选用该上游版本 `requirements/test.txt` 中的 4.55.2；安装与推理状态应以之后的记录为准。
- 独立环境 `/root/data/stowaway-nvidia/envs/vllm-cu128` 已 `--no-deps` 安装 vLLM 0.11.0、Transformers 4.55.2 主包；依赖仍在下载和安装中，不能运行或视为可用环境。Torch 2.8 wheel 另行续传至 `/root/data/stowaway-nvidia/wheels/`，未下载完整前不可安装。
- Torch 2.8.0 wheel 下载完成，SHA-256 `b2aca0939fb7e4d842561febbd4ffda67a8e958ff725c1c27e244e85e982173c` 与 PyPI 元数据一致；正在安装主 wheel 及其依赖，仍未验证 vLLM 环境。
- Torch 2.8.0 主 wheel 已以 `--no-deps` 安装，vLLM 所需其余依赖仍在安装过程中；代理下载缓慢，部分 CUDA 核心 wheel 同时下载到 `wheels/`，未完成下载的文件不可用。安装状态及失败信息见 `vllm_011_dependencies_after_torch.log`，CUDA wheel 的 `vllm_*_download.log`。无 `pip check` 通过记录或实际 Llama 推理结果前，环境仍视为未完成。

## 2026-11-10 分布式优化器 DP2 诊断

- `stowaway_nvidia/megatron_optimizer_brs.py` 实验包装 Megatron bucket group，收集本组所有 DX 句柄并与 NCCL work 一起等待。当前仅 world=DP=2、单机、单 optimizer instance、FP16/BF16 通信、固定分支；全 DX 时插入一个小型 NCCL all-reduce 以满足 PyTorch 2.7 coalescer 的 work 要求，因此不能将其作为性能实现。惰性集体创建和跨流顺序仍有约束。
- 为验证真实 DP 组，使用 TP1/DP2、小模型、BF16 梯度通信、distributed optimizer、overlap grad reduce、full recompute，参考与 DX 各 3 步；每卡 DX bucket 命中 3 次，参考与 DX 两 rank 每步 20 个 BF16 保存参数逐元素一致。日志 `megatron_optbrs_dp2_base.log`、`megatron_optbrs_dp2_dx_retry.log` 和 `megatron_optbrs_dp2_parameter_compare.log`。首次运行出现同形状 context 被不同调用流使用的错误（`megatron_optbrs_dp2_dx.log`），改为各 context 固定 worker stream 并由事件建立依赖后重跑通过。
- 同配置单独采集 forward/backward 返回后的 `main_grad`：参考与 DX 各两步，每个 rank/step 的 20 个梯度逐元素一致，日志 `megatron_optbrs_dp2_grad_compare.log`。这仅覆盖该小模型和相应梯度采集时点。
- 使用真实 Megatron `_ParamAndGradBucketGroup` 的合成 bucket 测试，同组 3 bucket、2 lane、rank 输入 shard 别名，每种同步/异步和 SUM/AVG 组合各 128 轮；全 DX 和部分 NCCL 回退在单连接模式通过，混合模式在默认连接下通过。日志 `optimizer_bucket_multihandle_all_dx_conn1.log`、`optimizer_bucket_multihandle_conn1.log`、`optimizer_bucket_multihandle_default.log`。测试合成输入，不代表真实多 bucket 训练、TP+DP 混合或 8 卡正确性。
- 为优化器适配器增加调用通信组对象的严格匹配保护后，单连接多 bucket 混合回退回归（`optimizer_bucket_multihandle_groupguard.log`）和 TP1/DP2 三步保存参数对照（`megatron_optbrs_dp2_guard_parameter_compare.log`）均通过，后者两 rank 每步 20 个参数逐元素一致。
- 进一步在合成多 bucket 检查中调用 Megatron 实际 `finish_grad_sync()` 消费异步句柄及 reset，而不是直接调用 handle.wait；单连接、混合回退、SUM/AVG、同步/异步各 128 轮通过，见 `optimizer_bucket_finish_sync_conn1.log`。
- 增加非标准回退 collective 返回真实 Work 时的句柄保留，跳过 PyTorch fast-path 的不可等待占位符；回归 `optimizer_bucket_finish_sync_workguard.log` 和真实 DP2 训练参数对照 `megatron_optbrs_dp2_workguard_parameter_compare.log` 通过。当前没有专门注入非标准 fallback Work 的验证，仍属未覆盖边界。
- 将优化器适配阈值设置为 1 GB，强制小模型所有 bucket 回退 NCCL：TP1/DP2 三步每卡回退 3 次、DX 0 次，所有保存参数与原参考逐元素一致，日志 `megatron_optbrs_dp2_allfallback_compare.log`。
- 复核 PyTorch coalescer 后，marker 触发条件改为实际 coalesce 操作列表为空，避免非标准回退返回 Work 但未登记 NCCL fast-path 时出现未定义 work。混合 bucket 检查 `optimizer_bucket_coalescing_fix_conn1.log` 与真实 DP2 参数比较 `megatron_optbrs_dp2_coalescing_fix_compare.log` 通过；依赖 PyTorch 2.7 私有状态，其他版本未验证。
- marker 修改后全 DX 多 bucket 同步/异步 SUM/AVG 各 128 轮通过（`optimizer_bucket_coalescing_fix_all_dx.log`）；FP32 梯度通信的 TP1/DP2 小模型两步走 stock 回退，日志 `megatron_optbrs_fp32_fallback.log`，每卡回退 2 次，此项未逐参数比较。
- 将真实 `finish_grad_sync()` 的合成三 bucket 混合回退检查延长到每种 SUM/AVG、同步/异步各 2000 轮，单连接下通过，日志 `optimizer_bucket_finish_sync_2000_conn1.log`；该检查不替代真实多 bucket 训练。

## 2026-11-10 中断后继续：同步等待修复

- 复查 PyTorch 2.7 `_coalescing_manager` 实现发现：同步模式结束时只调用其 NCCL work 的 `wait()`，不会遍历先前添加到 `cm` 的 DX handles。因此上一版同步优化器路径缺少调用流对 DX 完成的依赖，旧的同步多 bucket 测试可能被短操作时序掩盖，不能单凭它们认定同步路径正确。现已在同步 coalescing 返回前逐个等待自定义 handle；异步模式继续由 Megatron 的 `finish_grad_sync()` 等待。
- 新增 `tests/test_optimizer_sync_fence.py`：人为在 DX worker 上排入 GPU 延迟，紧随同步调用在调用流复制结果并检查。单连接和默认连接两种模式各 32 轮、两个尺寸通过，日志 `optimizer_sync_fence_conn1.log` / `optimizer_sync_fence_default.log`。混合同步/异步多 bucket 回归各 256 轮通过，日志 `optimizer_bucket_syncfix_regression_20261110.log`。
- 真实 Megatron TP1/DP2，小模型 BF16 梯度通信、distributed optimizer、关闭 `overlap-grad-reduce`，参考和 DX 各 3 步；每卡命中 3 次，同步模式各 rank 各步 20 个保存参数逐元素一致，`megatron_optbrs_dp2_sync_compare_20261110.log`。尚未验证混合 TP/DP、真实多 bucket 训练或性能。
- 恢复时发现先前代理曾停止，vLLM 依赖安装和 CUDA wheel 续传已被中断；代理恢复后重新启动下载，未完成文件仍不可使用，等待完整性检查及实际运行。
- `SymmetricBuffers` 在任何对称分配之前用 world 组检查两 rank 的完整分配形状与 dtype；故意使用不同尺寸时两端均报错，相同布局可分配释放（`symmetric_layout_guard_20261110.log`）。相同调用顺序下该检查可避免布局错配，但无法防止某个 rank 根本不进入集合调用或调用不同类别导致的死锁。DX 多形状 256 轮和 TP1/DP2 同步小模型两步参数对照回归通过，见 `dx_layoutguard_regression_20261110.log`、`megatron_optbrs_dp2_sync_layoutguard_compare_20261110.log`。增加该检查会增加 context 创建期的同步和 host 开销，先前采样结果不能代表其初始化成本。
- Megatron TP 路径支持在同一进程同时启用 DX 行/列和独立 BRS；column DX backward 通过可注入 RS 接口使用 BRS，而不是固定调用 NCCL。小模型 TP2/SP、full recompute 3 步：每卡 row/column 各 48 次、BRS layers 30 次，两个 rank 各步 20 个参数按 rtol=.02、atol=.002 对照通过，最大绝对差 .00048828125，见 `megatron_combined_dx_brs_compare_20261110.log`。27B 载具维度单层 seq=4096、MBS1/GBS2、TP2/SP、distributed optimizer（DP1）、full recompute 两步：每卡 row/column 各 16 次，BRS layers 12 次，两个 rank 每步 11 个 BF16 参数逐元素一致，见 `megatron_27bvehicle_seq4096_combined_compare_20261110.log`。不是完整模型、真实数据或 TP+DP 混合验证。
- 新增 `bench/production_gate.py`，使用当前 NVIDIA 原型与 DCU G2 同类判据，成批 4 个新输入连续检查 RS、AG、AG+dgrad，另以容差检查 wgrad。BF16，M/K/N=2048/512/512 各 64 次和 8192/1024/1024 各 32 次，在默认及单连接两种配置均零失败；上述四组汇报的 RS、AG 分块/完整张量、dgrad 和 wgrad 的最大差均为 0，日志 `production_gate_{2048,8192}_{conn1,default}_20261110.log`。这是适配原型的新测试，不是原始 DCU 生产 benchmark，也未覆盖目标调用点全部尺寸。
- DX column 诊断 hook 增加按输出宽度范围或显式宽度集合筛选。小模型下 `STOWAWAY_COLUMN_WIDTHS=256`，结合 DX row 与 BRS，TP2/SP full recompute 各 3 步；每 rank row 48、column DX 24、column fallback 30，BRS layers 30，所有 rank/step 20 个 BF16 参数与原参考逐元素一致（`megatron_combined_selective_width_compare_20261110.log`）。选择方式是诊断性宽度筛选，不是自动规划器；相同宽度的调用点无法靠此独立选择。
- 上游 Megatron local linear 构造器接收但不保存 `tp_comm_buffer_name`；诊断 prelude 现将它保存为标识，允许 `STOWAWAY_DX_SITES` 从 qkv/proj/fc1/fc2 中选择，避免宽度相同时无法区分。小模型 TP2/SP、full recompute、3 步仅选择 fc1/proj 并启用 BRS，每 rank row/column 各 24 次且分别 fallback 24/30 次，BRS layers/mappings 30/12 次；每 rank/step 20 个 BF16 参数与参考逐元素一致，日志 `megatron_combined_selective_sites_compare_20261110.log`。这是静态诊断选择，并未连接未通过验证的规划器。
- 完整 TE transformer 层的 TP2/SP 小模型可运行。`--tp-comm-overlap` 默认 multicast 失败，TE 报告当前设备/驱动/toolkit 不支持并建议 `UB_SKIPMC=1`；改为该设置、gloo bootstrap 后 userbuffers 建立并运行。完整 TE 基线与 TE overlap（两个都是 TE 层）、full recompute 各 3 步，参数对照在 rtol=.02、atol=.002 下通过，最大差 .00048828125，日志 `megatron_te_full_overlap_compare_20261110.log`。另按模块检查配置：qkv/fc1 的前向 AG、proj/fc2 的前向 RS 为 True，见 `megatron_te_overlap_layerflags_20261110.log`；该检查说明层配置并运行，并非 profiler 证明每个核的时序。TE 2.16.1 CUDA12 wheel 的平台元数据警告尚未消除，软件组合仍属诊断环境，非正式候选性能结果。

## 2026-11-10 恢复后：TP 跨流和 MoE 算子

- TP BRS 的每个尺寸上下文绑定独立 worker 流，调用流在投递前建立输入依赖，同步调用在当前流等待结果，异步调用由返回句柄在消费流等待。原先相同尺寸换调用流会抛错。注入 GPU 延迟，在两条生产流和第三条异步消费流交替发起，同步/异步及 Megatron layers/mappings 路径各 64 轮；单连接、默认连接均通过，日志 `tp_brs_cross_stream_conn1_20261110.log`、`tp_brs_cross_stream_default_20261110.log`。这不支持多线程并发发起或不同 rank 不一致的调用顺序。
- 修改后真实 Megatron 小模型 TP2/SP、DX+BRS、full recompute、3 步，每卡行列各 48 次、BRS layers 30 次；每 rank 每步 20 个保存 BF16 参数与参考按 rtol=.02、atol=.002 比较通过，最大差 .00048828125，日志 `megatron_combined_brs_worker_compare_20261110.log`。首次引用了保存 5 步的参考目录，比较器正确拒绝步数不一致；改用同配置 3 步参考后通过。
- 增加设备偏移驱动的 Triton 专家线性正确性原型 `stowaway_nvidia/moe_grouped.py`，FP16/BF16 前向、输入梯度、权重梯度和空 expert；测试包含全空、倾斜路由、尾行、非对齐宽度、1024 行和 int32/int64 偏移，两张 A800 分别通过 PyTorch 参考对照，日志 `moe_grouped_triton_gpu{0,1}_20261110.log`。BF16 两层投影、SiLU、动态路由 12 轮的前向和梯度参考对照通过，日志 `moe_grouped_mlp_gpu0_20261110.log`。该固定网格实现未调优、偏移由调用方保证合法、不支持二阶梯度，也未实现跨卡 dispatch 或接入 Megatron。Triton 3.3 对动态标量条件选择偏移的 coalesce pass 发生编译断言，改为带 mask 的加载后通过；空输入不编译 wgrad 内核。
- 相同 DX+BRS worker 修改在 27B 载具维度单层、seq4096、TP2/SP、分布式优化器 DP1、full recompute 两步也完成，每卡 row/column 16 次、BRS layers 12 次，两个 rank 每步 11 个 BF16 参数在同一容差内通过，最大差 .00048828125（`megatron_vehicle_brs_worker_compare_20261110.log`）。仅为单层 mock 数据载具，不能称完整 Qwen 模型。
- 原 MoE 中 `torch.bincount` 在当前 CUDA/PyTorch 的同步调试模式下触发同步错误（`moe_bincount_sync_probe_20261110.log`），因此不能直接照搬并声称完整路径无 host 同步。新建 `moe_routing.py` 固定大小 Triton 计数和前缀和，空输入、无效 ID、1/4/17/128 专家和 int32/int64 对照通过，预热之后在 PyTorch 同步调试模式运行也未触发错误（`moe_routing_histogram_20261110.log`）；PyTorch 明确提醒该模式不能检测所有同步，不能单凭此证明完整 MoE 无同步。`torch.argsort(stable=True)` 同模式简单探针未触发错误（`moe_argsort_sync_probe_20261110.log`），未证明其所有输入和版本的行为。
- 本机 top-k 路由与两次专家投影、SwiGLU、加权合并组成 `moe_local.py`；FP16/BF16、各 8 组路由，包括全部 token 路由至同一专家，前向、输入、路由权重和专家权重梯度对照 PyTorch 按 rtol=.04、atol=.02 通过，预热后前向在 PyTorch 同步调试模式未触发错误，日志 `moe_local_swiglu_sync_20261110.log`。初次逐 token 参考的 BF16 权重梯度出现舍入顺序造成的差异，现用按专家成组 GEMM 的参考与原项目运算方式对照；不构成跨卡 EP、Megatron 或性能验证。
- 新增 `moe_exchange.py`：NVSHMEM 对称堆、peer SM store 和 stream wait 的固定容量 EP2 dispatch/返回前向原型，与 Triton 专家线性组成 SwiGLU 计算。随机路由和全部路由到同一个专家、两卡各连续 512 次、周期性注入 GPU 延迟，整个 burst 中不读回结果，两种连接模式均通过参考对照，日志 `moe_ep2_burst_conn1_20261110.log`、`moe_ep2_burst_default_20261110.log`。使用两卡、每卡 32 token、top-k=2、4 专家、隐藏 32 的小算例，测试不代表生产规模；只支持 BF16 前向、固定大小和单 caller stream，不支持梯度、Megatron 或 CUDA Graph。无效路由 ID 由调用方保证不会出现，否则无丢 token 保证。
- 较大 EP2 诊断：每卡 256 token、隐藏 128、专家中间 192、4 专家、top-k 2，单连接连续 32 轮包括倾斜路由，与参考对照通过（`moe_ep2_medium_conn1_20261110.log`），仍不代表生产模型尺寸。vLLM 依赖包通过代理续传；为避免与续传重复下载，已停止尚在下载同一 cuBLAS 包的 pip 进程，下载完成后须先校验哈希再安装。
- EP2 适配器严格绑定创建 context 的 CUDA 流，避免指针表创建与首次 dispatch 跨流时的未定义依赖；不同流拒绝，默认连接 128 轮回归通过（`moe_ep2_streamguard_default_20261110.log`）。预热后 EP2 dispatch、专家计算及 combine 在 PyTorch 同步调试模式下执行通过（`moe_ep2_syncdebug_20261110.log`），该模式无法检测所有同步。vLLM CUDA12.8 cuBLAS wheel 已与 PyPI 公布的 SHA-256 匹配并安装，cuDNN 正在续传，其余依赖重新安装中；vLLM 环境仍未验证。
- 新增 `moe_ep.py`：在 EP2 exchange 上实现前向及一次反向，并将输入梯度、路由权重梯度和专家权重梯度与汇集两卡的 PyTorch 参考比较。固定小形状（每卡 16 token、4 专家、top-k 2、隐藏 32）、随机与倾斜路由，默认和单连接各 64 次通过，日志 `moe_ep2_autograd_default_20261110.log` 和 `moe_ep2_autograd_guard_conn1_20261110.log`。当前限制所有计算输入需要梯度、同一 context 在一次 backward 完成前不能复用、不能重复 backward，不支持梯度累积或重计算和多线程；尚未接入 Megatron，不能视为生产 MoE 训练验证。cuDNN wheel 已完成并与 PyPI SHA-256 匹配、安装，vLLM 其余依赖仍在安装中。
- 加入重复 backward 的 generation 保护并测拒绝路径，默认连接 16 轮 `moe_ep2_repeated_backward_guard_20261110.log`；每卡 64 token、隐藏 128、FFN 192、4 专家、top-k 2 的 EP2 前向和梯度、单连接 16 轮对照通过（`moe_ep2_autograd_medium_20261110.log`）。vLLM pip 当前在下载其他 CUDA 依赖，尚无可运行结果。
- `megatron_moe.py` 将 Triton 专家线性前后向诊断接入上游 `SequentialMLP`，保留原 dispatcher 与参数。Megatron local MoE，TP1/EP2、4 专家、top-k 2、SwiGLU、2 层小模型、mock 数据训练两步，每 rank 命中 4 次且回退 0 次；两 rank 每步 26 个 BF16 参数与上游参考按 rtol=.02、atol=.002 比较通过，最大差 7.63e-6（`moe_megatron_gg_compare_20261110.log`）。首次接入因 dispatcher 传入 CPU counts 安全回退（`moe_megatron_gg_fallback_20261110.log`），现复制 counts 回 GPU 再运行；因此上游已有的 host 同步仍存在。适配器还逐次堆叠复制参数，不能报告性能或声称整层 sync-free；没有使用独立的 NVSHMEM EP 原型，尚未验证优化器状态或完整模型。
- 同配置开启 full recompute，Megatron 参考和专家适配各两步，每 rank 命中 8 次、回退 0 次；每 rank 每步收集的 26 个 `main_grad` 与参考按 rtol=.02、atol=.002 对照通过，最大差 3.82e-6，见 `moe_megatron_gg_grad_compare_20261110.log`。此项仍使用上游 dispatcher，不验证独立 EP2 NVSHMEM 原型的 Megatron 集成。
- `megatron_moe_ep.py` 将 NVSHMEM EP2 dispatch、专家 MLP 和反向试验性接入完整 Megatron `MoELayer`。TP1/EP2、两层小模型、4 专家、top-k2、BF16、mock 数据、无重计算的 2 步训练每卡命中 4 次；每 rank 每步 26 个保存参数与上游参考在 rtol=.02、atol=.002 内通过，最大差 .00048828125（`moe_megatron_ep_compare_20261110.log`）。另一次相同配置采集 26 个 `main_grad`，最大差 .00078582857、同容差通过（`moe_megatron_ep_grad_compare_20261110.log`）。开启 full recompute 时每卡 0 次命中、4 次回退并训练成功（`moe_megatron_ep_recompute_fallback_20261110.log`）。每层 context 必须完整一次反向后才能复用，不支持梯度累积、重计算、CUDA Graph、共享专家、无效路由、多机、并发和其他 NVSHMEM 适配器组合；缓冲区最坏容量、权重堆叠使其不适于性能结论。
- 更正上条中“梯度累积”限制：每次前向完成反向后再执行下一个微批的顺序累积可以运行。TP1/EP2、GBS4/MBS1、两层小模型训练两步，每 rank 8 次命中，参数与参考在同容差内通过，最大差 .00048828125（`moe_megatron_ep_accum_compare_20261110.log`）。仍不能在相同 context 上保留多个未完成反向的前向，其他累积调度尚未覆盖。
- 以原 BW MoE 载具的隐藏 2048、专家 FFN 768、128 专家、top-k8，TP1/EP2、单层、序列 128、MBS1/GBS2、无重计算、mock 数据各跑一步；NVSHMEM 路径每卡命中 1 次、回退 0 次，两个 rank 各 139 个保存 BF16 参数与参考在 rtol=.02、atol=.002 内通过，最大差 .00024414062，日志 `moe_vehicle_ep_compare_20261110.log`。这是短序列单层的载具维度检查，非原载具 4096 序列、12 层、EP8，也不是实际 Qwen 模型或性能结果。
- 另以两层小模型、序列 512、TP1/EP2、4 专家、top-k2 和无重计算执行两步；每卡 EP 命中 4 次、零回退，两 rank 各步 26 个保存参数在同一容差内通过，最大差 .00048828125，`moe_megatron_ep_seq512_compare_20261110.log`。
- 相同小模型启用 distributed optimizer 和 overlap grad reduce、TP1/EP2 两步，EP 路径每卡命中 4 次，两个 rank 每步 26 个保存参数在同一容差内通过，最大差 .00048828125（`moe_ep_distopt_compare_20261110.log`）。此项未单独比对优化器状态。
- vLLM 独立环境依赖安装完毕，加入 NVSHMEM 3.3.9、nvshmem4py 0.1.2、cuda-core 0.2.0 后 `pip check` 通过；torch 2.8.0+cu128、Triton 3.4.0、vLLM 0.11.0、Transformers 4.55.2 可导入，识别两卡并完成小 GPU 矩阵乘法。真实 Llama 双卡推理验证状态以后续记录为准。
- 真实 Llama-3-8B，vLLM 0.11 eager、TP2、prefill M1024、4 条自然文本提示、每条贪心生成 48 token：原版与 stock-SP 各卡 prefill 命中 7 次，192 token 完全一致（`vllm_gate_stock_sp_compare_20261110.log`）。完整 DX 各卡命中 7 次，192 个位置有 12 个不同；首次分歧分别在第 2 条 step43 和第 4 条 step37（0 起始），后续生成可能连锁分歧（`vllm_gate_dx_compare_20261110.log`）。单连接模式重跑的 DX 与默认连接 token 完全一致，仍与参考相差 12（`vllm_gate_dx_connections_compare_20261110.log`）。只启用 DX 列投影与 stock-SP 完全一致，只启用行投影复现 12 处差异（`vllm_gate_dx_col_compare_20261110.log`、`vllm_gate_dx_row_compare_20261110.log`）。这些实验中多数解码步回退原版（每 rank 190 次），不代表全形状覆盖。
- 对真实 Llama 输入，4 次 prefill × 64 个行投影逐层诊断：DX 分块 GEMM+RS 与同样分块 GEMM+NCCL RS 的最大绝对差为 0；相对于整矩阵 GEMM+NCCL，约 19.4% 的 BF16 元素不同，两 rank 整体相对 RMS 约 .00290/.00280，最大绝对差 1/2（`vllm_gate_row_diag2_20261110.log`）。这说明采样中分块 GEMM 造成数值路径差异，不证明所有输入无通信错误；首次分歧的候选 logits 很接近，不能仅凭此将严格 token 差异归为可接受误差。新增 `VS_ROW_EXACT=1` 整矩阵 GEMM + NVSHMEM 独立 RS 对照，在相同 4×48 token 与原版完全一致（`vllm_gate_dx_exact_compare_20261110.log`），不提供行 GEMM/通信重叠，不能将其计作完整 DX 性能。以上均非完整 70B、长上下文、并发、量化或性能验收。
- 增大 Llama-3-8B 的 prefill 到 M2048，8 条自然文本、每条贪心 64 token，整矩阵 GEMM + NVSHMEM RS 对照与 stock 的 512 个生成 token 完全一致；每 rank prefill 适配命中 10 次、回退 507 次（`vllm_gate_m2048_exact_compare_20261110.log`、`vllm_gate_m2048_exact_20261110.log`）。这是有限提示的实际推理检查，回退次数主要是非 bucket 的解码步，不是完整形状覆盖或性能结果。
- M2048 同样 8×64 的分块行投影重叠路径与原版有 164 处 token 不同（含分歧后的连锁生成），见 `vllm_gate_m2048_overlap_compare_20261110.log`。因此将 vLLM 适配器默认改为整矩阵 GEMM + NVSHMEM RS，仅显式 `VS_ROW_EXACT=0` 启用未过严格门的重叠行路径；该默认策略不等于优化路径已达成，也不能作为性能结果。
- 修改默认值后不设置 `VS_ROW_EXACT` 重跑 M1024、4×48，日志确认两 rank `row_exact=True`、各命中 7 次，生成 token 与 stock 全部一致（`vllm_gate_default_guard_compare_20261110.log`）。
- 真实 Llama-3-70B，TP2、BF16、eager、prefill M1024、2 条自然文本各生成 16 token：stock 在 `gpu_memory_utilization=.90` 下生成成功（日志 `vllm_gate_70b_stock_20261110.log`）。NVSHMEM 默认安全路径首次使用 .90 与 1 GiB 对称堆时报 KV cache 无可用空间，并在异常退出时产生 NVSHMEM buffer 清理警告（`vllm_gate_70b_dx_20261110.log`）；改为 `.97` 与 256 MiB 对称堆后成功，每 rank 命中 5 次、回退 32 次，32 个生成 token 与 stock 相同（`vllm_gate_70b_dx_compare_20261110.log`），结束前约每卡 78 GiB。stock 是加入 prompt hash 前启动，比较器提示旧文件无法用哈希验证输入身份；两次使用相同脚本及模型路径生成提示。不同内存预算、样本很少、未对照 logits，不能作为并发/长上下文或性能结论。
- 70B stock 又以 `.97` 重跑并保存 prompt SHA-256，与 NVSHMEM 默认路径两个输入哈希均一致、32 个生成 token 完全一致（`vllm_gate_70b_hash_compare_20261110.log`）。模型直接从共享 `/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B` 加载，没有复制到 `/root/data`。
- `/root/data` 当前剩余约 62 GiB；新建的两个环境为 `megatron-cu126` 约 1.3 GiB、`vllm-cu128` 安装完约 9.9 GiB。Qwen 共享权重 config 实际为 `qwen3_5` / `Qwen3_5ForConditionalGeneration`（64 层含 linear attention）；现有 Transformers 4.55.2 不能识别，vLLM 0.11 源码无 qwen3_5 模型实现，不能当 Qwen3 普通 Transformer 直接运行。外网版本核查尝试时用户指定代理拒绝连接，未更改现有稳定环境。
- Llama-3-8B、M512、4×48，两个文件均带提示 hash；stock-SP 与 stock token 一致，之前的列+整矩阵行 DX 结果有 1 处 token 差异（`vllm_gate_m512_compare_20261110.log`），只启用 NVSHMEM 行投影结果与 stock-SP 一致（`vllm_gate_m512_row_compare_20261110.log`），只启用分块列投影复现该差异（`vllm_gate_m512_col_compare_20261110.log`）。因此进一步将默认 DX 组件改为仅行投影，并保留整矩阵 GEMM；列分块重叠需显式启用。即使目前采样一致，默认路径也未通过所有形状、并发和长文本的严格验证。
- 默认收紧后不设置组件选项重跑 M512、4×48，两个 rank 各命中 7 次、`components=['row'] row_exact=True`，提示哈希与 192 个生成 token 均与 stock 相同（`vllm_gate_m512_default2_compare_20261110.log`）。
- 收紧后的默认组件再跑真实 70B M1024、2×16，提示 hash 和 32 token 均与相同 `.97` 显存预算的 stock 相同（`vllm_gate_70b_default2_compare_20261110.log`）；样本、长度和配置限制仍与前述一致。
- 共享权重自带 README 第 565 行推荐 Qwen3.6 使用 `vllm>=0.19.0`，同时说明 Transformers 需要较新版本；这比当前已验证的 vLLM 0.11 高，不能用现有 Llama 适配器直接覆盖混合 attention。待用户指定代理恢复后，再核对依赖、A800 支持和迁移方式，不能在未核对时覆盖已验证环境。
- vLLM 的 Torch 2.8/Triton 3.4 环境中另跑双卡原型算子门，M1024/K4096/N2048、8 次，AG 分块与参考逐位一致，GEMM+RS 按该 harness 的归一化误差最大 .0078125、小于 .02，dgrad/wgrad 检查通过（`vllm_torch28_dx_operator_20261110.log`）。该低轮次随机算子测试不替代真实生成的严格 token 门，分块重叠仍为诊断选项。
- Qwen3-14B 使用共享真实权重在 vLLM 0.11、TP2、eager、M512、2 条自然文本各生成 16 token 的 stock 基线已运行并保存结果（`vllm_qwen3_14b_stock_20261110.log`）。当前 Qwen3 尚未接入 DX；其 Q/K normalization 和 Llama 路径不同，不能直接套用现有 Llama 适配器。Qwen tokenizer 没有 BOS，gate 已允许无 BOS 的固定长度输入。
- 已针对 vLLM Qwen3 模型接入 Q/K normalization：真实 Qwen3-14B、TP2/eager、M512、2 条自然文本各 16 token，stock-SP 和默认 NVSHMEM 行通信各两 rank 命中 5 次，两个输入 hash 和 32 个 token 均与 stock 一致，见 `vllm_qwen3_14b_sp_compare_20261110.log` / `vllm_qwen3_14b_dx_compare_20261110.log`。默认不启用分块 GEMM 重叠、主要 decode 步回退；样本很少，尚不是性能或完整形状验收。
- 当前代码重新完成单连接 M=4096/8192/16384/32768 两个载具共 32 点的 BF16 算子对照，每点 8 次新输入数值检查、8 组×16 连发，28 点 stock/DX 时间比大于 1，4 点不大于 1；详见 `results/a800/双卡算子对照_20261110.md`。载具对应矩阵形状，非完整模型；该轮未给出端到端加速。
- 当前代码重新运行 Megatron TP2/SP 小模型、DX+BRS、full recompute 3 步，参数对照通过，每卡 row/column 各 48、BRS layers 30（`tp2_final_megatron_compare_20261110.log`）。Qwen3-14B vLLM 默认 NVSHMEM 路径在 M1024、4 条自然文本各生成 48 token，对照输入哈希和 192 个 token 均一致（`vllm_qwen3_14b_m1024_compare_20261110.log`）。
- 使用本轮 32 个算子形状重新运行原代价模型的留一载具离线评估，19/32 与事后更快的两候选一致，stock/选择/oracle 的逐点中位时间之和为 314.451/306.481/302.438 ms，结果 `planner/data_a800/costmodel_tp2_accept_loo_20261110.json`。该数值是各点相加，不是端到端模型运行时间；未接入在线规划器。
