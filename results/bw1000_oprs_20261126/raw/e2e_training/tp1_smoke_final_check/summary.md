# Megatron 训练端到端：tp1_smoke_final_check（TP=1，每臂独立进程；步长中位数自第 3 步起）

| 臂 | kind | plan | 退出码 | 步数 | 步长中位 (ms) | 训练 loss | 验证 loss | 最大已分配 (MB) | 命中（站点=选择:次数） | 回退（原因:次数） | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | serial | - | 0 | 8 | 404.5 | 1.182455E+01 | 1.161879E+01 | 15816.44 | - | - | ok |
| n1 | null | null.json | 0 | 8 | 407.35 | 1.182455E+01 | 1.161879E+01 | 15855.44 | - | plan:640 | ok |
| o1 | oprs | oprs_triton_dist.json | 0 | 8 | 456.85 | 1.182465E+01 | 1.161868E+01 | 15855.44 | fc1=triton_dist:160 fc2=deo:160 qkv=triton_dist:160 | plan:160 | ok |
| t1 | always_triton_dist | always_triton_dist.json | 0 | 8 | 429.4 | 1.182467E+01 | 1.161868E+01 | 15831.44 | fc1=triton_dist:160 fc2=triton_dist:160 proj=triton_dist:160 qkv=triton_dist:160 | - | ok |

退出码 = 每个臂最后一个 rank 日志里的 `EXIT n`（python 的真实退出码；CSV 另列 mpirun 自身的返回值）。带引擎世界的臂 mpirun 恒返回 1：进程不做 MPI_Finalize 即离开，prterun 记为异常终止（本机栈的拆解顺序定律），不表示训练失败。命中与回退为各 rank 日志中 OPRS_HIT / OPRS_FALLBACK 行的求和（训练结束后的评估若继续派发，其增量行也计入）。

✗ = 状态不是 ok 的臂（失败、超时被杀、计划要替换却一次没替换、站点形状不在计划里）：它的步长不进入任何比值——坏掉的臂不当作慢臂。

## 同批次配对比值（步长时间之比，> 1 = 更快；第 k 次该臂 对 第 k 次 serial）

| 臂 kind | 对 serial：各对 | 中位 | 最小..最大 |
|---|---|---|---|
| null | [0.993] | 0.9930 | 0.9930..0.9930 |
| oprs | [0.8854] | 0.8854 | 0.8854..0.8854 |
| always_triton_dist | [0.942] | 0.9420 | 0.9420..0.9420 |

OPRS 对 always_triton_dist（always_triton_dist 步长 / oprs 步长，> 1 = OPRS 更快）：[0.9399] → 中位 0.9399（0.9399..0.9399）
