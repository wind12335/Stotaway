# Megatron 训练端到端：tp1_smoke_oprs_async_tp_v4（TP=1，每臂独立进程；步长中位数自第 3 步起）

| 臂 | kind | plan | 退出码 | 步数 | 步长中位 (ms) | 训练 loss | 验证 loss | 最大已分配 (MB) | 命中（站点=选择:次数） | 回退（原因:次数） | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n1 | null | null.json | 0 | 8 | 407.3 | 1.182455E+01 | 1.161879E+01 | 15855.44 | - | plan:512 | ok |
| o1 | oprs | oprs_async_tp.json | 0 | 8 | 417.7 | 1.182455E+01 | 1.161879E+01 | 15855.44 | fc2=deo:128 | plan:384 | ok |
| d1 | always_deo | always_deo.json | 0 | 8 | 436.3 | 1.182455E+01 | 1.161879E+01 | 15871.44 | fc1=deo:128 fc2=deo:128 proj=deo:128 qkv=deo:128 | - | ok |
| a1 | always_async_tp | always_async_tp.json | 0 | 8 | 408.5 | 1.182455E+01 | 1.161879E+01 | 15815.44 | fc1=async_tp:128 fc2=async_tp:128 proj=async_tp:128 qkv=async_tp:128 | - | ok |
| s1 | serial | - | 0 | 8 | 397.25 | 1.182455E+01 | 1.161879E+01 | 15815.44 | - | - | ok |
| o2 | oprs | oprs_async_tp.json | 0 | 8 | 418.3 | 1.182455E+01 | 1.161879E+01 | 15855.44 | fc2=deo:128 | plan:384 | ok |
| s2 | serial | - | 0 | 8 | 397.8 | 1.182455E+01 | 1.161879E+01 | 15815.44 | - | - | ok |

退出码 = 每个臂最后一个 rank 日志里的 `EXIT n`（python 的真实退出码；CSV 另列 mpirun 自身的返回值）。带引擎世界的臂 mpirun 恒返回 1：进程不做 MPI_Finalize 即离开，prterun 记为异常终止（本机栈的拆解顺序定律），不表示训练失败。命中与回退为各 rank 日志中 OPRS_HIT / OPRS_FALLBACK 行的求和（训练结束后的评估若继续派发，其增量行也计入）。

✗ = 状态不是 ok 的臂（失败、超时被杀、计划要替换却一次没替换、站点形状不在计划里）：它的步长不进入任何比值——坏掉的臂不当作慢臂。

## 同批次配对比值（步长时间之比，> 1 = 更快；第 k 次该臂 对 第 k 次 serial）

| 臂 kind | 对 serial：各对 | 中位 | 最小..最大 |
|---|---|---|---|
| null | [0.9753] | 0.9753 | 0.9753..0.9753 |
| oprs | [0.951, 0.951] | 0.9510 | 0.9510..0.9510 |
| always_deo | [0.9105] | 0.9105 | 0.9105..0.9105 |
| always_async_tp | [0.9725] | 0.9725 | 0.9725..0.9725 |

OPRS 对 always_deo（always_deo 步长 / oprs 步长，> 1 = OPRS 更快）：[1.0445] → 中位 1.0445（1.0445..1.0445）

OPRS 对 always_async_tp（always_async_tp 步长 / oprs 步长，> 1 = OPRS 更快）：[0.978] → 中位 0.9780（0.9780..0.9780）
