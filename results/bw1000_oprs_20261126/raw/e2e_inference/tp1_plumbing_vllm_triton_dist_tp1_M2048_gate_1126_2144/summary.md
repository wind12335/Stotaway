# vLLM 自然文本检查：tp1_plumbing_vllm_triton_dist_tp1_M2048_gate_1126_2144（M = 2048；本目录只跑了 VS_GATE=1，没有计时格）

| 臂 | kind | plan | 自然文本提示耗时 (s) | 实际预填 token | SP 前向 / 原版前向 | 命中 | 回退 | 载入 / 准备 (ms) | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| stock | stock | - | 0.274 | 2048 | 0 / 0 | 0 | 0 | - / - | ok |
| sp_rccl | sp_rccl | 63c452f2af46 | 0.275 | 2048 | 6 / 191 | 0 | 96 | 186.3 / 53.2 | ok |
| all_deo | all_deo | dce7031f0e8e | 0.270 | 2048 | 6 / 191 | 96 fc1=deo:24,fc2=deo:24,proj=deo:24,qkv=deo:24 | 0 | 190.9 / 16.7 | ok |
| always_triton_dist | always_triton_dist | 67320f65685b | 0.287 | 2048 | 6 / 191 | 96 fc1=triton_dist:24,fc2=triton_dist:24,proj=triton_dist:24,qkv=triton_dist:24 | 0 | 187.5 / 33.3 | ok |
| oprs | oprs | cd100a3b4e71 | 0.282 | 2048 | 6 / 191 | 72 fc1=triton_dist:24,fc2=deo:24,qkv=triton_dist:24 | 24 | 186.5 / 47.2 | ok |

## 自然文本输出检查（M = 2048，每臂 4 个提示 × 48 个贪心 token，对 stock）

| 臂 | 与 stock 完全相同的 token | 各提示第一个不同位置 | 首 token logprob 差 | 相同前缀上 max logprob 差 | 提示各位置 logprob 差 max / 平均 |
|---|---|---|---|---|---|
| all_deo | 192/192 | 无 / 无 / 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |
| always_triton_dist | 192/192 | 无 / 无 / 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |
| oprs | 192/192 | 无 / 无 / 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |
| sp_rccl | 192/192 | 无 / 无 / 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |

提示各位置 logprob 差 = stock 与该臂各自模型对同一个提示 token 给出的 logprob 之差，与贪心 token 是否打平无关。带随机权重的小模型只能说明算术与通路，不代表任何真实模型。
