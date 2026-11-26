# vLLM 自然文本检查：tp1_plumbing_vllm_async_tp_tp1_M2048_gate_1126_2207（M = 2048；本目录只跑了 VS_GATE=1，没有计时格）

| 臂 | kind | plan | 自然文本提示耗时 (s) | 实际预填 token | SP 前向 / 原版前向 | 命中 | 回退 | 载入 / 准备 (ms) | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| stock | stock | - | 0.275 | 2048 | 0 / 0 | 0 | 0 | - / - | ok |
| always_async_tp | always_async_tp | 2fcca8734b99 | 0.281 | 2048 | 6 / 191 | 96 fc1=async_tp:24,fc2=async_tp:24,proj=async_tp:24,qkv=async_tp:24 | 0 | 186.4 / 4.9 | ok |
| oprs_deo_only | oprs_deo_only | 3b078d2ebeb7 | 0.279 | 2048 | 6 / 191 | 24 fc2=deo:24 | 72 | 208.8 / 16.5 | ok |
| oprs | oprs | 58186d9fc17e | 0.281 | 2048 | 6 / 191 | 24 fc2=deo:24 | 72 | 183.4 / 15.8 | ok |

## 自然文本输出检查（M = 2048，每臂 4 个提示 × 48 个贪心 token，对 stock）

| 臂 | 与 stock 完全相同的 token | 各提示第一个不同位置 | 首 token logprob 差 | 相同前缀上 max logprob 差 | 提示各位置 logprob 差 max / 平均 |
|---|---|---|---|---|---|
| always_async_tp | 192/192 | 无 / 无 / 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |
| oprs | 192/192 | 无 / 无 / 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |
| oprs_deo_only | 192/192 | 无 / 无 / 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |

提示各位置 logprob 差 = stock 与该臂各自模型对同一个提示 token 给出的 logprob 之差，与贪心 token 是否打平无关。带随机权重的小模型只能说明算术与通路，不代表任何真实模型。
