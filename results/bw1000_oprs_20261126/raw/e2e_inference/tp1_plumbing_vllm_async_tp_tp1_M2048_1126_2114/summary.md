# vLLM 预填充端到端：tp1_plumbing_vllm_async_tp_tp1_M2048_1126_2114（M = 2048；每格 = 该臂各遍的中位时间，单位 s）

| 臂 | kind | plan | init | ttft | ttft2 | thru | 实际预填 token（ttft / ttft2 / thru） | SP 前向 / 原版前向 | 命中 | 回退 | 载入 / 准备 (ms) | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | stock | - | 11.726 | 0.005 | 0.009 | 0.074 | 2048 / 4096 / 32768 | 0 / 0 | 0 | 0 | - / - | ok |
| b1 | always_async_tp | 2fcca8734b99 | 10.480 | 0.012 | 0.022 | 0.179 | 2048 / 4096 / 32768 | 25 / 36 | 400 fc1=async_tp:100,fc2=async_tp:100,proj=async_tp:100,qkv=async_tp:100 | 0 | 173.8 / 4.7 | ok |
| o1 | oprs | 58186d9fc17e | 8.951 | 0.008 | 0.015 | 0.121 | 2048 / 4096 / 32768 | 25 / 36 | 100 fc2=deo:100 | 300 | 182.5 / 16.6 | ok |
| r1 | sp_rccl | - | — | — | — | — | - / - / - | - / - | - | - | - / - | FAILED rc=1 (no row written) |
| d1 | all_deo | - | — | — | — | — | - / - / - | - / - | - | - | - / - | FAILED rc=1 (no row written) |

状态不是 ok 的臂（NO-SP-FORWARD：从未跑到 M-token 的 SP 前向；NO-HITS：计划替换了站点却一次没替换；FAILED：进程以非零码退出；incomplete：init 之后没有留下记录）不进入任何比值。比值一律用同批次配对。

## 比值（中位时间之比，> 1 = 更快；第 k 次该臂 对 第 k 次参照臂；多次时给出 中位 [最小..最大]）

| 臂 | 格 | 对 stock | 对 sp_rccl |
|---|---|---|---|
| s（stock） | ttft | — | - |
| b（always_async_tp） | ttft | 0.383 | - |
| o（oprs） | ttft | 0.571 | - |
| s（stock） | ttft2 | — | - |
| b（always_async_tp） | ttft2 | 0.405 | - |
| o（oprs） | ttft2 | 0.614 | - |
| s（stock） | thru | — | - |
| b（always_async_tp） | thru | 0.414 | - |
| o（oprs） | thru | 0.611 | - |

| 恒定选择臂 / OPRS | 格 | 比值（> 1 = OPRS 更快） |
|---|---|---|
| always_async_tp / oprs | ttft | 1.491 |
| always_async_tp / oprs | ttft2 | 1.517 |
| always_async_tp / oprs | thru | 1.474 |

（本目录没有自然文本检查文件 gatetext_*_M<M>.json：数值一致性未测；用 VS_GATE=1 另跑一批）
