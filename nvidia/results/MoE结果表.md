# MoE 结果表（NVIDIA A800 单机 8 卡）

## 1. 小型训练诊断

TP1/EP2，4 个数据并行组；2 层、8 专家、top2、hidden 1024、专家 FFN512、seq512、GBS8、BF16、mock data。每臂 20 步，去掉前 3 步取中位数，加速比 = 原版 / 候选。

| 臂 | ms/step | 对原版 | 状态 |
|---|---:|---:|---|
| 上游 NCCL EP2 | 50.1 | 1.000× | 正常完成 |
| Triton 顺序专家 GEMM，原版 dispatcher | 1887.0 | 0.0265× | 每 rank 40 次命中、0 fallback、无跳步或 NaN；未做梯度门槛 |

该候选仍读取主机 offsets 并逐次堆叠权重，不能等同原 BW1000 免同步整层 MoE。NVSHMEM EP 适配器仍只支持 world=2；EP8、真实路由偏斜、完整层前后向和 MoE 预测器均未完成。日志 `/root/data/stowaway-nvidia/results/raw/tp8_moe_{ep2_stock,gg}_r0_20261111.log`。
