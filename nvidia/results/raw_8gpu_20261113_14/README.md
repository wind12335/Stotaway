# 八卡实验原始结果副本

该目录是 `/root/data/stowaway-nvidia/results` 中八卡实验的精简副本，保留算子计时 JSON/JSONL、预测器留一评估、对应推理结果 JSON、训练原始运行日志和模型形状派发验证。目录结构相对于原 `results/` 保持不变；代码及早期文档中出现的 `/root/data/stowaway-nvidia/results/...` 可以对应到此目录下的相对路径。原始失败日志未纳入；原始实验工作目录中的日志未改动。

- `tp8_dx_bf16_retest_20261114/`：99 点 BF16 DX 重测，保留每次的 stock、DX、样本、调度和检查数量。
- `tp8_dx_fp16_matrix_20261114/`、`tp8_sync_matrix_20261113/`：FP16 DX、Syncopate 和 stock 测量；原始采集来自不同进程/批次。
- `tp8_flash_matrix_20261113/`、`tp8_flash_missing_dispatch_20261114/`：原生和包含布局还原的派发计时口径不同；JSON 中的检查/计划与结果按原名保留。
- `tp8_*lomo*.json`：预测选择和事后 oracle；数据参与开发，不当作独立盲测。各候选批次及计时口径不同的具体说明见其文件和算子表。
- `raw/tp8_qwen36_*`、`tp8_20261111/vllm_bwmatch_*`：训练日志和推理结构化结果；Qwen 训练使用随机初始化与 mock 数据，Llama 使用真实权重。

此副本不包含模型权重、软件环境、第三方仓库和编译产物。结果文件中的绝对路径、软件版本和 GPU 信息保留原样，便于追溯，不能直接视为另一机器上可复现的运行环境。
