# 外部后端的 A800 实验适配

此处保存八卡实验使用的补丁，避免结果依赖数据盘上的未版本化工作副本。源仓库请使用各自许可，补丁是基于指定提交生成的修改；本项目没有把外部仓库完整引入。

- Syncopate 基于 `7892aaf3e8404d651bfc74e5ec9765744be7dee1`。在一个独立工作副本里使用 `git apply /path/to/nvidia/patches/syncopate_7892aaf_a800.patch`。`tests/__init__.py` 在实验工作副本中是空文件；补丁把测试脚本所需的导入路径、rank 生成文件与本机 SM 数做了适配。当前接入器将该工作副本 `tests` 加入 Python 搜索路径。
- FlashOverlap BF16 基于 `89100a68080d4caa210bc8030d65fa96cbf53386`。在独立工作副本里使用 `git apply /path/to/nvidia/patches/flashoverlap_89100a6_bf16_a800.patch`，然后按上游方式编译。该补丁用于实验 BF16 GEMM 和 NCCL；不应覆盖原 FP16 构建。四个实际模型 RS 形状及一个派发验证形状的 A800/TP8 搜索配置保存在 `../configs/flashoverlap_bf16_a800_tp8/`，复制到工作副本 `configs/`；配置与设备和内核构建绑定，需要在其他环境验证。
- Triton-distributed 的本次实验源码版本为 `552fedcb04d0caf909f6ac165594d2afbe301b88`；Megatron-LM 为 `571370c829ca768fe37244f4e2e7f28d8accc4ab`，vLLM 版本为 0.11.0。依赖本项目脚本中指定环境和 NVSHMEM，使用前请检查实际路径与版本。

`git apply --check` 已在上述干净基线上检查两份补丁；本次八卡运行使用的是数据盘上的原工作副本，补丁文件没有在当前单卡重新编译或复跑八卡。实验脚本中的绝对路径是本次机器路径，需要按实际部署环境设置。
