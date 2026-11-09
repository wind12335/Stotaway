# 双机 16 卡（TP8 × DP2）

每台机器 8 卡组成一个 TP 组（重叠方法只作用于机内 TP 通信），跨机只做 DP 梯度同步（RCCL，走 RDMA）。
同一条命令在两个 worker pod 上各提交一次（共享文件系统）：

```bash
bash <repo>/e2e/dual16/vcjob.sh
```

- 默认：`T23_SHAPES=70b,27b`（一个作业连跑两个模型），每个模型 8 臂
  `BASE.1,NULL.1,ALT.1,PLAN.1,BASE.2,NULL.2,ALT.2,PLAN.2`，MBS 8，16 次迭代；约 2 小时 40 分 ≈ 42 卡时。
- PLAN 臂：BASE.1 跑完后，每个节点各自读 BASE.1 的整步时间，调用 `planner/planner.py` 选配置
  （报告写到 `<run>/<shape>/plan_node<N>.md`，两节点的决定必须一致）；ALT = 预测器没选的那个配置。
- RDMA：`enable_rdma.sh` 按平台方式启用（`SHCA_HOME` 指向 shca 资产目录）；成功判据是 rank 日志里出现
  `NET/IB : Using shca_*` / `via NET/IBext_v8/.../GDRDMA`，结果写 `rdma_verdict_node<N>.txt`。
  `RDMA_DISABLE=1` = TCP 对照。
- 进度（从另一台机器看共享目录）：`bash e2e/dual16/watch.sh`；汇总：`python3 tools/dual16_report.py <run 目录>`。
- 镜像缺包时把 site-packages 副本放到共享存储并设 `SW_PYSITE`（见 `3rdparty/README.md`）。
- 结果：`results/raw/dual16_runs/<rdma|tcp>_<job>/`；论文用的那次在 `results/raw/dual16/`。

## vLLM（LLaMA3-70B prefill，TP8 × PP2）

```bash
bash <repo>/e2e/dual16/vllm16_vcjob.sh          # 两个 worker 同一命令
```

- 形态：每台机器 8 卡 = 一个 TP 组（对称堆在机内，重叠方法只作用于机内 TP 通信）；两台机器是流水线的两段
  （PP2，节点 0 跑前 40 层、节点 1 跑后 40 层），段间激活走 RDMA。vLLM 的 rank 顺序 = pp_rank × 8 + tp_rank，
  所以 TP 组天然落在机内。
- 臂：`stock rccl dx plan` 各两遍（ABAB），与 8 卡 `e2e/vllm/run.sh` 同口径（gate 一致性 + ttft / ttft2 / thru）。
  跑完计时臂后自动跑自然文本一致性检查（`VS_GATE=0` 关掉）。默认约 1 小时 50 分 ≈ 30 卡时；先冒烟：`VS_ARMS="stock dx" VS_REPS=1`（约 35 分钟 ≈ 9 卡时）。
- 镜像没有 vLLM 时：在有 vLLM 的机器上先跑 `bash e2e/dual16/pack_vllm_env.sh`，生成
  `3rdparty/pyenv_vllm16.tar.gz`（vllm + flash_attn + DCU 垫片 + vLLM 运行时依赖，清单 `vllm_site_closure.txt`），
  作业里 `import vllm` 失败就自动解包到 `/tmp` 并加到 PYTHONPATH。
- 结果：`results/raw/dual16_runs/vllm_<rdma|tcp>_<job>/`（`vsp.csv`、`gate_*.json`、节点 0 的 `report.md`）。
