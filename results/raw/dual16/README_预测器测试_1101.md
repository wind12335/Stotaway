# 双机 16 卡：预测器 + 重叠方法测试（11-01）

脚本：`bench/dual16/opus_dual16_vcjob.sh`（备份 `.bak_1101b`），两个 worker pod 上执行同一条命令。

## 测什么
16 卡 = TP8 × DP2（每台机器一个 TP 组，跨机只做 DP 梯度同步，走 RCCL/RDMA）。在这个设置下：
- 重叠方法（创新点 1）：fbag 相对原版、相对空对照的整步加速；
- 预测器（创新点 2）：PLAN 臂先读同节点 BASE.1 的整步时间，调用 `bench/planner/planner.py` 决定配置，再按这个配置跑；
  对照臂是预测器"没选"的那个配置，看预测器选的是不是更快。

## 臂
| 臂 | 是什么 |
|---|---|
| BASE | 原版（厂商 GEMM + RCCL 串行） |
| NULL | 空对照（环境、对称堆、包装层同 fbag，一个算子都不接管） |
| FBAG | RS 两处用重叠方法，AG 两处用原版 |
| FBC | 四处全用重叠方法 |
| FCQ / FCL | fbag + 只 QKV / 只 fc1 用重叠方法 |
| PLAN | 预测器选的配置（BASE / FBAG / FBC / FCQ / FCL 之一），选择和完整报告写到 `$OUT/plan_node<N>.md` |

## 提交命令（推荐先跑 70B，约 16 卡 × 40–55 分钟 ≈ 11–15 卡时）
```
# 70B MBS8：预测器预计选 FBC；对照 = FBAG（默认臂就是这一组）
T23_SHAPE=70b bash /root/private_data/tri-dist-hygon/bench/dual16/opus_dual16_vcjob.sh

# 70B MBS4（可选第二个点）：预测器预计选 FBAG；对照要换成 FBC
T23_SHAPE=70b MBS=4 TRAIN_ITERS=12 T23_ARMS=BASE.1,NULL.1,FBC.1,PLAN.1,BASE.2,NULL.2,FBC.2,PLAN.2 \
  bash /root/private_data/tri-dist-hygon/bench/dual16/opus_dual16_vcjob.sh
```
- 两个 pod 的环境变量必须完全一致。
- 想跳过预测器、直接指定配置：`T23_PLAN_CFG=FBC`。
- 27B（NL32）每臂约 11 分钟，8 臂约 25 卡时，建议放在 70B 之后再决定。

## 看结果
- 每臂每节点日志：`bench/results/opus_dual16/<rdma|tcp>_<job>/<ARM>/node<N>/out_r*.log`
- 预测器决定：`plan_node0.md`、`plan_node1.md`（两份必须一致；不一致会在 vcjob 日志里看到两个不同的 `predictor ... ->` 行）
- 判据：PLAN 两遍中位 < 对照两遍中位 ⇒ 预测器选对；FBAG / PLAN 对 NULL 的比值 = 重叠本身的收益。

## 已知风险
- 16 卡的每层周期 GPU 时间含跨机梯度同步，比 8 卡大一些；70B MBS8 离截止点 205 ms 很远（≈278 ms），选择稳定。
- 两个节点各自从本节点 BASE.1 日志读整步时间做决定；DP 同步下两边整步时间相同，只有正好落在截止点附近才可能不一致。

## 从 notebook pod 盯作业（11-01 加）
- vcjob 的每条日志同时写到 `<RUN_DIR>/status_node0.log`、`status_node1.log`；每臂结束一行带 `iters=` 和 `med(it3+)=`。
- 盯：`bash bench/dual16/opus_dual16_watch.sh`（自动取最新的 rdma_*/tcp_* 目录，每 60 s 打印变化；两节点都 T23 DONE 后退出）。
- 汇总：`python3 bench/dual16/opus_dual16_report.py <RUN_DIR>`（每臂中位、对 BASE / NULL、PLAN 对预测器没选的那个配置）。
- 注意：两个 pod 之间的共享存储不是实时的（10-26 实测新文件最长约 10 分钟才可见），notebook 侧看到的进度可能滞后几分钟；判活看 r7（rank 7 打印迭代行），不看 r0。
- 预测器在 pod 里跑不起来时（例如缺 numpy），脚本用 notebook 上算好的选择兜底（70b MBS8 → FBC，70b MBS4 → FBAG，27b MBS8 → FBAG，27b MBS4 → BASE），并在 status 里打 [warn]。

## 11-01 晚更新（以此为准）
- 一次提交跑完两个模型：默认 `T23_SHAPES=70b,27b`（先 70B 后 27B），每个模型 8 臂：BASE.1 NULL.1 ALT.1 PLAN.1 BASE.2 NULL.2 ALT.2 PLAN.2。
  ALT = 预测器没选的那个配置（PLAN 选 FBC → ALT 跑 FBAG；PLAN 选 FBAG → ALT 跑 FBC）。
- 规模与 8 卡相同：同样的模型参数（27B NL32 / 70B NL8，seq 4096，MBS8，TP8+SP，full recompute），每个 DP 组 8 个 micro-batch，16 次迭代；只多了 DP=2 跨机梯度同步。
- 结果目录：`<RUN_DIR>/70b/<臂>/node<N>/`、`<RUN_DIR>/27b/...`；RDMA 判决 `<RUN_DIR>/rdma_verdict_node<N>.txt`（第一臂开 NCCL_DEBUG=INFO 只看 INIT,NET）。
- 修了一个旧 bug：TRAIN_ITERS 之前没 export，内层会用默认 25 次迭代。
- 提交命令（两个 worker 都执行同一条）：`bash /root/private_data/tri-dist-hygon/bench/dual16/opus_dual16_vcjob.sh`
