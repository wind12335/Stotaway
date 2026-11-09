# Opus 2026-10-29 晚 — 实验发现（随测随更）

> 本文件是开发期逐条实验记录（F01–F55），原样保留。文中的路径指开发仓库（`tri-dist-hygon/bench/...`、`tri-dist-hygon-opus`），在本仓库中对应：脚本 → `bench/ probes/ e2e/`，原始日志 → `results/raw/`，代码 → `src/stowaway/`，预测器 → `planner/`。

所有数字来自本目录 `opus_*` 原始文件；正式 a2_ops_matrix_8card.csv 未改动。

## F1 a2 AG/AR 污染格重测（fresh process / 每模型独立 mpirun）
- `opus_a2_ag_fresh.csv`: AG(v12ring) 全格输 0.16–0.94×（llama3-8b/qwen3-32b M32768 0.94 最好），
  stock busbw 48–68 GB/s 健康。旧表 3.41× / 2.43× / 1.77× 等全部为污染窗假象。
- `opus_a2_rerun.csv`: 同进程跑 4 模型 → 第 13 格起 stock 塌到 5–19 GB/s（进程内退化，
  非硬件窗）。**定律：a2 型长进程 bench 必须每模型新进程**。
- AR(v14) 全格输（干净格 0.03–0.36×）。

## F2 fabric 置换律（`opus_perm.log`）
全 8 rank 同时引擎 push 64MB：shift_j（r→r+j，j=1..7）每 rank 139.6–142.7 GB/s 与 j 无关；
seq7（单流 7 块 7 个不同对端）152；ring7（7 块只发邻居）152；fan7（7 流并发）144。
拓扑 = HSW 交换机 1 跳全连。⇒ 17–25 GB/s "单播扇出墙"对引擎置换 push 不存在。

## F3 SM 跨卡写（`opus_smx.log`）
SM push 进 peer 堆 80 CTA 108.9 GB/s、pull 99 GB/s，3 轮逐轮变异位级 0 错。
"SM store 不落地"定律作废（S75d 的 kernel_copy_bf16 被 warmup 把 n 特化成 1）。

## F4 直连交换 RS（DX）vs 环
### 纯 RS（`opus_dxrs_pure.csv`，±1 整数载荷位级 md==0 全臂全格，2 run/格）
| MB | RCCL iso | ring1 iso | ring4 iso | DX iso | DX vs RCCL | DX vs ring1 |
|---|---|---|---|---|---|---|
| 33.6 | 0.499 | 1.156 | 1.176 | 0.865 | 0.58× | 1.34× |
| 134 | 1.611 | 1.485 | 1.667 | 1.272 | 1.27× | 1.17× |
| 268 | 3.139 | 2.533 | 2.771 | 2.162 | 1.45× | 1.17× |
| 403 | 4.651 | 3.620 | 3.885 | 3.048 | 1.53× | 1.19× |
| 537 | 6.150 | 4.647 | OOM | 3.917 | 1.57× | 1.19× |
DX busbw 92→120 GB/s；单 op host 0.85–1.0ms（ring 1.12–1.22ms）。

### GEMM+RS（`opus_dx_rs.csv` + `opus_dx_rep.csv`，3 次 fresh-process run 级中位）
DX 赢 v9ring：70b M8192 1.148 vs 0.877，M32768 1.312 vs 1.191；405B 全 M 段 1.03–1.06×；
72B M8192/32768 赢。**DX 输 v9ring：M16384（70b 0.889 vs 1.102，72b 0.867 vs 1.054）与
M4096**——待 probe 分解（dx − gemm8 在 M16384 = +2.16ms vs M8192 +0.46ms，非单调 = 疑 sum
核/slot 带宽或 GEMM 争用，非传输）。
头条复现：M8192 N28672 K8192 stock 14.257 / v9ring 13.030（1.094×，与旧 13.07 吻合）/
**DX 12.341（1.155×）**。

## F5 车辆（进行中，`opus_veh/driver.log`）
27B NL32 MBS8，BASE / K4 冠军 / DX(defer, P=2)，ABAB，12 iter。

### F5 车辆中间结果（opus_veh/driver.log，27B NL32 MBS8，12 iter，iter3+ 中位）
| 臂 | 形态 | ms/iter | vs BASE |
|---|---|---|---|
| B1 | stock | 36531.5 | — |
| K1 | ring K4 冠军（defer, MIN_MB128） | 36188.5 | −0.94% |
| D1 | DX K=1（defer, P=2） | 36452.6 | −0.22% |
- D1 fallback=0，val loss 0.18654 vs BASE 0.18660 / K1 0.18665（bf16 归约序带内）。
- 同深度对照：ring K1 在 F2b（R_K1）= 37653.8（+3.1% 慢于 base，跨批次仅作方向）⇒ DX K1 比 ring K1 快 ~3.2%。
  车内 K 深度是主旋钮（F2 定律），DX 需要 K 槽 → opus_dxvk（K lanes, P=max(2,K) 槽集, 槽集/ack 按 (p,home) 分格）已写，排队门控后上车。

### F5 车辆 ABAB 终表（B1 K1 D1 B2 D2 K2，同批，fresh mpirun，12 iter，iter3+ 中位）
| 臂 | run1 | run2 | 均值 | vs BASE |
|---|---|---|---|---|
| BASE | 36531.5 | 36664.4 | 36598.0 | — |
| K4 环冠军 | 36188.5 | 36258.4 | 36223.5 | −1.02% |
| DX K=1 | 36452.6 | 36189.2 | 36320.9 | −0.76% |
- BASE 两轮差 133ms = 本批噪声尺度。K4 两轮都赢（成立）；DX 对 K4 配对 +0.73% / −0.19%，均值差 0.27% < 噪声 ⇒ **DX K=1 ≈ K4 环**（环 K=1 跨批 +3.1% 慢于 BASE，仅方向）。
- 全臂 fallback=0；SOAK2B "stock 速度" 现象本批未复现。
- 待：DX K=4（opus_dxvk）位级门 → 车辆 DK1/DK2。

### F6 DX GEMM+RS 分解（opus_dx_probe.log，臂序 stock→gemm8→dx，**无 v9ring 前置**）
| 形状 | stock | gemm8 | DX | DX× | rank0: GEMM(DX 内) / 等待 / 求和 |
|---|---|---|---|---|---|
| 70b M8192 | 2.648 | 1.857 | 2.300 | 1.152 | 2.08 / 0.02-0.05 / 0.146 |
| 70b M16384 | 5.147 | 3.624 | 4.305 | **1.196** | 3.94 / 0.02-0.05 / 0.291 |
| 70b M32768 | 10.141 | 6.455 | 7.723 | 1.313 | 7.07 / 0.02-0.07 / 0.582 |
| 72b M8192 | 3.329 | 2.722 | 3.054 | 1.090 | 2.86 / 0.023 / 0.146 |
| 72b M16384 | 6.544 | 5.321 | 5.940 | **1.102** | 5.61 / 0.023 / 0.291 |
| 72b M32768 | 12.631 | 9.461 | 10.361 | 1.219 | 9.72 / 0.022 / 0.581 |
- **撤回** F4 的"DX 输 M16384"（0.889/0.867）：rep 跑里 v9ring 臂在同进程先于 DX，污染后续臂（同 a2a→DX 现象）。⇒ 同进程臂序律：先建/拆自有对称堆 ctx 的臂会拖慢下一个引擎臂。公平对比 = 每臂独立进程（opus_dx_iso 排队中）。
- **DX 墙 = GEMM×(1+并发拷贝干扰 10-14%) + 一次求和尾巴（0.146ms/16.8MB 分片，线性）**；到达等待仅 20-70µs ⇒ 通信全藏。
- 下一刀：①7 块在本地块 GEMM 期间预求和 → 尾巴只剩 2 输入 add（估 −0.1ms@M8192 / −0.4ms@M32768）；②限制拷贝引擎并发压 GEMM 干扰。

### F7 纯 RS @335.5MB / 536.9MB（opus_dxrs_a2a.log，同进程臂序 rccl→ring1→a2a→dx→dxinc）
- 旧 a2a（rs_a2a_op）iso 3.958 / 6.182ms ≈ RCCL；DX 改 incast 发送序（d=0..R-1）比置换序慢 +10~19% iso / +30~70% pipe
  ⇒ R12e "a2a 撞扇出墙" 的真因 = 多对一发送序 + 7 次分开的 add，不是直连本身。
- 该跑 DX 536.9MB 4.604ms vs 先前独立跑 3.917ms（RCCL/ring1 复现 ±0.1%）⇒ 同进程 a2a 前置污染（与 F6 同律），复核排队中（opus_dxrs_contam：dx→rccl→ring1→a2a→dx）。

### F8 DX-AG（Megatron-SP AG+GEMM 直连，opus_dx_ag.log，每 (模型, up/gate_up) 独立进程，臂序 stock→gemm1→dxag）
| 模型 / 形 | M4096 | M8192 | M16384 | M32768 |
|---|---|---|---|---|
| 70b up (K8192 Nc3584) | 0.66 | 0.95 | **1.08** | **1.21** |
| 70b gate_up (Nc7168) | 0.80 | 0.92 | **1.10** | **1.12** |
| qwen3-32b up (K5120 Nc1280) | 0.46 | 0.68 | 0.98 | 0.98 |
| qwen3-32b gate_up (Nc2560) | 0.46 | 0.69 | **1.09** | **1.08** |
| llama3-8b up (K4096 Nc1792) | 0.43 | 0.59 | 0.90 | **1.02** |
| llama3-8b gate_up (Nc3584) | 0.44 | 0.70 | 0.86 | **1.11** |
- 全格 CLEAN。**首个干净的 8 卡 AG 胜格**（旧 v12ring 干净重测全输 0.16–0.94）：70B M≥16384 1.08–1.21×。
- 赢区 = GEMM 时间 ≥ 传输时间；小 M 输在固定成本（~0.9-1.1ms：7 次入队 + staging 拷贝在计算流上串行）。
- 下一刀已排队（opus_dx_ag2：staging 搬到侧流，与上一块 GEMM 重叠）。

### F9 车辆 DX K=4（opus_dxvk，位级门 burst 400 + modes 300 全 PASS 后上车）
- DK1 = **35845.6 ms/iter**（fallback=0）：vs BASE 均值 36598.0 = **−2.06%**；vs K4 环均值 36223.5 = **−1.04%**（同晚，但非严格 ABAB 交错，DK2 复测中）。

### F9b 车辆 DX K=4 两轮
| 臂 | run1 | run2 | 均值 | vs BASE |
|---|---|---|---|---|
| DX K=4 | 35845.6 | 36148.0 | 35996.8 | −1.64% |
- vs K4 环均值 36223.5 = −0.63%（约 1.7× BASE 两轮差 133ms；非严格 ABAB 交错，方向成立、幅度弱）。全臂 fallback=0，val loss 0.18654 ≈ BASE 0.18660。

### F10 臂序污染复核（opus_dxrs_contam.log，单进程 dx→rccl→ring1→a2a→dx）
| 载荷 | DX 首位 | RCCL | ring1 | a2a | DX 末位（a2a 之后） |
|---|---|---|---|---|---|
| 335.5MB | 2.625 | 3.907 | 3.055 | 4.354 | 3.047（+16%） |
| 536.9MB | 3.933 | 6.143 | 4.648 | 6.698 | 4.542（+15%） |
- **同进程臂序污染定律成立**：先跑的建拆对称堆臂拖慢后续引擎臂 15-16%；RCCL/ring1 不受影响。干净 DX 335.5MB 2.625ms = RCCL 的 1.49×、ring 的 1.16×。

### F11 X2 偏斜×消费序（opus_dx_agskew3.log，全 rank 取最慢者计时，70B gate_up M8192，victim=rank3，2 轮）
| δ (µs) | stock | DX 固定序 | DX oracle 序（victim 最后吃） |
|---|---|---|---|
| 0 | 4.45 | 4.78–4.86 | 4.83–4.87 |
| 500 | 4.88 | 4.84–4.85 | 4.81 |
| 1000 | 5.38 | 4.80–4.87 | 4.80–4.88 |
| 2000 | 6.37 | **5.70–5.72** | **4.85** |
- stock 斜率 ≈ 0.96 µs/µs（偏斜几乎全额转嫁）。
- DX 固定序：δ ≤ 1000 全吸收，δ = 2000 曝露 0.86ms ⇒ 松弛窗 W_fixed ≈ 1.1ms。
- DX oracle 序：δ = 2000 仍为 0 曝露 ⇒ 把落队块排到最后，W 扩到 ≈ 剩余 6 块 GEMM（≈2.5ms）。
- **两轮完全复现，差 0.86ms = 15%**。⇒ 曝露 = max(0, δ − W(order))；预测器的作用 = 选顺序把 W 从 W_fixed 扩到 W_oracle。δ 落在 (W_fixed, W_oracle) 区间时预测器有用。车辆真实 δ 分布待 X3。

### F12 X1 无栅栏 DX vs 每 op 全员栅栏 DX（async-TP 式）vs RCCL（opus_x1_skew.log，纯 RS 335.5MB，victim 前置 δ，全 rank 取最慢者，2 轮×W∈{4096,8192}）
| δ (µs) | RCCL 曝露 | DX 无栅栏 | DX+栅栏 |
|---|---|---|---|
| 250 | 219–228 | **1314–1625** | 232–235 |
| 500 | 454–463 | **889–952** | 484–489 |
| 1000 | 951–962 | **1515–1672** | 977–984 |
| 2000 | 1944–1957 | 1900–1917 | 1972–1976 |
| 4000 | 3930–3944 | 3766–3788 | 3959–3965 |
- δ=0：DX 3.007ms vs RCCL 4.33ms（1.44×），栅栏版 3.046ms（栅栏本身只花 40µs）。全格 md=0。
- **新现象：无栅栏置换调度在中等偏斜下脆弱**——δ∈[250,1000]µs 时曝露超出 δ 0.4–1.4ms；δ≥2ms 时反而 < δ 3–5%。栅栏把曝露恰好压回 δ（脆弱消失，但永远全额付 δ）。
- 假说：迟到 rank 的推送与其余 rank 的置换相位错开 → 同一目的地被两路同时写 → 退化成 incast（F7：incast 慢 10–19% iso / 30–70% pipe）。验证 = X1b dxrot（迟到者从"此刻无人写入"的步 j0=1+δ/t_step 接入置换，跳过的目的地殿后）。

### F13 X3 车辆到达偏斜遥测（opus_x3/，27B DX K=4 + 观测核，12 iter，8 rank × 18432 op = 147456 op）
- DT1 = 35828.3 ms/iter，fallback=0，val loss 0.1865405 = DK1 逐位同值 ⇒ 观测核零扰动。DX K=4 三轮：35845.6 / 36148.0 / 35828.3，均值 35940.6 = BASE −1.80%，K4 环 −0.78%。
- 每步典型到达延迟线性 327→2635µs（≈0.37ms/步 ≈ 113GB/s/块），即置换序结构本身。
- **原始"最后到达者"= r+1 占 97–99%，W8 粘性命中 0.97–0.99 —— 纯结构假象**（置换序里 r+1 永远第 7 步）。照搬 i1_predictor 的 victim 投票会以 99% 置信度"预测"一个毫无信息的落队者。⇒ 预测器必须先扣步序。
- **扣步序后的真实偏斜**：p50 156µs / p90 555 / p99 1686。69.4% < 250µs，27.3% ∈ [250,1000)，2.9% ∈ [1000,2500)，0.4% ≥ 2500。
- **没有粘性落队者**：残差最迟者占比最高 0.19–0.24（均匀 = 1/7 = 0.14），W8 粘性 0.16–0.19。
- **E-P2 预注册门 G0b（粘性 ≥ 60%）FAIL**。车辆（mock 数据、同形 op）里 victim 预测没有燃料，与 R12 "mock 数据无偏斜" 一致。
- X2 的消费序窗口（W_fixed≈1.1ms）覆盖 96.7% 的 op ⇒ 顺序优化在本车辆最多触及 ~3% op。
- 唯一开放点：27% op 落在 X1 脆弱带 [250,1000)µs（X1 是单 rank 注入 + 孤立 op，口径不同）；X1b 判定脆弱机理后再评估。

### F14 X1b 相位接入（dxrot）验证撞车假说（opus_x1b.log，纯 RS 335.5MB，W=8192，2 轮，t_step 实测 330µs，全格 md=0）
| δ (µs) | DX 原序曝露（2 轮均值） | DX 相位接入（2 轮均值） | 回收 |
|---|---|---|---|
| 250 | 1428 | **315** | −1113 |
| 500 | 937 | 843 | −94 |
| 750 | 1568 | 1074 | −494 |
| 1000 | 1532 | 1052 | −480 |
| 1500 | 1993 | 1380 | −613（< δ） |
| 2000 | 1906 | 1813 | −93 |
- **撞车假说成立**：只改迟到 rank 的起始步（j0=1+round(δ/t_step)），中等偏斜的超额曝露大部分消失，δ≥1ms 后曝露低于 δ。
- δ=500 回收弱：δ/t_step≈1.5，round 选步落在两步中间 —— 需要连续相位（子块粒度）或更细的选步规则。
- 意义：预测器的正确角色 = **相位对齐**（由观测到的迟到量 δ 推出接入步），不是 victim 身份预测（F13：车辆无粘性 victim）。所需信息（自己迟到多久）可在设备侧直接测得，不需历史投票。
- 待做：车辆内实现（迟到量 = 本 rank fork 时刻 vs 同 op 对端 ticket 首达时刻），车辆 27% op 落在 [250,1000)µs 是其燃料上界。

### F15 X5 车辆前向融合 row-parallel GEMM+RS（DX 形态，复制版引擎 /root/private_data/tri-dist-hygon-opus，opus_vehf/driver.log，27B NL32 MBS8，12 iter，iter3+ 中位，每臂 fresh mpirun，臂序 fall1 fdk1 base1 ffc2a fall2 fdk2 base2）
| 臂 | 轮 (ms) | 均值 | vs base | vs fdk |
|---|---|---|---|---|
| base | 36636.6 / 36661.1 | 36648.9 | — | +1.51% |
| fdk = DX-BRS K4 (复制版) | 36027.2 / 36177.8 | 36102.5 | −1.49% | — |
| fall = fdk + o_proj/fc2 前向融合 | 37647.3 / 37772.0 | 37709.7 | +2.89% | **+4.45% (输)** |
| ffc2 = fdk + 仅 fc2 融合 | 36884.8 (1 轮) | — | +0.64% | +2.17% (输) |
- 融合真走: [mgd] early stats fused=12288 fallback=0 (=32层×2×8mb×12it×2(full recompute)); BRS mappings 类 12384→96。test loss 0.20226-0.20236 全臂一致。
- 门测 1.55-1.78× (opus_fused_gate.log) 的对照是 GEMM+RCCL RS, 车里真对照是 GEMM+DX RS ⇒ 门测赢幅不可外推。
- 车内单 op 计时 (opus_vehf2/ftim1, compute 流事件): opwall 中位 3.35ms (39 次 flush 均值, r0/r5 相同); host 入队中位 20.3ms (一半 flush >10ms, 另一半 ~1.8ms) ⇒ host 入队远超设备时间, 融合 op 是同步式(末尾 wait 全部到达+lane join 回 caller 流), host 被 hipMemcpyAsync/WaitValue 背压阻塞。
- 反向缺陷: DX 反向 grad_weight 用 full.transpose().contiguous() (多一次 335MB 拷贝), 原版是 grad_output.t().matmul(); 修复开关 OPUS_DX_BWD=t, 验证见 opus_vehf3。
- **纯前向（Megatron evaluate 计时器，max over ranks，test-set 那次 = 2 eval iter × 8 mb，无 recompute；opus_vehf/*/out_r7.log）**：fall 17769.5 / 17849.5，fdk 19235.7 / 19262.0，base 20023.1 / 20031.1，ffc2 18313.2 ⇒ **融合前向 vs fdk −7.4%，vs base −11.1%**（两轮极差 <0.5%）。即融合算子在前向里真赢，训练步输 1.6s 全部来自反向/recompute 段。
- 车内单 op（compute 流事件）：融合 opwall 3.35ms vs fdkt（stock GEMM + DX mapping RS）5.09ms ⇒ 每次省 1.74ms × 1024 次/iter（512 前向 + 512 recompute）≈ 1.78s/iter 的设备收益；host 入队两者同量级（20.3 vs 17.0ms 中位）⇒ 入队不是区别项。训练步净输 1.6s ⇒ 反向段多花 ~3.4s，待 fallt（去转置拷贝）和后续反向剖析定位。
- **gate2（opus_fused_gate2.log，同形，对照 = GEMM + DX RS）**：K2176 stock 6.269 / gemm+dx 5.217 / fused 3.507 ⇒ 1.49×（b2b 16 op 1.54×）；K640 4.888 / 3.979 / 3.241 ⇒ 1.23×（b2b 1.34×）。融合对公平对照仍真赢。（rc=139 = 结果全出后的 teardown segfault）
- **反向 wgrad 形态（同 log）**：转置 contiguous 拷贝本身 4.585ms；K2176 NN(拷贝) 8.716 vs TN(view, 原版) 2.347（3.7×），K640 5.934 vs 0.825（7.2×）。×256 次/K/iter ⇒ ~2.94s/iter 额外开销，扣前向收益 ~1.78s ⇒ 预测净 +1.16s，实测 fall−fdk = +1.61s ⇒ 训练步输主因 = 我写的反向缺陷，非融合机理。修复臂 fallt 见 opus_vehf3。
- **fallt1（融合 + 原版 TN wgrad，OPUS_DX_BWD=t；opus_vehf3/driver.log）= 34532.4 ms，fallback 0，test loss 0.2022966** ⇒ vs fdk 均值 36102.5 −4.35%，vs base 均值 36648.9 −5.78%（单轮；fdk3 / fallt2 同批待补，跨批对比须以同批 fdk3 为准）。
- 同批 fdk3 = 36009.6 ms（fdk 三轮 36027.2/36177.8/36009.6，极差 0.47%）⇒ fallt1 vs 同批 fdk3 **−4.10%**。fallt2 待补。
- **fallt2 = 34560.5 ms**，fallback 0，test loss 0.2022311。同批 ABA（fallt1 34532.4 / fdk3 36009.6 / fallt2 34560.5）：fallt 均值 34546.5 ⇒ **vs 同批 fdk3 −4.06%**，vs fdk 三轮均值 36071.5 −4.23%，vs base 两轮均值 36648.9 −5.74%。两轮 fallt 极差 0.08%。
- **X5 判决：GO。** 前向融合 row-parallel GEMM+RS（DX 分块，7 个远端块按置换序先算先推、own 块最后、home 一次 fp32 求和）在 27B 车里真赢，叠加在 DX-BRS 之上再省 ~1.5s/iter。
- **反向门（opus_bwd_gate.log，dx_ag_bwd_op = DX 置换全收集 grad_out + 每块到达即 dgrad + TN wgrad）**：64 op 背靠背逐 op 变异，max|full−ref|=0、max|gw−ref|=0、rel(gi)=0 ⇒ **位级 PASS**。计时 vs stock(RCCL AG + dgrad + TN wgrad)：K2176 7.560→6.087 **1.24×**（b2b 1.17×）；K640 4.056→3.538 1.15×（**b2b 1.006× 持平**）。rc=139 = 结果全出后 teardown segfault。车辆验证见 opus_vehf5（fbag/fallt/fbag 同批）。
- **分相位（opus_vehf4，均为 TN wgrad）**：ffwd1（只融合首遍前向，no_grad 段）= 35127.4 ms，fallback 0，test loss 0.2022311 ⇒ vs 同批外 fdk3 36009.6 省 882 ms（−2.45%）；全融合 fallt 均值省 1463 ms。frec1 待补（可加性预测 ≈35430）。

### F16 X6 车内相位对齐可行性（opus_x3/ 遥测复算，27B DX K=4，8 rank × 18432 op，扣步序 + 扣每 op 公共平移后的每发送者迟到残差）
- 迟到残差 op→op 自相关 lag1 中位 −0.011（min −0.222 / max 0.138），lag4 0.050；P(下一 op 迟到>250µs | 本 op 迟到) = 0.072–0.122，基线率 0.083 ⇒ **迟到不跨 op 持续，历史无预测力**。
- 推论：host 侧按历史选接入步 j0（X1b 的 dxrot 形态）在本车辆没有燃料 —— host 入队领先设备，选 j0 时只能用历史。只有"设备侧当场测迟到、当场改发送序"可能有效，但引擎 push 序在 host 入队时固定，设备侧改序须换 SM 推送（109 vs 141 GB/s，F1），先付 23% 带宽。
- **X6 判决：host 侧相位对齐 NO-GO；设备侧变体 PARK**（燃料上界 = F13 的 27% op ∈[250,1000)µs，且 K=4 多 op 在途已部分吸收）。
- frec1（只融合 recompute 段）= 36027.9 ms，fallback 0，test loss 0.2023175 ⇒ vs fdk3 36009.6 **+0.05% = 零收益**。分相位**不可加**：首遍前向单独 −882 ms，recompute 单独 +18 ms，两者都融合 −1463 ms（recompute 融合只在首遍前向也融合时才贡献 ~580 ms）。各 1 轮，run 间噪声 ~±100 ms（fdk 三轮极差 168 ms），差值远超噪声。机理未定；候选 = 同一 iter 内 BRS mapping 路径（DXRSContext K=4 共享 lane，与反向 deferred layers RS 同 ctx）和融合 ctx 混用的串行化/切换代价。
- **fbag1（前向融合 + DX-AG 融合反向，OPUS_DX_BWD=ag；opus_vehf5）= 34087.5 ms**，fallback 0，test loss 0.2023175 ⇒ vs fallt 两轮均值 34546.5 −459 ms（−1.33%），vs fdk3 −5.34%，vs base −6.99%。高于门测外推的 ~0.3s。同批 fallt3 / fbag2 待补。
- 附注：各臂 test loss 只在少数离散值间跳（0.2022311 / 0.2022966 / 0.2023175 …，同配置 fallt1≠fallt2），属 run 间非确定性，跨臂 loss 对齐口径 = 落在同一 1e-4 带内。
- fallt3 = 34666.8 ms（fallt 三轮 34532.4 / 34560.5 / 34666.8，均值 34586.6，极差 0.39%），test loss 0.2022311 ⇒ fbag1 vs 同批 fallt3 −1.67%。fbag2 待补。
- **fbag2 = 34361.5 ms**，fallback 0，test loss 0.2022311。同批 fbag1 / fallt3 / fbag2 = 34087.5 / 34666.8 / 34361.5 ⇒ fbag 均值 34224.5 **vs 同批 fallt3 −1.28%**，vs fallt 三轮均值 34586.6 −1.05%，vs fdk3 −4.96%，vs base 两轮均值 36648.9 **−6.62%**。
- **融合反向判决：KEEP（小幅）。** 两轮 fbag 极差 0.80%（大于 fallt 的 0.39%），赢幅 ~1.1-1.3% 约为 1.5-3 个极差，方向一致但边际不厚。

### F17 列并行前向融合 DX all-gather + GEMM（dx_ag_gemm_op，侧流 staging，复制版；opus_agf_gate.log，M32768 K5120 bf16）
- 正确性：64 op 背靠背逐 op 变异，max|full−ref|=0、max|out−分块参考|=0 ⇒ 位级 PASS（rc=139 = teardown segfault）。
| 层 | stock (RCCL AG + GEMM) | dx-ag-gemm | 每 op 同步 | b2b 16 op |
|---|---|---|---|---|
| fc1 Nc4352 | 6.800 (2.589 + 4.269) | 6.914 | **0.983×（输）** | 1.08× |
| QKV Nc896 | 3.669 (2.589 + 1.118) | 2.853 | **1.29×** | **1.25×** |
- QKV 融合后只比 RCCL AG 单独多 0.26ms ⇒ 小 GEMM 几乎被传输全盖住；fc1 的重叠收益被 8 块 M4096 分块 GEMM 的效率损失抵掉（同 F8 定律：GEMM 与传输时间之比决定赢输）。我事前预测 QKV 更差，**预测错误**。
- 车辆验证见 opus_vehf6（cp/ref/cp 同批，ref = fbag）。
- **车辆 cp1（fbag + 列并行 QKV+fc1 融合，引擎 staging；opus_vehf6）= 35325.5 ms**，fallback 0，cp_fused 12288 / cp_skip 96，test loss 0.2022311 ⇒ vs fbag 两轮均值 34224.5 **+3.22%（输）**。
- 纯前向 evaluate（test set，max over ranks）：cp1 19182.9 vs fbag 17721.9 / 17971.6、fdk3 19199.7 ⇒ **输在前向本身**（+1.2–1.46s，约 +1.3ms/列并行 op），非反向争用。门测（孤立）QKV 1.29× 在车里不成立。
- 待诊断：车内单 op 计时 opus_vehf8（cti=融合 / cts=stock，同 op 集，compute 流事件）；SM staging 门（opus_agf_gate_sm.log / opus_bwd_gate_sm.log）；QKV-only 与 SM staging 车辆臂 opus_vehf7。
- ref1（fbag，opus_vehf6 同批）= 34340.3 ms，fallback 0 ⇒ fbag 三轮 34087.5 / 34361.5 / 34340.3，均值 34263.1（vs base −6.51%）。cp1 vs 同批 ref1 **+2.87%（输）**。cp2 待补。
- cp2 = 35446.1 ms，fallback 0。同批 cp1 / ref1 / cp2 = 35325.5 / 34340.3 / 35446.1 ⇒ cp 均值 35385.8 **vs 同批 ref1 +3.05%（输）**，两轮 cp 极差 0.34%。
- **列并行融合（引擎 staging，QKV+fc1）判决：NO-GO**，输格保留。SM staging / QKV-only / 车内单 op 计时诊断继续（opus_vehf8、opus_*_gate_sm.log、opus_vehf7）。
- **车内单 op 计时（opus_vehf8，compute 流事件，每 op 同步上一 op 结束事件，6 iter；数字为 63–64 次 flush 中位的均值，r0/r5）**：QKV 融合 3.43–3.46 vs stock 3.74 ms（−0.3）；fc1 融合 5.98–6.04 vs stock 6.87–7.15 ms（−1.0）⇒ 设备侧每 iter 应省 ~0.65s。但同口径整步 cti1 35302.9 vs cts1 34393.6（+0.9s）⇒ **损失在 op 跨度之外**（op 间空泡），首要嫌疑 = host 入队阻塞（每次融合调用 ~60 次 host 调用，其中 7 次 staging hipMemcpyAsync 在引擎队列排在 RS push 后，T21 hostprof 已证该调用排满时阻塞 host）。SM staging 臂直接检验此假说；未验证前不下定论。
- **SM staging AG 门（opus_agf_gate_sm.log，OPUS_AG_STAGE=sm）**：位级 PASS（64 op，full/分块参考差 0）。孤立计时：QKV sync 2.994（引擎版 2.853）/ b2b 3.356（引擎版 2.890）⇒ b2b 赢幅 1.25×→1.07×；fc1 sync 6.959 / b2b 6.192（引擎版 6.914 / 6.241）。孤立场景 SM 拷贝与 GEMM 抢 CU，略慢；车内价值在于绕开引擎队列（host 阻塞假说），须车辆臂判定。
- **SM staging 反向门（opus_bwd_gate_sm.log）**：位级 PASS（64 op，full/gw 差 0，rel(gi)=0）。K2176 sync 6.108 / b2b 5.815（引擎版 6.087 / 6.383）⇒ b2b 1.17×→1.29×；K640 sync 3.525 / b2b 4.161（引擎版 3.538 / 3.977）⇒ b2b 1.006×→0.961×（输）。注意：OPUS_AG_STAGE=sm 同时作用于列并行 AG 与融合反向 AG，cps 臂两者一起换。
- 最终批（opus_vehf7，fresh mpirun 各 12 iter）：cps1(SM staging QKV+fc1) → cpq1(QKV-only 引擎 staging) → ref2(fbag) → cps2 → cpq2。
- cps1（SM staging QKV+fc1 + SM staging 融合反向；opus_vehf7）= 35284.6 ms，fallback 0 ⇒ 与引擎 staging 的 cp1/cp2（35325.5 / 35446.1）同档 ⇒ **换 staging 拷贝方式不回收损失，"staging 引擎拷贝阻塞 host"假说不成立**（单轮，cps2 待补）。
- cpq1（只融合 QKV，引擎 staging，OPUS_CP_MAX_NC=1024）= 34842.5 ms，fallback 0，test loss 0.2022660，evaluate(val/test) = 19251.56 18360.08 ，'cp_fused': 6144, 'cp_skip': 6240 ⇒ vs fbag 三轮均值 34263.1 **+1.69%（仍输，但比 QKV+fc1 的 +3.05% 少一半）**。同批 ref2 待补。
- ref2（fbag，opus_vehf7 同批）= 34471.0 ms，fallback 0 ⇒ fbag 四轮 34087.5 / 34361.5 / 34340.3 / 34471.0，均值 34315.1（vs base −6.37%），极差 1.12%。vs 同批 ref2：cps1 +2.36%，cpq1 **+1.08%（约等于 fbag 自身极差，QKV-only 的输赢未定，待 cpq2）**。
- cps2 = 35285.7 ms，fallback 0 ⇒ cps 两轮 35284.6 / 35285.7（极差 0.003%），vs 同批 ref2 34471.0 **+2.36%**。**列并行 QKV+fc1 融合终判 NO-GO**（引擎 staging +3.05% 两轮、SM staging +2.36% 两轮，均对同批 fbag）。staging 拷贝方式不是损失来源。
- cpq2 = 34779.1 ms，fallback 0 ⇒ cpq 两轮 34842.5 / 34779.1，均值 34810.8，vs 同批 ref2 34471.0 **+0.99%**，vs fbag 四轮均值 34315.1 +1.44%；两轮 cpq 都高于 fbag 四轮的最大值 34471.0 ⇒ 输幅小但方向一致。
- **列并行融合终判：全部 NO-GO**（QKV+fc1 引擎 +3.05% / SM +2.36% / QKV-only +0.99%，均对同批 fbag）。车内单 op 设备时间更快（QKV −0.3、fc1 −1.0 ms）而整步更慢 ⇒ 成本在 op 跨度之外；staging 拷贝方式已排除，host 入队计时（opus_vehf9）待定位。
- **今晚最终配置 = fbag**（前向融合 row-parallel GEMM+RS + 反向融合 DX-AG+dgrad，均 DX 直推）：四轮均值 34315.1 ms，**vs base 36648.9 −6.37%**，fallback 0，loss 同带。
- **host 入队计时（opus_vehf9，无事件无同步，每层 20 次 flush × 256 次调用，r0/r5）**：
| 层 | 融合 host 中位 / 均值 | stock host 中位 / 均值 | 融合设备时间（opus_vehf8） |
|---|---|---|---|
| QKV Nc896 | 4.6–5.1 / 8.0–8.2 ms | 0.33–0.36 / 2.6–2.8 ms | 3.43–3.46 ms |
| fc1 Nc4352 | 3.0–3.8 / 4.7–4.9 ms | 0.34–0.37 / 0.37–0.42 ms | 5.98–6.04 ms |
- ⇒ **融合列并行的 host 入队是 stock 的 ~10×（中位），QKV 的入队时间超过其设备时间** ⇒ host 发命令赶不上 GPU，op 间出空泡；按均值差估每 iter 多 ~5s host 时间（512×(8.1−2.7) + 512×(4.8−0.4)），只要其中一部分落在关键路径上就足以解释整步 +0.9s。**列并行融合输的主因 = host 入队开销**（每次调用 ~60 次 host API + 阻塞），非设备侧。修复方向（未做）：HIP graph 捕获单次融合调用 / 独立 host 发射线程。

### F18 MoE 预测器燃料探针（opus_moe_trace.py / _analyze.py；Qwen3-30B-A3B 推理前向，ShareGPT 中/英/代码混合，24 步 × 8192 token × 48 层，EP=8 连续 token 切分，128 专家 → 16/rank；opus_moe_trace_analysis.log）
- 偏斜：每 rank 接收量 max/mean p50 1.334 / p90 1.570 / max 1.937；单格 max/mean p50 1.903。
- 调度代价（单位 = 均匀理想）：固定置换序 p50 1.531，BvN 下界 1.334 ⇒ 固定序/下界 p50 1.127、p90 1.258、max 1.590（最差层 30/18/7/45/44）。
- 已知真实矩阵的匹配调度（每步最大权匹配）/下界：p50 1.043、p90 1.113 ⇒ 可回收固定序超额的 ~66%（1.127→1.043）。
- **用预测矩阵排序：上一步 1.126、EMA 1.127 = 零回收**。细粒度格预测比均匀还差（rel-L1：上一步 0.299、上一层 0.355、EMA 0.252、均匀 0.260）。
- **粗粒度可预测**：每目的 rank 接收负载 lag-1 相关中位 0.705（层间 0.302–0.885）；MAE EMA 0.099 vs 均匀 0.162；热 rank 跨步粘性 0.489（随机 0.125）。
- 结论：MoE 有偏斜燃料（≈13% a2a 超额，dense 车辆没有），但"预测流量矩阵来排推送序"无效；能预测的只有"哪个 rank 热"。精确矩阵在 dispatch 前本可得（gate 先于 a2a），代价是 splits 元数据同步 = 控制面问题。限制：推理前向非训练、24 步、单一模型。
- **免同步容量探针（同 trace）**：每格预留 m×预测量，第一轮直推、溢出走第二轮。均匀预测 ×1.5 覆盖字节 p50 0.984 / p10 0.961，×2.0 覆盖 1.000 / 0.993；EMA 预测 ×1.5 0.983、×2.0 1.000 ⇒ **格级预测对容量规划也无增益，静态 1.5–2× 均匀容量就够**。预测器在 MoE 上的唯一燃料 = 每目的 rank 负载（lag-1 0.705）与热 rank 粘性（0.489）。

### F19 控制面：融合 op 的 host 时间去向（opus_vehf10/chb1 + opus_push_hostblock.log）
- 车内（fused AG，3×256 次调用，r0）：host 5.4–5.7 ms/次，其中 **7 次 42MB peer push 占 4.01–4.53 ms**（~600µs/次）；matmul 发射 0.38–0.41、事件 0.29–0.36、票据 0.20–0.23，其余 <0.3。
- 孤立（空闲设备，8 rank 同时）：42MB peer push host 0.056–0.062 ms/次（单次、16 连发、WaitValue 之后、7 对端连发均同），本地拷贝 0.006–0.014 ⇒ **push 本身不阻塞 host；车内 ~10× 放大 = 引擎队列反压**（lane 上 ack-wait 未满足时队列积压，hipMemcpyAsync 在队列满时阻塞 host，同 T21 hostprof 定律）。
- 推论：跨 rank 的 host 耦合 —— 我的 lane 等对端 ack，对端 ack 要等对端 host 把 ack 核入队，而对端 host 正阻塞在它自己的 push 上 ⇒ 链式拖慢。修复 = 把 lane 的 wait/push/票据交给独立发射线程（OPUS_ISSUE=thread，ctypes 释放 GIL），主线程照常入队计算与 ack。门测排队中（opus_*_gate_thr.log）。

### F20 8 卡 MoE dispatch 真实路由（opus_moe_a2a8_{rank,perm}{,2}.log；Qwen3-30B-A3B 4 个切片，接收偏斜 p10/p50/p90/p99 = 1.18/1.33/1.56/1.82；T=1024/rank、H2048、top8、bf16；每 rank ~33.5MB；全部 CLEAN，接收行多重集与 RCCL 逐行一致）
| 切片 | 我方 rank 序（两轮） | 我方置换序（两轮） | RCCL varlen | RCCL 等分（同量） |
|---|---|---|---|---|
| p10 | 1.569 / 1.566 | 1.489 / 1.533 | 0.96–1.00 | 0.40 |
| p50 | 1.568 / 1.570 | 1.489 / 1.529 | 0.99–1.03 | 0.40 |
| p90 | 1.611 / 1.619 | 1.488 / 1.522 | 1.05–1.10 | 0.40 |
| p99 | 1.512 / 1.613 | 1.489 / 1.512 | 1.10–1.14 | 0.41 |
- **我方全输 RCCL varlen：0.63–0.75×**（输格保留）。置换序比 rank 序快 2–8%（incast 定律在 a2a 上成立，但不是主矛盾）。
- 对照：RCCL 等分 0.40ms vs RCCL varlen 0.96–1.14ms ⇒ **RCCL 自己也为可变 splits 付 0.56–0.73ms**（count a2a + .tolist() host 同步 + 按目的重排）。我方 1.49–1.62ms 里同样是元数据 RMA + 2 次窄同步 + D2H 计划。⇒ **MoE a2a 的瓶颈 = 控制面（split 大小必须先到 host 才能给引擎 push 定尺寸）**，数据面只占 ~0.4ms。
- 下一刀：免同步 SM 直写原型（opus_moe_a2a_sf.py：最坏情况容量区、设备侧原子领槽 + 跨卡 SM 存储 + release 旗、接收端流等旗，host 永不知道大小），已排队。

### F19b 发射线程（OPUS_ISSUE=thread）AG 门（opus_agf_gate_thr.log）
- 位级 PASS（64 op，full/分块参考差 0）。孤立计时与非线程版同档：fc1 sync 1.010× / b2b 1.082×，QKV sync 1.212× / b2b 1.146×（孤立场景无反压，线程本应无增益——它的价值只在车里检验）。

### F21 免同步 MoE dispatch 原型（opus_moe_a2a_sf.py；最坏容量区 + 设备原子领槽 + 跨卡 SM 存储 + release 旗 + 接收端 WaitValue；host 不知 splits）
- 第 1 轮（opus_moe_a2a_sf_1.log）：p10 0.411 / p50 0.429 / p90 0.482 / p99 0.544 ms vs RCCL varlen 0.967 / 1.018 / 1.056 / 1.127 ⇒ **2.35 / 2.37 / 2.19 / 2.07×**；每切片 40 轮逐轮变异，计数 + 每源行多重集与 RCCL 一致 ⇒ 全 CLEAN。第 2 轮待补。
- 低偏斜贴近 RCCL 等分无元数据下界 0.40ms ⇒ 控制面税被消除；时间随偏斜涨（被最热目的接收量绑定）⇒ 预测器燃料（F18：每目的负载 lag-1 0.705、热 rank 粘性 0.489）。
- 限制：SM 存储占 CU（与专家 GEMM 重叠时须验证）；最坏容量 T·K 行/源 = 268MB/rank recv 区。
- 反向线程门（opus_bwd_gate_thr.log）：位级 PASS；K2176 sync 1.342× / b2b 1.256×（非线程 1.237 / 1.17-1.29）；K640 1.118× / b2b 0.979×。
- 第 2 轮（opus_moe_a2a_sf_2.log，独立进程，跑在 overlap 改动之前）：0.413 / 0.434 / 0.483 / 0.546 vs RCCL varlen 0.965 / 0.992 / 1.099 / 1.123 ⇒ **2.34 / 2.29 / 2.27 / 2.06×**，全 CLEAN。两轮极差 ≤0.005ms ⇒ **孤立 dispatch 2.06–2.37× RCCL 可引用**。重叠（与专家 GEMM 抢 CU）版排队：opus_moe_a2a_sf_ovl.log。
- **发射线程车辆臂 cpt1 楔死（opus_vehf11/cpt1，15:09 后零输出 >4min）**：py-spy 全 8 rank 同态 = 主线程阻塞在 dx_gemm_rs_op 的 memcpy_async（融合 RS push）+ opus-issue 线程阻塞在 AG push 的 memcpy_async。车辆 GPU_MAX_HW_QUEUES=4 / CUDA_DEVICE_MAX_CONNECTIONS=1：发射线程让主线程越过 AG push 继续入队，流共享 4 条 HW 队列，lane 上等对端 ack 的 WaitValue 堵住同队列里本该释放对端的 ack/计算 ⇒ 跨 rank 环形等待。**定律：在共享 HW 队列上，host 侧解耦发射会把隐式的 host 顺序栅栏拆掉，暴露设备侧循环依赖；发射线程方案在该配置下 NO-GO**（孤立门测 PASS 不能代表车内）。剩余线程臂取消，已杀干净。
- **重叠探针（opus_moe_a2a_sf_ovl.log，dispatch 侧流 ‖ 专家 FFN 尺寸 GEMM 计算流，GEMM 单独 0.346ms）**：
| 切片 | sf 单独 | GEMM‖sf（串行和） | rccl_v 单独 | GEMM‖rccl_v（串行和） | 重叠后 rccl/sf |
|---|---|---|---|---|---|
| p10 | 0.411 | 0.748 (0.757) | 1.013 | 1.073 (1.359) | 1.43× |
| p50 | 0.435 | 0.762 (0.780) | 1.022 | 1.098 (1.368) | 1.44× |
| p90 | 0.490 | 0.822 (0.838) | 1.104 | 1.195 (1.451) | 1.45× |
| p99 | 0.550 | 0.875 (0.896) | 1.123 | 1.199 (1.469) | 1.37× |
- ⇒ **SM 直写与 GEMM 几乎零重叠**（0.748 vs 串行 0.757：两者抢 CU，等于串行）；RCCL 能藏住 GEMM 的 79–88%。重叠场景下我方赢幅从孤立 2.04–2.46× 缩到 **1.37–1.45×**（仍赢，因为总量小）。
- 推论：理想形态 = **DMA 引擎数据面（不占 CU，可与专家 GEMM 真重叠）+ 设备侧/免同步控制面**。引擎 push 需要 host 给定字节数 ⇒ 用**预测容量**定尺寸（无同步），超出预测的残余行走小 SM 路径。这是预测器在 MoE 上的真实工作（容量定尺），见下一实验。

### F22 混合 dispatch（引擎数据面按容量定尺 + SM 残余；opus_moe_a2a_hy_{1,2}.log，m = 容量/均值 ∈ {1.25,1.5,2.0}，每格 20 轮逐轮变异全 CLEAN）
- 孤立：0.83–1.07 ms（两轮），vs RCCL varlen 1.01–1.17 ⇒ **1.09–1.35×**（比纯 SM 免同步 0.41–0.55 慢一倍：多了一次本地 pack 拷贝 + 7 次 m×均值行的引擎 push，p10 切片 m=1.25 时残余行 0%）。
- **与专家 GEMM 重叠：gemm‖hy 1.17–1.31 ≈ 串行和（hy + 0.34）⇒ 输 RCCL 0.87–0.995×，也输纯 SM 版（0.75–0.88）**。引擎 push 本应不占 CU，却没有重叠 —— 嫌疑：host 在 push 处阻塞（push 排在同流 pack 核之后），GEMM 的入队被推迟到 push 发完。诊断已排队：push 排在运行中核之后的 host 时间（opus_push_hostblock2.log）+ GEMM 先入队的重叠序（opus_moe_a2a_hy_3.log）。
- 残余行占比（r0）：p50 切片 m=1.25 9.5%、m=1.5 5.1%、m=2.0 0.5%；其余切片 ≈0 ⇒ 静态 1.5–2× 容量足够（与 F18 一致）。

### F23 免同步 MoE 完整一轮（dispatch → 专家 → 回写 combine → 加权归约；opus_moe_a2a_sf2b_{1,2}.log；每切片 30 轮逐轮变异，与本地参考 max|diff| = 0）
| 切片 | 免同步完整轮（两轮） | RCCL 路径（count a2a + tolist + varlen dispatch + 专家 id a2a + 反向 varlen + 反排 + 归约） | 加速 |
|---|---|---|---|
| p10 | 0.982 / 0.968 | 2.401 / 2.437 | **2.45 / 2.52×** |
| p50 | 1.008 / 1.013 | 2.504 / 2.482 | **2.48 / 2.45×** |
| p90 | 1.130 / 1.119 | 2.734 / 2.713 | **2.42 / 2.43×** |
| p99 | 1.223 / 1.213 | 2.853 / 2.857 | **2.33 / 2.36×** |
- 全程无 host 同步、无 split 交换、无屏障；combine 按 4B tag 直接写回源 rank 的行位置（免索引交换）。RCCL 路径正确性同验（max|diff| 0）。

### F22b 重叠序诊断（opus_push_hostblock2.log + opus_moe_a2a_hy_3.log）
- push 排在同流运行中 2ms 核之后：host 0.072 ms；排在他流忙时 0.091 ms ⇒ **push 不因前序核阻塞 host**，F19 的车内 600µs/push 只能来自队列反压/ack 链。
- 混合 dispatch 重叠：dispatch 先入队 1.23–1.31（≈串行）；**GEMM 先入队 1.03–1.19 ⇒ 0.96–1.21× RCCL** ⇒ 差异在设备调度序（8192 个 pack 程序占满 CU，后入队的 GEMM 等 CU），不在 host。混合版即便调好序仍慢于纯 SM 免同步重叠（0.75–0.88）⇒ 本规模下纯 SM 免同步最优，混合引擎版 PARK（本地 pack 拷贝代价吃掉引擎不占 CU 的收益）。
- **设计规律（数据面放置三角）**：引擎数据面不占 CU 但要 host 定尺寸 + 受队列反压；SM 数据面无 host 但与 GEMM 抢 CU（零重叠）；RCCL 居中。大载荷 + 重 GEMM（dense RS，fbag 赢）宜引擎；小载荷 + 轻专家 GEMM + 可变 split（MoE dispatch/combine）宜 SM 免同步。⇒ 按算子选路 = 代价模型（点 2）的输入。

### F24 MoE 训练车辆（Qwen3-30B-A3B 形状裁 12 层，8.10B 参数，EP=8 TP=1 DP=8，MBS1 seq4096，128 专家 top8，SequentialMLP，Megatron stock alltoall dispatcher；opus_moeveh/driver.log）
- 整步（iter3+ 中位，12 iter）：base1 1522.1 / base2 1522.7 ms；计时臂 tim1 1541.9（+1.3%，含 warmup 统计已弃）/ tim2 1584.7（+4.1%）。
- **每 MoE 层调用分相（tim2，跳过前 3 iter，5 次 flush × 48 次调用，r0 / r7，compute 流事件 dev / host ms）**：
| 相 | dev r0 | dev r7 | host |
|---|---|---|---|
| pre（路由元数据 + count 全收集 + permute1） | 1.09–1.34 | 2.74–3.14 | 1.6–2.3 |
| a2a1（token + probs 的 EP a2a，含 D2H 同步） | 2.28–2.45 | 2.30–2.46 | **17.7–22.0** |
| post（按本地专家排序） | 1.32–1.39 | 1.12–1.25 | 0.7–0.8 |
| exp（16 专家 SequentialMLP） | 3.47–3.73 | 3.22–3.77 | 4.8–5.7 |
| cpre（反排序） | 1.41–1.54 | 1.25–1.35 | 0.5–0.6 |
| a2a2（combine a2a） | 2.63–3.94 | 2.93–4.29 | 0.3 |
| cpost（unpermute） | 1.44 | 1.30 | 0.2 |
| 合计 | 14.1–15.5 | 15.9–16.9 | 26.6–30.3 |
- 读法：a2a1 的 host 17.7–22 ms = "before_ep_alltoall" 的 D2H 同步——host 每层都要等设备排空（失去全部 run-ahead）；同步后 host 每层还要 ~9 ms 入队、设备每层 ~15 ms 执行。pre 的 r0/r7 差（1.1 vs 3.1 ms）= count 全收集上的 rank 偏斜等待。
- 量级：前向（含 recompute）每步 24 次层调用 × 两次 a2a ≈ 5–6.5 ms ⇒ ~130–160 ms/步；加反向 a2a（未挂钩）粗估全部 a2a + 元数据同步 ≈ 250 ms / 1522 ms ≈ 16%。⇒ 免同步 a2a 车内上界约 5–10%（不含 run-ahead 恢复的额外收益）。
- 车辆体量 T=4096/rank（134MB/rank/方向）是 F21–F23 基准（T=1024）的 4×：RCCL 在车里 a2a1 ~2.3ms ≈ 58GB/s/rank，带宽已不差；免同步 SM 版在该体量的赢幅需重测（opus_moe_a2a_sf_t4096.log 运行中）。
- 车辆规模孤立 dispatch（opus_moe_a2a_sf_t4096.log，T=4096/rank，134MB/rank/方向，8 轮变异 CLEAN）：免同步 SM 1.313 / 1.449 / 1.591 / 1.839 vs RCCL varlen 2.533 / 2.538 / 2.996 / 2.980 ⇒ **1.62–1.93×**（T=1024 时 2.06–2.37×）。
- **车内 DX 引擎 a2a 替换（opus_moe_a2a_patch.py，OPUS_MOE_SF=1；保留 Megatron 的 split host 同步，只换数据面 = 置换序引擎 push + 4B 票据 + 本格 WaitValue + 区域压实 + ack；全部在计算流上）**：
  - 两个装配坑：① train() 钩子不触发（stock 分支下所有调用回落 RCCL，sfv1 无效）→ 改为首次 a2a 时惰性 bootstrap；② 晚加载 MPI 库 → MPI_Comm_size segfault（sfv2）→ prelude 起始处 RTLD_GLOBAL 预加载 MPI（S39 次序律，与 mgd_prelude2 引擎臂同法）。
  - **sfv3 正确性：前 200 次 a2a 与 RCCL 位级比对 0 失败，全程 792 次 0 回落**，loss 轨迹与 base 一致（iter6 1.169006 vs 1.169025）。
  - **性能：未校验段 iter4–6 1798.9–1823.5 ms vs base 1522 ⇒ +18%（输）**。分相计时臂 sft1 运行中定位。
- **sft1 分相（引擎版 + 计时，12 iter，1867.2 ms/步）vs stock tim2**：a2a1 dev 3.1–3.7（stock 2.3–2.5），a2a2 dev 5.0–6.4（stock 2.6–4.3），cpre 2.1–3.7（stock 1.3–1.5），exp 4.5–5.2（stock 3.2–3.8）⇒ **引擎版每个 a2a 都慢于 RCCL，且把相邻计算相（cpre/exp）也拖慢**。读法：push 后的区域压实又是 7 次引擎本地拷贝，全在计算流上与票据/ack 串行；引擎拷贝与 RCCL 的 SM 内核相比没有体积优势（单 a2a 134MB，RCCL 车内 ~58GB/s/rank 已不差），每次 a2a 额外的 7 次 WaitValue + 14 次 4B memcpy 是纯延迟。**引擎数据面判 NO-GO**（在 MoE a2a 上）。
- 改为 SM 数据面（OPUS_MOE_SF_MODE=sm）：1 个 push 核（8 目的一次发射）+ 1 个 release 旗核 + 7 次本格 WaitValue + 1 个压实核 + 1 个 ack 核；位级门 smv1 运行中。
- **SM 数据面车内版（OPUS_MOE_SF_MODE=sm）**：smv2/smv3 前 200 次与 RCCL 位级一致（0 失败，1440 次 0 回落）。第一版每次调用用可分页 torch.tensor 传参（隐式排空流）→ 1864 ms；改钉页环 + 常量旗地址后 smv3 1728.0 / smt1(计时) 1780.1 ms vs base 1522 ⇒ **仍 +13.5%（输）**。分相：a2a1 3.1–4.2 / a2a2 5.2–6.1 ms（stock 2.3–2.5 / 2.6–4.3）。
- 结论：**在保留 Megatron split host 同步的前提下，只换数据面赢不了**——孤立基准的 1.6–1.9× 主要来自省掉 count 交换 + tolist host 同步（控制面）；车内 RCCL 拿到的已是"已知 splits"的纯数据 a2a，而我方多出区域压实（再读写 134MB）+ 每次 a2a 的 ack 耦合。真正收益须去掉同步本身 ⇒ 专家阶段要免同步（设备侧 sizes 的 grouped GEMM 或按容量补齐）。先测 Megatron 自带的 drop-and-pad（静态 split、无同步）作为"同步代价"上界：opus_moeveh pad15/pad20 臂运行中。
- **drop-and-pad 上界（Megatron 自带静态 split = 无 host 同步，按容量补齐）**：pad1.5 1473.1 / 1474.8，pad2.0 1597.1，dropless base 1522.1 / 1522.7 / 1580.3（base3 晚批漂移 +3.8%）⇒ **pad1.5 比 dropless −3.2% ~ −6.7%**，pad2.0 补齐计算吃掉收益反而 +1~5%。⇒ 本车辆（12 层、MBS1）去掉同步的总价值 ≈ 3–7%/步，是"免同步 dropless"（无补齐浪费）的大致上界。工程代价（设备侧 sizes 的 grouped GEMM + 免同步 a2a + 反向）高，收益薄 ⇒ MoE 端到端暂 PARK，保留算子级结果（F21/F23：孤立 2.06–2.52×，与专家 GEMM 重叠 1.37–1.45×）。

### F25 端到端：70B 形状（hidden 8192 / ffn 28672 / 64H 8KV，NL8，MBS8，TP=8 SP；opus_e2e_70b/driver.log，16 iter，iter3+ 中位，每臂独立 mpirun）
- b1 stock 18855.7 ms（val 0.1099631）；f1 fbag **17492.2 ms（−7.23%）**（val 0.1099532），fused 4096/4096 fallback 0，BRS 2304 relayed。第二对 b2/f2 + 自动分派器 a1 运行中。
- 实现与提交包：在线贪心分派器 OPUS_AUTOTUNE=1（mgd_tp_patch.py：row/bwd/col 运行时开关 + train_step 钩子，全 rank MAX 归约决策）；双机 16 卡包 e2e/dual16/vcjob.sh + opus_arm_inner_16.sh（TP8×DP2，BASE/FDK/FBAG/FAT，MGD_BRS_MODS=mappings,layers），待用户提交。
- **70B 终表（ABAB）**：stock b1/b2 = 18855.7 / 18928.5（均值 18892.1，极差 0.39%）；fbag f1/f2 = 17492.2 / 17480.4（均值 17486.3，极差 0.07%）⇒ **fbag −7.44%**，val loss 0.1099505–0.1099532 vs stock 0.1099631–0.1099707（同带）。27B 为 −6.37% ⇒ GEMM 越重赢幅越大。
- a1（自动分派器第一版）= 17702.0：**钩子未生效**（das adaptor 把 train() 换成 hcu_megatron.training.training.train，其调用的是 hcu 模块自己的 train_step），分派器全程停在默认"全开"(row+bwd+col) ⇒ 该数 = fbag+列并行，比 fbag 慢 1.2%（与 27B 列并行输一致）。已修：在 train() 入口对所有已加载的 *training.training 模块的 train_step 挂钩；a2 重跑中（it=3 base 探针 18885.6 ms 已出）。
- **分派器 a2（钩子修复后，70B；探针 = 1 次同步迭代）**：base 18885.6 → +row 17887.6（KEEP）→ +bwd 17574.0（KEEP）→ +col 17345.5（KEEP）⇒ CHOSEN row+bwd+col。**判错**：独立进程实测全开 a1 = 17702.0 > fbag 17486.3（列并行在稳态输 ~1.2%）。原因：单次同步迭代从空队列开始，抹掉了 host 入队超过设备时间造成的稳态空泡——正是列并行输的机理（F19）⇒ 探针协议本身必须保留 host run-ahead。修复：探针 = 3 次迭代，仅在探针边界同步，计时后 2 次流水迭代（OPUS_AT_PROBE=3）；27B a1 与 70B a3 用新协议重跑。
- **更正（上一条"判错"作废）**：a1 的"全开"默认是 row+bwd(**t**)+col（OPUS_DX_BWD 未设时 _RT.bwd="t"），不是 fbag+col；a2 选出的是 row+bwd(**ag**)+col，其决策后 8 次迭代 17302.4–17386.4 ms（中位 ~17345）< fbag 17480–17492 ⇒ **在 70B 列并行融合是赢的（≈−0.8%），分派器判对了**。与 27B（列并行输 1–3%）相反，符合"GEMM/传输比决定赢输"定律（70B fc1 Nc=7168、QKV Nc=1280，GEMM 更重）。⇒ 这正是分派器的价值：不同形状最优配置不同。跨进程对比仅作参考，同批独立进程 fbag+col 臂已排队确认（opus_e2e_70b fbc*）。

### F26 驱动脚本环境泄漏（10-30 晚，接手会话复盘发现）
- `e2e/megatron/run.sh` 每臂 unset 清单漏了 OPUS_AUTOTUNE / OPUS_CP（及 OPUS_CP_MAX_NC/CP_TIMING/AT_*/AG_STAGE/ISSUE）⇒ 同一次 run_e2e 里 fat/fbc 臂之后的所有臂继承这两个开关。日志实证（out_r0 的 args 打印 + [autotune] 行）。
- **受污染**：
  - 70B 链 a3→c1→f3→c2：c1/f3 实为分派器臂（f3 环境 OPUS_AUTOTUNE=1 + OPUS_CP=1），c2 开跑 26s 后停链。三次分派器（a3/c1/f3）全部 CHOSEN row+bwd+col，决策后末 3 iter：a3 17340/17330/17385，c1 17311/17349/17425，f3 17349/17341/17351 ⇒ 分派器决策可复现。
  - 27B 晚批 a1→f1：f1 "fbag-6GB 38616.7" 实为分派器（中位含探针）⇒ 交接续篇 §7 "6GB 堆本身拖慢 ~4%" 推论作废（混淆）。f1 选 row+bwd、丢 col（+col 38585.8 vs 38616.0，差 30ms 边际）；a1 选全开（38259.8 vs 38556.6）⇒ 27B 两次分派器在 col 上不一致，而独立进程列并行在 27B 输 1–3% ⇒ 探针对 col 的代价可能低估，代价模型回测须核。
- **未受影响（已核环境）**：70B b1/f1/b2/f2（fat 排最后）⇒ fbag −7.44% 有效；27B vehf5–7 ref1/ref2 无 OPUS_CP、cp_fused=0 ⇒ fbag −6.37% 与列并行 NO-GO 有效（那批用旧 kind 名 fcp*，每臂显式设置）。
- 修复：unset 清单补齐（run_e2e.sh:21-22）。同批固定配置确认重跑：`opus_e2e_70b_cp/`（f1:fbag c1:fbc f2:fbag c2:fbc，16 iter）。

### F27 70B 列并行同批确认（opus_e2e_70b_cp，ABAB f1:fbag c1:fbc f2:fbag c2:fbc，16 iter，iter3+ 中位；修复泄漏后，四臂 out_r0 环境已核：f* 无 OPUS_CP/AUTOTUNE、cp_fused=0；c* OPUS_CP=1、cp_fused=4096）
| 臂 | 轮 (ms) | 均值 | 极差 |
|---|---|---|---|
| fbag | 17504.4 / 17542.0 | 17523.2 | 0.21% |
| fbc (fbag+列并行) | 17418.1 / 17423.4 | 17420.8 | 0.03% |
- 配对：c1 vs f1 −0.49%，c2 vs f2 −0.68% ⇒ **70B 列并行同批赢 −0.58%**（两对同号，赢幅 ≈ 2.8× fbag 极差）。fallback 0，val loss 同带（0.1099357–0.1099373）。
- 与 27B 相反（列并行输 +0.99~+3.05%）⇒ "最优配置随形状变化"在同批固定臂上成立，分派器动机站住。
- 小悬项：分派器决策后迭代（a3/c1/f3 跨批 ~17340）比同配置固定臂 fbc 17421 低 ~0.5%；同期 fbag 也从 17486 漂到 17523。跨批，不下结论。

### F28 27B 干净窗分派器（opus_e2e_27b_at，3GB 堆，b1:base f1:fbag a1:fat，20 iter，每臂独立进程；环境已核）
| 臂 | iter3+ 中位 | vs base |
|---|---|---|
| base | 37040.1 | — |
| fbag | 34989.4 | −5.54%（1.059×） |
| fat 决策后（iter15–20 中位） | ~35843 | −3.23% |
- 探针：base 37253.5 → +row 36274.5 KEEP → +bwd 35860.1 KEEP → +col 36084.4 **drop** ⇒ CHOSEN row+bwd = fbag 配置。**选择判对**（27B 关列并行，与 70B 开列并行相反，按形状选路成立）。
- **但决策后速度比同批同配置 fbag 慢 ~854 ms（+2.4%）**。差异只有两处：fat 臂预热期全开（建了 cp ag-gemm ctx：额外对称堆 + 2 条流）且 ColumnParallelLinear 被包装（col 关时走 orig）。嫌疑 = 探测残留（同进程建/拆对称堆拖慢后续引擎臂 10–20% 的臂序污染律在分派器内部复现）。70B 未见此损（选 col，ctx 在用）。
- 诊断批 opus_e2e_27b_atres（f1:fbag x1:fatx n1:fatn f2:fbag，16 iter）：fatx = 预热全开后直接钉 row+bwd（不探测）；fatn = 预热不开 col（不建 cp ctx）再钉 row+bwd。新开关 OPUS_AT_FIXED / OPUS_AT_WARM_COL（fork mgd_tp_patch.py，默认关）。

### F29 分派器"选对却更慢"定位（opus_e2e_27b_atres，27B 3GB 堆，16 iter，iter3+ 中位；f2 的 driver rc 行缺失——驱动随会话被杀，out_r7 16 iter 完整）
| 臂 | 含义 | ms/iter | vs fbag 均值 34981 |
|---|---|---|---|
| f1 / f2 fbag | 固定 row+bwd | 34951.3 / ~35011 | — |
| x1 fatx | 预热全开 → 不探测直接钉 row+bwd | 35892.0 | **+2.6%** |
| n1 fatn | 预热不开 col（不建 cp ctx）→ 钉 row+bwd | 35190.3 | +0.6% |
- ⇒ 损失 = **列并行 ctx 常驻 ≈ +2.0%**（fatx−fatn 702 ms）+ 列并行包装/其余 ≈ +0.6%（fatn−fbag 209 ms，≈3.5× fbag 极差）。不是探测本身。
- 修复（fork，OPUS_AT_RELEASE 默认开）：col 被丢弃时 DXAGGemmContext.close()（dushmem free + 事件销毁，集体调用）+ 恢复原版 ColumnParallelLinear._forward_impl + empty_cache。验证批 opus_e2e_27b_atfix（f1:fbag a1:fat f2:fbag a2:fat，24 iter）排在 70B MBS 扫描之后。

### F30 70B MBS 扫描（opus_e2e_70b_mbs{1,2,4}，NL8，ABAB base/fbag，14 iter，iter3+ 中位；MBS4 运行中）
| MBS（M） | base | fbag | 差 | 我方路径实际接管的 op |
|---|---|---|---|---|
| 1（4096） | 2830.8 / 2826.2 | 2790.7 / 2795.3 | **−1.30%** | **0**（BRS eligible 0、fallback 5600；row fused 0）|
| 2（8192） | 5141.0 / 5134.2 | 5298.3 / 5302.3 | **+3.13%（输）** | 仅纯 RS：BRS relayed 5600（134MB）；row 不接管（分片 16.8MB < MGD_MIN_MB 32）|
- **MBS1 是混淆信号**：一个 op 都没走我方路径，fbag 仍比 base 快 1.3%（两轮极差 0.16% ⇒ 真实）。base 与 fbag 的差别只剩：fork 在 PYTHONPATH 首位、6GB dushmem 对称堆、引擎 bootstrap（MPI+dushmem init）、MGD_* 包装器。⇒ **整步"对 stock"的收益里可能含一部分非重叠成分**，必须用空对照（同 fork/堆/bootstrap/包装、阈值设到不可达 ⇒ 0 op 接管）把它分离。空对照批 opus_e2e_70b_null1（MBS1）+ opus_e2e_70b_null（MBS8：b/n/f/n/b）已排队。论文的重叠收益口径应改为 fbag vs 空对照。
- **MBS2：纯 DX RS 在车内 134MB 输 3.1%**（孤立门测 134MB 赢 1.27×）——又一例孤立赢、车内输。现有分派器只开关 row/bwd/col，**不管纯 RS 路径（静态阈值 MGD_BRS_MIN_MB=128）** ⇒ 分派器应把纯 RS 也纳入开关集。
- **MBS4（M16384）**：base 9670.2 / 9639.9（均值 9655.1），fbag 9091.3 / 9097.8（均值 9094.6）⇒ **−5.80%（1.062×）**，两轮极差 ≤0.3%。70B 整步曲线：M4096 −1.3%（零接管，混淆）/ M8192 +3.1%（仅纯 RS，输）/ M16384 −5.8% / M32768 −7.44% ⇒ 赢幅随 M 单调增（M≥16384 起）。空对照分离非重叠成分后定稿。

### F31 夜间三批（10-30 23:19 → 10-31 01:00）
**27B 分派器修复验证（opus_e2e_27b_atfix，24 iter）：未通过。** 本窗整体慢（fbag 37129.9 / 37334.7，比 20:36 那批 34989 慢 6%，机器漂移）。两次分派器**都保留了 col**（+col 38396.8 vs +bwd 38532.0；38243.4 vs 38537.6，差 0.35–0.77% > 0.3% 门），决策后 iter ~38390 / ~38245 ⇒ 比同批 fbag 均值 37232 慢 **+2.7–3.1%（选错）**。
- 机理（与 F29 同源）：预热全开建了 col 上下文，它常驻拖慢之后所有"col 关"的探针（F29 量得 +2.0%），于是 col 看起来"白送"。20:49 那次恰好落在门内侧（36084 vs 35860 → drop），今晚落在外侧 ⇒ 分派器决策被自身残留污染，结果随窗口翻转。release-on-drop 只在丢弃 col 时生效，解决不了这个。
- 修复：预热不开 col，col 上下文在 +col 探针时才建（OPUS_AT_WARM_COL=0），丢弃时释放。验证批 opus_e2e_27b_fatw（f/w/f/w，22 iter）已排。预注册门：两次都丢 col，且决策后中位与同批 fbag 差 ≤0.5%。
**探针协议对照（opus_e2e_27b_p2，OPUS_AT_PROBE=2 单次同步探针）**：两次也都保留 col（38381 / 38292 vs +bwd 38528 / 38568）⇒ 与 PROBE=3 同样被残留污染，**本批无法检验探针协议定律**（混淆），定律继续标"待验证"，待 fatw 修复后若需要再测。
**空对照（opus_e2e_70b_null1，70B MBS1，零接管）**：base 2828.4 / 2810.7（均值 2819.6），null 2794.8 / 2792.9（均值 2793.9）⇒ **null 比 base 快 0.91%**，两轮同号 ⇒ 非重叠成分真实存在（同条件 fbag −1.3%）。来源二分批 opus_e2e_70b_bisect1（base / 仅 fork+堆 / +BRS 包装 / +row 包装 / 全空对照，各两轮）排在 fatw 之后。MBS8 空对照运行中（b1 base 18949.4）。**论文口径：重叠收益 = fbag 对空对照；对原版的总差另列，并说明非重叠成分来源。**
**空对照 MBS8（opus_e2e_70b_null，70B M32768，b/n/f/n/b，14 iter）**：base 18949.4 / 18951.4（均值 18950.4），null 18837.8 / 18838.7（均值 18838.3），fbag 17498.1。
- null vs base **−0.59%**（两轮极差 0.005%，真实）；**fbag vs null −7.11%（1.077×）= 纯重叠收益**；fbag vs base −7.66%（总差）。⇒ 70B 头条口径定为 **1.077×（对空对照）**，非重叠成分 0.59% 另列（来源二分批 opus_e2e_70b_bisect1 待出）。MBS1 非重叠 0.91%、MBS8 0.59%（绝对值约 26 ms vs 112 ms，非固定成本也非固定比例，待二分）。

### F32 分派器修复 fatw（opus_e2e_27b_fatw，预热不开 col + 按需构建 + 丢弃释放，22 iter）
- w1：base 39468.8 → +row 37761.6 KEEP → +bwd 37265.7 KEEP → +col 38029.4 **drop**（+2.0%，col 真实代价不再被残留掩盖）→ released col ctx → CHOSEN row+bwd。决策后末 5 iter 37173.5–37228.0（约 37200）vs 同批 fbag f1 37149.1 ⇒ **+0.14%，在 0.5% 门内**。w2 / f2 待出。

### F33 10-31 夜间链结果（opus_night1031/，03:14–07:15 全部完成，0 BROKEN）
**算子矩阵（11 dense 模型 × M 2048–32768，3 轮中位；每模型一个进程内循环全部 M——违反"每臂独立进程"律，见下）**
- GEMM+RS（dx）：赢 15/55；M32768 赢 7/11，几何均值 1.095；全格几何均值 0.744；若每格取 max(ours, RCCL)（理想分派）1.042。
- AG+GEMM（dxag，up + gate_up）：赢 34/110；**M≥16384 赢 34/44，M32768 赢 22/22（几何均值 1.186）**；M≤8192 全输；理想分派 1.043。
- 纯 RS：复现 0.546 / 1.283 / 1.478 / 1.541 / 1.600×（33.6→537 MB，md=0）。
- 反向 AG+dgrad（8 模型 × 3 M，位级 PASS 24/24）：M32768 赢 8/8（1.017–1.199），M16384 4/8，M8192 0/8。
- **疑点：RS 的 M16384 非单调凹陷**（70B 1.155 → 0.874 → 1.314；72B 1.072 → 0.880 → 1.198；dx 5.987 ms 而 gemm8 仅 3.720）。F6 独立进程测 70B M16384 = 1.196 ⇒ 疑为进程内多 M 建/拆 ctx 的臂序污染。qwen3-8b M32768 stock 5.821（M16384 1.855 的 3.1×）也异常。⇒ 每格独立进程复测 opus_day1031（D1/D2），预注册：70B/72B RS M16384 ≥ 1.05 即判污染。
**整步终版（每臂独立进程，18 iter，iter3+ 中位，回退 0）**
| 形状 | base | 空对照 | fbag | 分派器（决策后） | fbag 对空对照 |
|---|---|---|---|---|---|
| 70B M32768 | 17811.5 / 17813.8 | 17788.7 / 17798.2 | 16739.6 / 16746.2 | ~16958 / ~16765 | **−5.91%（1.063×）** |
| 27B M32768 | 37159.7 / 37123.2 | 37000.8 / 37033.9 | 35080.6 / 34981.8 | ~34944 / ~35074 | **−5.36%（1.057×）** |
- 本窗 70B 原版比 10-30 晚快 6%（17812 vs 18950），纯重叠收益从 7.11% 降到 5.91%（同口径、不同窗）⇒ 论文报区间并注明窗口。
- **分派器（预热不开 col，F32 修复）4/4 次都关 col**：27B 选后 ~35009 vs fbag 35031（−0.06%，**M1 通过**）；70B 选后 ~16862 vs fbag 16743（+0.71%，第一次 +1.3% 偏多）。70B +col 探针 +0.74% / +0.88% ⇒ 与 F27"70B 列并行同批赢 0.58%"矛盾，固定臂复测 opus_day1031/e2e_70b_col（D3）。
- 空对照非重叠成分：70B MBS8 本窗 −0.11%（上窗 −0.59%），MBS4 −1.33%，MBS2 −0.76%，MBS1 −0.91%；27B −0.33%。来源已二分到 BRS 包装（opus_e2e_70b_bisect1：fork+堆 2828 = base，+BRS 包装 2795）；D4 测是否为 6GB 对称堆在 RS 包装激活后的效应。
- 默认值变更：fork mgd_tp_patch.py `OPUS_AT_WARM_COL` 默认 0（按需建 col）。16 卡包 e2e/dual16/vcjob.sh 同步：FAT 用按需建 col，新增 NULL 空对照臂，unset 清单补齐（备份 .bak_1031）。
- **F33b 判决（opus_day_rs_fresh.csv，每格独立进程，M16384 首轮）**：70B 1.196（夜间 0.874）、72B 1.096（0.880）、dsv3-d 1.334（0.885）、mistral-l 1.410（1.004）、gemma3-27b 1.130（0.798），全 CLEAN ⇒ **预注册门（≥1.05）通过：夜间 RS 矩阵受进程内多 M 臂序污染，作废，以 opus_day_rs_fresh 为准**。AG 矩阵同样每格重测中（D2）。

### F34 小 M 输区的上限分析（夜间矩阵 stock / gemm 下界，纸面）
- RS：完美隐藏上限 = stock / gemm8（8 块 GEMM 本身）。M4096 与 M8192 **11/11 格上限 > 1.05**（1.05–2.41），实测 dx 只有 0.41–1.21；M2048 只有 4/11 上限 > 1.05（7/11 格分块 GEMM 本身已比 stock 整体慢 ⇒ 切块路线在 M2048 物理上赢不了）。
- AG：上限 = stock / gemm1，M2048–8192 **22/22 格上限 1.15–3.36**，实测全输。
- 输的原因 = 每 op host 固定成本：dx host 1.27–1.57 ms/op（stock 0.07），纯 RS 33.6 MB dx host 0.92 ms 而 RCCL 全程 0.51 ms。例：qwen3-8b RS M8192 stock 1.008 / gemm8 0.503 / dx 1.569（host 1.339）。⇒ 小 M 区的损失是控制面（host 入队），不是物理上限；上限显示有空间。
- 候选小 M 技术：(a) 把一次融合 op 的 ~60 次 host 调用压成一次图重放（HIP graph；T 系列图捕获 POC 已过）；(b) 不切 GEMM，整块 vendor GEMM 后用单内核 SM 直写做 RS/AG（一次发射，host 不按对端发命令；SM 跨卡写 109 GB/s 已测）。⇒ 分派器从"开/关"扩展为三档选路（SM 单发 / DMA 分块重叠 / RCCL）。

### F35 小 M 候选 (a) 判决：整块 vendor GEMM + 单内核 SM 直写（opus_smrs.py，opus_smrs_gate.csv）—— **NO-GO**
- 3 模型（qwen3-8b / llama3-70b / llama3.1-405b）× M 2048/4096/8192 × {RS, AG-up, AG-gate_up} × grid {160, 320} = 54 格，**全部 CLEAN，0/54 ≥1.05，最好 0.951（405B gate_up M8192）**，小模型最差 0.63–0.79。grid 160→320 无改善（SM 写带宽不是瓶颈）。
- host 降到 0.25–0.34 ms/op（DMA 分块 1.3–1.6）——控制面问题解决了，但数据面输：70B RS M4096 原版通信段 ≈0.60 ms（1.439−0.836），我方 push+flag+wait+sum ≈0.73 ms。不切块 = 零重叠，只剩"通信本身比 RCCL 快"这一条路，而小载荷 RCCL 本就高效（纯 RS 33.6 MB 0.51 ms）。
- 协议定律：第一版内核内 acquire 自旋等对端旗（push 内等 ack、sum 内等 arr）**死锁超时**；改为流上 hipStreamWaitValue32（DX 已验证机制）即通。⇒ 本栈内核内跨卡自旋等待不可用（与 S18/S73 spin 三态机类同），等待只放流上。
- 按用户规则转候选 (b) 图捕获。**先验风险**：T18 v18graph（10-23）已在 v9ring 上做过全 op 图重放，正确但性能为负——图重放/节点派发税 ≈ +1.0–1.35 ms/op（v18_graph 4147 vs eager v9ring 2543 µs @70B M8192），PARK。小 M 的整 op 只有 1.4–2.8 ms，同量级的税会吃掉全部收益 ⇒ (b) 先做 30 分钟单卡探针测 DX 节点数下的图税，过门（图重放 ≤ eager 设备时间 + 0.1 ms）再写跨卡协议。

### F36 候选 (b) 预门：单卡图重放税探针（opus_graphtax.py / opus_graphtax.log）—— **PASS**
- DX RS 骨架（8 块 GEMM + 7 次 lane 拷贝 + 事件 + 求和，无跨卡协议），eager vs torch.cuda.graph 重放，3 模型 × M 2048/4096/8192。
- host：eager 0.83–0.96 ms/op → 图重放 **0.065–0.068 ms/op**（降 13×）。
- 图税（图重放 b2b − eager b2b）：−0.40 … +0.03 ms，**9/9 格在预注册门（≤ +0.1 ms）内**。T18 的 +1.0–1.35 ms 税在本 torch/DTK 栈 + 这个节点数下不复现。
- 注意：骨架的绝对设备时间无意义（torch.sum(slots.float()) 的求和远慢于生产 k_dx_sum），只看税。
- 下一步：写跨卡图版 DX（K 面票据 + 设备侧 round 计数，沿用 T18 v18graph 已验证的协议与五条定律），小 M 3 模型位级门 + 计时。

### F37 候选 (b) 判决：图捕获的跨卡 DX（opus_dxg.py，opus_dxg_sweep.log / opus_dxg.csv）—— **NO-GO**
- 协议：常值二元信号量版 DX（arr/ack 只取 0/1，可被图固化），全部位级 CLEAN。3 模型 × M 2048/4096/8192 = 9 格。
- 图版 dxg：**0/9 赢**，0.40–0.76×；eager 版 dxc（同协议）：70B M8192 1.17×、405B M8192 1.10×、405B M4096 1.01×、70B M4096 0.98×（b2b 1.09×），其余输。图版每格都比 eager 慢（70B M8192 3.96 vs 2.27 ms）。
- 三条平台定律（DTK 26.04）：
  1. **hipStreamWaitValue32 不进图**：捕获期调用直接落在活流上执行（第一次捕获即挂在 synchronize）。图里等待只能用 1-CTA spin 内核。
  2. **图执行器把捕获的多流 DAG 串行化**：不加"到达等待依赖本方推送"的边时重放必死锁（诊断：重放 8 s 后全 rank ack 未被清零一次 ⇒ lane 分支从未执行）。⇒ 图里分支不并发 ⇒ GEMM 与推送不再重叠 ⇒ 设备时间 +1.7 ms（70B M8192）。
  3. 这也重新解释了 T18 的"图重放税 +1.0–1.35 ms"：是**串行化丢掉重叠**，不是节点派发税（F36 单卡骨架无重叠可丢，所以测不出税）。
- 图重放 host 0.61–0.65 ms/op（eager 1.2–1.5），只降一半；主要损失在设备侧串行化。
- 按用户规则转"编译器方向"。可行路线判断：保留 eager 的多流并发（唯一能重叠的执行方式），把 ~45 次 Python→HIP 调用改成一次原生 C++ 调用发出（预估 host 0.2–0.3 ms）。上限参照 b2b（host 已被流水隐藏）：70B M4096 1.09×、405B M4096 0.95×、qwen3-8b M4096 0.47× ⇒ 只能救中大模型的中 M 格，小模型小 M 设备时间本身就输（块 GEMM 低效 + 拷贝干扰），应由分派器退回 RCCL。
- **F37b 双图形态（每条流各捕获一张线性图，流间用设备旗交接）—— 也死锁**：dxd（同结构 eager）跑通，dxg2 首次重放即挂在 host 侧的 replay 调用（7/8 rank 栈在 dual_replay）⇒ DTK 26.04 上图的启动在设备上是串行执行的，lane 图的自旋等待永远等不到 compute 图开跑。⇒ **图捕获路线在本平台关闭**：单图丢重叠（F37），双图互等死锁。

### F38 小 M 新候选：微批拆分 + RCCL 侧流重叠（opus_split.py，opus_split.log，ISO/TokenWeave/two-batch-overlap 同族；无对称堆、无逐对端 host 命令）
- 沿 B 维拆成 nb=2/4 份（SP 布局保持），第 h 份的 RCCL 集合与第 h+1 份的 vendor GEMM 在两条流上并发。3 模型 × M 2048/4096/8192 × {RS, AG} × RCCL 通道 {默认, 8, 4}，全部 CLEAN。
- **对默认通道的原版**（唯一公平口径；同通道比会夸大——通道数少时原版本身变慢，如 405B RS M8192 ch4 自比 1.39× 但原版慢了 40%）：每格取最好臂，**6/18 格 ≥1.03**：70B M8192 RS 1.049 / AG 1.069，405B M4096 RS 1.054 / AG 1.081，405B M8192 RS 1.124（ch8 ovl4）/ AG 1.065；70B M4096 RS 1.018。qwen3-8b 全输（0.75–0.95），M2048 全输。
- 拆分本身的代价（seq 臂，同流不重叠）0.8–0.99 ⇒ 拆分只在 GEMM 足够大时值得；ovl2 通常优于 ovl4。
- **与 DX 互补**：70B AG M8192 拆分重叠 1.069 > DX-AG 0.94–0.96；RS M8192 DX 1.16–1.17 > 拆分 1.05。⇒ 分派器的候选集应为三档：DX 分块直推（大 M）/ 拆分 + RCCL 侧流（中 M，尤其 AG）/ 原版 RCCL（小 M、小模型）。
- 全矩阵（11 模型 × M 2048–32768，默认 + ch8，每模型独立进程，2 轮）已排队（opus_split_matrix）。

### F39 今日复测链 D3/D4（opus_day1031/）
**D3 70B 列并行固定臂（f/c/f/c，16 iter，cp_fused 0 / 4096 已核）**：fbag 16423.0 / 16591.5（均值 16507），fbc 16172.9 / 16178.3（均值 16176）⇒ **列并行同批赢 −2.0%**（F27 为 −0.58%，同号更大）。
- ⇒ 夜间分派器（按需建 col）在 70B **两次都误丢 col**（+col 探针 +0.74% / +0.88%），选后 ~16862 vs 最优 fbc 约 −2% 处 ⇒ **70B 上分派器损失约 2.7%**。预热全开时 70B 选对、27B 选错（F31）；按需建时 27B 选对、70B 选错 ⇒ 两种预热都有偏差，col 的代价/收益与"col 上下文何时、以何种顺序建立"有关，不只取决于开没开。
- 诊断批已排（e2e_70b_atcol，E1 之后）：fbag / fbc / 钉死 row+bwd+col 且按需建（fatcl）/ 钉死 row+bwd+col 且预热建（fatcw）。判读：fatcl ≈ fbc ⇒ 探针计时窗口有问题；fatcl ≈ fbag+0.7% ⇒ 晚建上下文本身慢（分配顺序效应）。
**D4 非重叠成分 × 对称堆大小（70B MBS4，2 轮）**：base 8961.1 / 8950.5；BRS 包装（零接管）+6GB 堆 8907.4 / 8917.8（−0.5%）；+1GB 堆 9029.8 / 9027.8（**+0.8%**）。⇒ 非重叠成分随堆大小变号，来自引擎 bootstrap 后的内存布局，不是包装逻辑本身；与重叠收益无关，论文按空对照口径报即可，附一句说明。

### F40 E1 拆批切换点扫描（opus_day1031/split_e1.log，11 模型 × M 4096/8192/16384，默认通道，1 轮，每格取 nb=2/4 × seq/ovl 最好臂，对同进程原版）
| M | RS ≥1.03 | AG(up) ≥1.03 |
|---|---|---|
| 4096 | 2/11 | 1/11 |
| 8192 | 7/11 | 7/11 |
| 16384 | **11/11**（1.04–1.19） | **11/11**（1.05–1.24） |
- 小模型（qwen3-8b / llama3-8b / qwen3-32b）在 M8192 仍输，M16384 小幅赢（1.04–1.07）。
- **与 DX（今日独立进程）在 M16384 逐格比，赢家随模型变**：RS 上 70B DX 1.20 > 拆批 1.09，mistral-l DX 1.41 > 1.19，dsv3 DX 1.22 > 1.16；但 qwen3-8b 拆批 1.04 > DX 0.79，qwen3-32b 1.04 > 0.87，llama4-mav 1.07 > 0.92。AG 上 DX M16384 只有 13/22 赢，拆批 11/11 赢。
- ⇒ 三档的切换点不是一个 M 阈值，而是随（模型宽度 N/K, M）二维变化；正好是规划器要预测的对象。单轮数据，接近 1.0 的格待补第 2 轮。

### F41 70B 列并行诊断首轮（opus_day1031/e2e_70b_atcol，16 iter，第 2 轮在跑）
| 臂 | 含义 | ms/iter |
|---|---|---|
| f1 fbag | 固定 row+bwd | 16418.7 |
| c1 fbc | 固定 row+bwd+col（OPUS_CP=1，开局即建） | 16196.4（−1.35%） |
| l1 fatcl | 分派器钉死全开，col 上下文**按需晚建** | **16721.2（+1.8%，比 fbc 慢 3.2%）** |
| w1 fatcw | 分派器钉死全开，col 上下文**预热时建** | 16182.0（≈ fbc） |
- ⇒ **晚建的 col 上下文本身就慢 ~3%**（与列并行是否划算无关）：分派器按需建 col 时，+col 探针测到的是"晚建"的慢版本，于是在 70B 误丢 col。预热建则在 27B 上让常驻上下文拖慢不开 col 的探针（F29/F31）。两种方向的偏差同源：**对称堆/上下文的建立时机改变了后续所有引擎 op 的性能**（与 F10 臂序污染、F39 堆大小变号同一现象）。
- 修复方向：所有候选路径的上下文在预热时按固定顺序建立；不选的路径在决策后释放；探针只比较"上下文都已常驻"状态下的开关差。待验证：70B fatx（预热建 col → 钉 row+bwd → 释放）对 fbag 是否有残留。
- 峰值核查（今日独立进程，n=2）：GEMM+RS 最高 mistral-l M16384/32768 1.41×（原版随 M 线性，可信）；qwen3-8b M32768 1.465× **不可作为峰值**（该格原版 5.73 ms 是 M16384 的 3.1×，原版自身异常）。AG+GEMM 最高 dsv3-d M32768 1.355×（n=1）。纯 RS 537 MB 1.60×（n=2，与 F4 的 1.57× 复现）。

### F42 融合算子峰值口径待定（10-31 19:20）
- 统一矩阵（opus_dx_rs.py 测试台自带的 DX 实现，fp16）GEMM+RS 峰值 1.41×（mistral-l M16384/32768）。
- 生产算子（fork 的 dx_gemm_rs_op，bf16，opus_fused_gate*.py）在 27B 车辆形状 M32768 N5120 K2176：stock 6.194 / 6.269 → fused 3.476 / 3.507 ⇒ **1.78× 两次复现**（K640：1.55×）；位级门 120 op PASS。同一形状下测试台实现明显更慢 ⇒ 矩阵可能低估了生产算子。
- 已排 opus_day1031/prodop.log：生产算子 × 11 模型 × M 8192/16384/32768（每格独立进程），跑完后摘要峰值以生产算子矩阵为准。
- 70B 列并行诊断第 2 轮：fbag 16596.7 / fbc 16133.5 / fatcl（晚建）16747.3 ⇒ 复现 F41。
- **F41b 残留检查首轮（opus_day1031/e2e_70b_fatx）**：fatx 因 OPUS_AT_WARM_COL 默认已改 0，实际是"预热不建 col → 钉 row+bwd → 恢复原版 ColumnParallelLinear"：x1 16416.0 vs 同批 fbag 16412.2（+0.02%）⇒ 70B 上分派器包装与钩子本身零开销（27B F29 的 +0.6% 未复现）。原本要测的"预热建 col → 释放"另排 fatxw 批（opus_day1031/e2e_70b_fatxw，prodop 之后）。

### F43 晚间队列（10-31 19:27–20:37）
**残留检查（70B，16 iter）**
- fatx（预热不建 col → 钉 row+bwd）：16416.0 / 16648.3 vs fbag 16412.2 / 16565.5（+0.3%，噪声内）。
- fatxw（预热建 col → 钉 row+bwd → 释放 col 上下文）：16236.9 / 16468.0 vs fbag 16427.3 / 16450.9（−0.5%，噪声内）⇒ **70B 上"预热全建、决策后释放"无残留**。分派器修法定为：所有候选上下文预热时按固定顺序建好，探针在全常驻状态下比较，决策后释放未选路径。27B 上需复验（F29 的 +2.0% 是未释放时测的）。
**E1 第 2 轮**：两轮中位与首轮计数相同（RS/AG M4096 2/11、1/11；M8192 7/11、7/11；M16384 11/11、11/11）⇒ 切换点稳定。
**生产融合算子矩阵首轮（opus_day1031/prodop.log，fork dx_gemm_rs_op，bf16）**
- 26/33 格出数，**24/26 赢**；几何均值 M8192 1.146 / M16384 1.458 / M32768 1.802。7 个大格（mistral3-d/dsv3-d/70B/72B/mistral-l M32768、405B M16384/32768）无输出，原因未知（stderr 被过滤），已带完整日志重跑（prodop2）。
- 峰值：qwen3-8b M32768 2.48× **不用**（该格原版 6.966 ms 再次异常，同 F42）。可信高值：llama4-mav-d M32768 1.769×、dsv3-d M16384 1.745×、mistral-l M16384 1.700×、llama3-8b M32768 1.681× ⇒ 与 27B 形状 1.78× 同量级。
- **正确性门"FAIL"是判据问题，不是算子错**：判据为 rel(ours, RCCL) < 2e-2。宽 K 模型（gemma3-27b 起）原版 RCCL 自身对 fp32 的误差已达 1.9–2.4e-2，而我方对 fp32 只有 0.72–1.01e-2（更准：fp32 累加一次求和 vs RCCL bf16 逐跳累加）。⇒ 门改为对 fp32 参考判定（rel(ours,fp32) ≤ rel(rccl,fp32)，或 < 2e-2），重跑沿用旧脚本，判读时按新口径。
- **F43b 7 个大格无输出的原因**：dushmem_malloc 返回 NULL（门测脚本未设 DUSHMEM_SYMMETRIC_SIZE，默认堆装不下 M32768 / 405B M16384 的纯 RS 上下文 P·R·sb ≈ 2–4 GB）。不是算子问题。已排 prodop3：这 7 格 × 2 轮，堆 12 GB。

### F44 分派器修法在 27B 上不通过（opus_day1031/e2e_27b_fatxw，3GB 堆，16 iter）
- fatxw（预热建 col 上下文 → 钉 row+bwd → 释放 col 上下文 + 恢复原版 ColumnParallelLinear，日志确认 "released col: 1 ctx closed"）：34839.1 / 35084.9 vs 同批 fbag 34080.2 / 34205.7 ⇒ **+2.3% / +2.6%，两轮同号**。
- ⇒ 27B 上**释放后残留仍在**（与 F29 未释放时的 +2.0% 同量级）；70B 上同一修法无残留（F43）。
- 定律（与 F10 臂序污染、F39 堆大小变号、F41 晚建上下文慢 3% 同源）：**在本栈上，对称堆上下文一旦建立再释放，会持久改变后续引擎 op 的性能；影响大小随形状和堆大小而变**。在线探针必须建立候选路径的上下文，因而会扰动它要测的系统。
- 推论（规划层设计）：在线贪心探针在本平台上**原理上有偏**，无法靠"建立顺序/释放"修干净。规划层改为：(1) 代价模型离线预测，不建未选路径；(2) 需要实测时用独立进程短跑（每候选一个进程，几个 iter），再以选中配置启动正式训练。这一条本身是论文规划层的论据：探针会扰动被测系统。

### F45 生产融合算子 GEMM+RS 完整矩阵（prodop + prodop2 + prodop3，11 模型 × M 8192/16384/32768 = 33 格，每格独立进程，每格 2 轮中位；大格 12 GB 堆）
| M | 赢 | 几何均值 | 范围 | 纯重叠部分（对 GEMM + 我方纯 RS 串行）几何均值 |
|---|---|---|---|---|
| 8192 | 8/11 | 1.147 | 0.92–1.40 | 1.210 |
| 16384 | **11/11** | **1.444** | 1.26–1.73 | 1.381 |
| 32768 | **11/11** | **1.680** | 1.42–1.81（qwen3-8b 2.48 除外） | 1.465 |
- 可信峰值：**dsv3-d M32768 1.809×**（原版 8.918 / 8.910 ms，为 M16384 的 1.96×，正常；两轮 1.81 / 1.81）。其次 llama4-mav-d 1.774×、mistral3-d 1.739×、27B fc2 形状 1.78×（F42 门测）。qwen3-8b M32768 2.48× 不用（原版异常）。
- **数值精度 33/33 格我方优于 RCCL**（对 fp32 参考：我方 0.62–1.02e-2，RCCL 1.45–2.74e-2）。旧门"对 RCCL < 2e-2"的 FAIL 全部是 RCCL 自身误差大导致，算子正确。论文可写：fp32 一次求和比 bf16 逐跳累加更准。
- 统一矩阵（测试台实现）M16384 几何均值 1.05、生产算子 1.444 ⇒ Evaluation 的算子级热力图改用本矩阵。
- E4 基线公平性批（selective recompute 下原版是否放得下、多快）已于 22:56 开跑。

### F46 E4 基线公平性（selective 重算）——结案
- 70B MBS8（M32768，头条形状）：原版 selective 重算 **OOM**（"HIP out of memory… 63.98 GiB total, 56.52 GiB allocated"）⇒ 评测规模下 full 重算是唯一可行配置，对照公平。论文一句话："在评测规模下 selective 重算超出显存，基线与我们均采用 full 重算"。
- 70B MBS2（M8192）：原版 selective 3667.9 / 3693.8 vs full 4774.2（快 23%）；我们 selective 3789.0 / 3792.3（输 3.0%，与 full 下输 3.1% 同）⇒ 小 M 结论不变（分派器退回原版）。
- 27B 部分按用户决定取消（节约卡时）。车辆脚本新增 VEH_RECOMP 开关（默认 full，备份 opus_arm_inner.sh.bak_1031）。

### F47 预测器（代价模型 + 规划器）v2：留一模型验证过"永不比原版慢"门（CPU，planner/，只用已测的两张算子级表）
- 改动：(1) 相对误差加权拟合；(2) 拆批模型加"每份 GEMM 大小"项（小切片低效）；(3) 带误差界的保守选择：候选方案的悲观时间 = 预测 × exp(训练集上该方案 80% 分位的乐观偏差)，只有它仍比原版快 2% 以上才选，否则退回原版；(4) 数据清洗：拆批测试台单次/连发比值相差 >1.5 倍的格子剔除（gemma3-27b RS M8192），gate_up 不再借用 up 的拆批数据。
- 结果（97 个调用点，11 个模型，每次留出一个模型测试）：
| 计时 | 规划器 | 总用我们 | 静态 M 阈值 | 事后最优 | 选得比原版慢 |
|---|---|---|---|---|---|
| 连发 | **1.139×** | 1.101× | 1.120× | 1.167× | **0** |
| 单次 | **1.142×** | 1.029× | 1.100× | 1.163× | **0** |
- 收益捕获率（相对事后最优）连发 83%、单次 87%；dx 时间预测误差 MAPE 9.2% / 6.7%。
- 安全旋钮扫描（sweep_guard.py）：不加保护时连发 1.149× 但有 4 格比原版慢（最多 8.1%）；q=0.8 降到 0 格，只损失 1 个百分点 ⇒ 论文可画"收益–安全"权衡曲线。
- 剩余短板：选错最大损失 24.6%，主要是 AG 大格保守地退回了原版（错失收益，不是变慢）。

### F48 预测器第二级：整步控制面项（planner/step_planner.py，CPU，只用已有数据，0 卡时）
- 问题：只用算子级数据时，预测器在 27B 和 70B 都选 fbc（四处全用重叠方法）；27B 实测 fbc 比 fbag（RS 两处用重叠方法，AG 两处用原版）慢 2.4–3.1% ⇒ 27B 选错。
- 机理：融合 AG 调用要在 GEMM 之前发 7 次推送，训练里每次推送阻塞 host（引擎队列反压 + 跨 rank ack 链），host 每次调用 3.4–4.9 ms，原版 0.35 ms（F17/F19）；融合 RS 的推送在 GEMM 之后，被自己的 GPU 时间遮住。host 领先 GPU 的余量是整层共享的预算，只有超出余量的部分才拖慢整步。
- 模型（每个层周期 = 一个 micro-batch 过一层的前向 + 重算 + 反向）：整步变化 = ΔD + λ·max(0, ΔH − ΔD − S)。ΔD 来自算子级代价模型；ΔH 来自平台常数（每种调用的 host 发命令时间）；S = 原版的 host 余量，S(模型) = S_27B + (该模型每层周期 GPU 时间 − 27B 的)。
- 拟合只用 27B 两组 E2E（只 QKV 融合 +0.99%、QKV+fc1 融合 +3.05%）：S_27B = 6.08 ms，λ = 0.599。**27B 是样本内（2 个参数配 2 个点，必然精确），不算检验**。
- **70B 是样本外检验**：预测 fbc 相对 fbag −1.38%，实测四次 −0.58 / −2.00 / −1.35 / −2.79%（均值 −1.68%）⇒ 方向和量级都对。
- 选择（预测加速不足 1% 不开，防 E2E 噪声误判）：27B → fbag，70B → fbc，与两边实测最优一致。E2E 数字直接用已有结果，不需要新跑。
- 局限：(1) host 常数只在 27B 测过，当作平台常数用于 70B；(2) 余量随 GPU 时间线性增长是假设；(3) "只 fc1 融合"等配置未测；(4) 只有 2 个模型，检验点只有 1 个。

### F49 11-01 批次 A/B：融合 AG 的 host 时间不是平台常数；整步线性余量公式被否定（opus_day1101/A_70b_host、B_27b_mbs4）
- **A（70B MBS8，fcth/fchs，8 iter，8 rank × 5 次 256 调用的中位）**：融合 AG host 时间 QKV 9.89 / fc1 6.72 ms，原版 0.334 / 0.336 ms。27B 同口径（opus_vehf9 cth1/chs1）为 4.61 / 3.62 vs 0.338 / 0.344 ms。70B/27B = 2.15× / 1.85×，推送字节只多 1.6×，每层周期 GPU 时间多 2.06× ⇒ **融合 host 时间 ≈ η × 每层 GPU 时间**（η QKV 0.033、fc1 0.024），原版 host 时间不变。机理假设：单缓冲槽位，第 n 次推送等第 n−1 次的确认，未满足的流等待之后发引擎拷贝阻塞 host ⇒ host 每次融合调用都等 GPU 追上（提前量清零），阻塞时长 = 提前量 ∝ GPU 时间。
- **B（27B MBS4，fbag/fbc ABAB，14 iter）**：fbag 19068.4 / 19033.5，fbc 19678.0 / 19666.6 ⇒ **fbc +3.26%**（两遍极差 ≤0.2%），loss 同带。跑前登记（B_preregistered_two_hypotheses.md）：常数假设 +12.0%、比例假设 +5.3%，按实测整步重算 +11.0% / +5.4% ⇒ **两个都错**。27B MBS8 +3.05% vs MBS4 +3.26%：每层 GPU 时间减半，损失百分比不变 ⇒ "余量 = S_27B + 每层 GPU 时间差"被否定；同一模型的损失 ∝ 每层 GPU 时间。
- 预测器第二级须重做（批次 C 的 host 时间数据一起用）。step_planner.py 已改为 host 时间按 η × D_cycle（SP_HOST_MODEL=prop，const 保留对照），线性余量公式待替换。

### F50 11-01 批次 C/D/E/F + 2 卡探针：融合 AG host 阻塞的机理定位（三个改法全部否决，根因 = 跨进程对端内存的引擎拷贝）
- C（OPUS_AG_PLANES=2/4，槽位多平面）：位级 PASS；车内 host QKV 5.06 / 4.85、fc1 3.64 / 3.57 ms（不变），整步 35455 / 35416 vs 同批 fbag 34373（+3.1–3.2%）⇒ "等上一次确认"假设否决。第二轮已截停。
- D（OPUS_AG_GATE=ev，等确认移门流）：位级 PASS；车内 host QKV 4.50 ms，整步 36253 ⇒ 否决。
- E（OPUS_AG_FORK=wv，分叉/汇合改本卡 WriteValue32+WaitValue32）：位级 PASS（含顺序压力模式：复用输入缓冲 + 延迟 GEMM + 无同步）；孤立连发 QKV 1.11→1.23–1.30×；车内 host QKV 4.66 ms，整步 35464 ⇒ 否决。
- F（hb 拆账，FORK=wv）：host/call 4.3–5.5 ms，其中 push_data 3.2–4.4 ms，其余项与旧写法同 ⇒ 卡的就是推送调用本身。
- 2 卡探针（opus_copycap*.py，GPU0 先睡 100/400 ms 再发 40×7 次拷贝）：
  - 同进程对端 torch 内存（peer access）：不卡（cc4，值分叉 1.5–6 ms）；
  - 跨进程对端内存（dushmem 对称堆 host.ptr，或 hipIpcOpenMemHandle 导入的普通 hipMalloc）：卡 host，时长 ≈ GPU 落后量的 ~25%（100 ms → 30 ms，400 ms → 100 ms），与拷贝次数（10 vs 40 次）、大小（1 vs 8 MB）、分叉写法（事件 / 写值）、API（hipMemcpyAsync D2D / Default / DtoD / PeerAsync / 反向拉取）都无关；
  - 本卡对称堆：不卡。
  ⇒ **定律：本平台 HIP 对跨进程导入的对端内存做引擎拷贝时，host 会被同步到 GPU 进度附近**（提前量清零）。重叠方法的所有对端推送都属此类（AG 和 RS 都是），AG 先推后算所以直接暴露，RS 先算后推被自己的 GEMM 盖住一部分。
- 推论：用流/分叉写法改不掉。剩余可能的出路：① 推送改成 SM 拷贝核（不调用 hipMemcpy，但占 CU，T21 v19 与 F17 SM staging 都有代价）；② 独立 host 发射线程（F19b 已楔死，需换队列设计）。三个开关默认关，生产路径未变。

### F51 11-01 批次 G：70B MBS4 / MBS2 判别（opus_day1101/G_70b_mbs{4,2}，fbag vs fcth ABAB，14 iter）
- 70B MBS4：fbag 8561.3 / 8604.8，fcth 8772.3 / 8773.4 ⇒ **+2.21%**（两遍 +2.46 / +1.96）。70B MBS2：fbag 4905.5 / 4938.2，fcth 5157.0 / 5156.1 ⇒ **+4.77%**。loss 同带。
- 跑前登记（G_preregistered.md）：H1"由每层 GPU 时间决定" +4.4% / +4.7%，H2"由模型决定" −0.6% / +0.9% ⇒ **判 H1**（MBS4 幅度偏小，MBS2 吻合）。同一 70B 模型：MBS8 选 fbc，MBS4/MBS2 选 fbag ⇒ 选择必须按形状定。
- 车内融合 AG host 中位：70B MBS4 QKV 4.73 / fc1 3.00 ms，MBS2 2.02 / 1.77 ms（MBS8 9.89 / 6.72）⇒ host 时间 ∝ 每层 GPU 时间再次成立（A 的结论跨 MBS 成立）。
- 反推每次融合 AG 调用 GPU 空等 e（step_planner_v3）：27B MBS8 1.69 / MBS4 0.65，70B MBS8 −0.13 / MBS4 0.96 / MBS2 0.73 ms。e 在每层周期 GPU 时间 ≤137 ms 时为正，273 ms 时为 0。L2（e = k·D，D < D* 时；否则 0）全体 5/5 选对，留一 4/5（唯一错的是 70B MBS8：它是唯一 D > 137 的点，留一时截止点无从估计 = 外推）。截止点只被夹在 137–273 ms 之间。
- 未解释：今日 fbag 比 10-29 同配置快（70B MBS4 8583 vs 9095，−5.6%；MBS2 4922 vs 5300，−7.1%）。G 只用同批相对值，不受影响；绝对值变化来源未查（需同批 base 复测才能判断，未做）。

### F52 11-01 双机 16 卡（TP8 × DP2，RDMA）70B：重叠方法与预测器都在跨机下成立（opus_dual16/rdma_liqsh2-81125/70b）
- RDMA：两节点 `Using network IBext_v8`、`NET/IB : Using shca_0..3`、跨机通道 `via NET/IBext_v8/0..3/GDRDMA`；日志另有 "GPU Direct RDMA Disabled for HCA 0..3"（各 80 行），只影响跨机 DP 梯度同步（RCCL 原版，所有臂相同），不影响机内 TP 的重叠方法。
- 70B NL8 MBS8，16 iter，iter3+ 中位，两遍：BASE 17949.1 / 17951.4，NULL 18127.2 / 18142.2，ALT（= FBAG，预测器没选）17459.5 / 17365.5，PLAN（= FBC，预测器选，两节点一致）16814.2 / 16806.4。
- ⇒ PLAN 对 BASE **1.068×**，对空对照 **1.079×**（8 卡 70B fbag 对空对照 1.077×）；**PLAN 比 ALT 快 3.46%，两遍同号**（判定规则：两遍都快 = 选对）⇒ 预测器第二个样本外检验点通过（参数全来自 8 卡；输入 = 16 卡 BASE.1 实测整步）。
- 两遍极差：BASE 0.01%、NULL 0.08%、ALT 0.54%、PLAN 0.05%。空对照比原版慢 1.0%（8 卡上快 0.1–0.6%，符号随环境变，F30/D4 已知），所以纯重叠收益按对空对照报。
- 仪器注意：Megatron 迭代行只在全局最后一个 rank（node1/out_r7）打印，node0 预测器读不到 BASE.1 整步，用 GEMM 粗估（70B 粗估 D=340 ms 与实测 264 ms 同侧，选择一致）。作业结束后修：两节点用同一来源的整步时间。
- **27B（同作业，NL32 MBS8，两遍）**：BASE 37540.9 / 37483.8，NULL 37443.7 / 37950.2，ALT（= FBC，预测器没选）36741.0 / 36819.6，PLAN（= FBAG，预测器选，两节点一致）35771.2 / 35752.4 ⇒ PLAN 对 BASE **1.049×**、对空对照 **1.054×**（8 卡 27B fbag 对空对照 1.057×）；**PLAN 比 ALT 快 2.77%，两遍同号** ⇒ 第三个样本外检验点通过。27B 与 70B 最优配置相反（27B 选 FBAG、70B 选 FBC），预测器两边都选对。空对照第二遍偏慢 1.3%（两遍极差最大的一臂），27B 对空对照比值按两遍均值。
- 作业 2026-11-01 17:50–20:26（2 h 36 min，16 卡 ≈ 42 卡时），16 臂全部 16 iter 完成，0 NaN、0 报错。汇总表：opus_dual16/rdma_liqsh2-81125/summary.md。

### F53 11-01 晚 MoE 单卡：免同步、免丢 token 的专家分组 GEMM（fork src/stowaway/kernels/moe_ggemm.py + csrc/moe_ggemm.cpp）
- 背景：Megatron 每层 `tokens_per_expert.tolist()` 把专家 token 数同步回 host 再逐专家 GEMM（F24 车辆输的根源之一）；`torch._grouped_mm` 在 ROCm 报 "not supported"，TE 未装。
- 做法：hipBLASLt ext GroupedGemm + **设备侧 UserArguments**（每组的行数和 A/B/D 指针由一个 triton 核在 GPU 上写，host 从不读 token 数）。两个定律：① 每组行数不能超过建图时的 host 行数（超出部分结果错），② 建图时各组行数不一致会选到慢 2× 的算法。⇒ **虚拟分组**：每个专家在设备上切成 ≤B 行的块，G = E + ceil(T/B) 组 × B 行的固定网格覆盖任意路由，空组 0 行。UserArguments 在 DTK 26.04 是 196 B 紧凑结构，字段按字节偏移写；参数填充合成一个 triton 核（逐字段 torch 写要 0.32 ms，合成后 ≈0）。
- 单卡实测（16 本地专家，32768 行，Qwen3-30B-A3B 专家形状，bf16，连发 30 次）：
| 形状 | 路由 | 我们（B=2048） | Megatron 逐专家 + 同步 | 同行数稠密 GEMM |
|---|---|---|---|---|
| fc1 K2048→N1536 | 均匀 / 对数正态 / 单专家 14× | 0.692 / 0.762 / 0.781 ms | 1.325 ms | 0.564 |
| fc2 K768→N2048 | 均匀 / 对数正态 / 单专家 14× | 0.571 / 0.599 / 0.634 ms | 1.154 ms | 0.383 |
  ⇒ **1.7–1.9×**，对任意路由都正确（对 fp32 参考相对误差 1.9–3.1e-3，与 torch 同级或更好），**不丢 token、不同步 host**。B=512/1024 选到慢算法（1.07–1.76 ms），B=2048/4096 最好。
- 预测器（每个专家收到多少 token）在路由轨迹上的结论（Qwen3-30B-A3B 推理，24 步 × 48 层）：专家负载很偏（p99 4.1× 均值，最大 14.7×），同层相邻步相关 0.749，可预测；若按容量补齐，按预测定容量（滑动平均 × 1.25）在 10% 溢出下只需 1.27× 行数，固定容量要 2.0×。**但有了上面的虚拟分组（固定网格、免丢 token），专家 GEMM 不再需要容量**；接收缓冲按 rank 定尺 F18 已证明预测无增益（静态 1.5–2× 就够）。⇒ 负载预测在本设计里的剩余价值只有：按形状选 B / 算法（代价模型问题，同稠密第一级），以及 8 卡上按热专家排 dispatch 顺序（未测）。
- 待做：反向（dgrad 同形；wgrad 是"按组可变 K"的分组 GEMM，UserArguments 的 k 字段同样可设备侧写）；接入 Megatron experts.py（替换 tolist 路径）；8 卡 dispatch/combine 免同步与重叠。

### F54 11-01 晚 MoE 单卡（续）：反向 + 接入 Megatron + 单卡车辆 A/B + MoE 侧规划
- **反向**（src/stowaway/kernels/moe_expert_mlp.py）：dgrad 复用虚拟分组；wgrad 每专家一组、可变 k（行数）由设备侧写入 UserArguments.k。16 专家 32768 行，三种路由下与 fp32 参考误差和 torch bf16 同级；0 行专家的梯度为 0。
  - 反向 dgrad+wgrad：fc1 1.60–1.81 vs Megatron 2.59–2.88 ms；fc2 1.26–1.35 vs 1.87–2.02 ms。
  - 专家 MLP 整体（fc1 → SwiGLU×probs → fc2，前向 + 反向）：9.71 vs 13.21 ms（1.36×），y / dx / dW1 / dW2 误差与 torch bf16 同或更小。
- **接入 Megatron**（integrations/opus_moe_gg_patch.py，OPUS_MOE_GG=1，默认关）：替换 SequentialMLP.forward；权重与梯度仍是各专家自己的参数（DDP 钩子不变）；fp8 / bias / 专家 TP / 非 SwiGLU 自动回退。
- **单卡车辆 A/B**（e2e/megatron/run_moe.sh (SW_NP=1) → results/opus_moe1gpu/，EP=1、16 专家 top-8、NL12、MBS1 = EP8 车辆每卡的专家工作量；12 iter，ABAB）：
| 臂 | 整步（iter3+ 中位） | 专家阶段（每层调用）设备 / host | 每层 MoE 合计设备 / host |
|---|---|---|---|
| 原版 b1 / b2 | 8366.2 / 8297.6 ms | 6.28 / 6.31、5.86 / 5.98 ms | 8.69 / 16.9、8.22 / 16.8 ms |
| 我们 g1 / g2 | 7013.5 / 7012.9 ms | 2.17 / 0.83、2.06 / 0.77 ms | 4.56 / 13.0、4.47 / 13.1 ms |
  ⇒ **整步 1.19×（−16%）**，专家阶段设备时间 **2.8×**（6.0 → 2.1 ms），host 时间 6.1 → 0.8 ms。loss 12 步逐步对齐（差 ≤ 2e-5 相对，bf16 求和顺序级）。
  - 剩余的 host 大头是 dispatcher 的 post 阶段（~9–11 ms/层调用 host，内含 D2H 同步）——EP=1 时也存在，要靠 8 卡上的免同步 dispatch 去掉。
- **MoE 侧规划**（planner/moe_plan.py，单卡几秒，不建通信上下文；结果库 moe_plan.json）：按形状实测 stock 与 gg[B=1024/2048/4096/8192]（前向 + 反向），选最快：
| 形状（本地专家 E，H，专家 ffn，行数） | 选择 | 对原版 |
|---|---|---|
| 16，2048，768，8192 | gg2048 | 2.71× |
| 16，2048，768，32768（Qwen3-30B-A3B 每卡） | gg2048 | 1.32× |
| 16，2048，768，65536 | gg2048 | 1.20× |
| 16，4096，1536，32768 | gg4096 | 1.21× |
| 32，7168，2048，32768（DeepSeek-V3 式宽专家） | gg1024 | 1.09× |
| 8，4096，14336，16384（Mixtral 式大专家） | gg1024 | 1.01× |
  ⇒ 收益随"每专家行数小、专家多"增大（省的是 host 同步 + 逐专家小 GEMM 的固定开销）；大专家少专家时接近持平。B 的最优值随形状变（1024 / 2048 / 4096）。

### F55 11-02 8 卡 MoE 整层免 host 同步（src/stowaway/kernels/moe_sf.py + integrations/opus_moe_sfl_patch.py，OPUS_MOE_SFL=1）
- 设计：路由器不变（含 aux loss）→ top-k 编号/权重 → SM 直写 dispatch 进专家卡对称堆（设备原子领槽、release 旗 + 流等待、无 count 交换、无 .tolist()）→ 设备侧按本地专家排序 → 分组专家 MLP（设备侧计数，F53/F54）→ 按 4B 标签写回源行 → 加权求和。反向同两平面反向。每个方向都有 ACK 面（接收方 → 写方），缓冲复用、轮次单调，前向/重算/反向调用序全员一致。
- **8 卡算子级门**（opus_moe_sf_gate.py，Qwen3-30B-A3B 层：128 专家/16 本地，top-8，H2048，专家 ffn 768，真实路由 4 个偏斜切片，每轮新 token / 权重 / 上游梯度，对 fp32 参考）：
| T/卡 | 正确性 | 前向 ours / RCCL 路径 | 前向+反向 ours / RCCL 路径 |
|---|---|---|---|
| 1024 | 12/12 PASS，溢出 0 | 2.81–3.20 / 4.24–4.94 ms（**1.51–1.56×**） | 7.03–7.90 / 10.99–11.65 ms（**1.48–1.57×**） |
| 4096 | 12/12 PASS，溢出 0 | 8.67–9.55 / 10.38–13.04 ms（**1.20–1.37×**） | 21.8–24.8 / 25.5–31.8 ms（**1.17–1.28×**） |
  out / dx / d路由 / dW1 / dW2 误差全部 ≤ RCCL 路径（常更小，后者 bf16 逐跳累加）。赢幅随偏斜增大（RCCL 路径对偏斜更敏感）。
- **8 卡车辆 EP8**（Qwen3-30B-A3B 形状裁 12 层，MBS1，10 iter，iter3+ 中位，opus_moe_sfl_ab）：原版 1535.1 / 1541.3 ms，**整层免同步 1376.0 / 1381.8 ms ⇒ −10.3%（1.115×）**，两遍极差 0.4%；末步 loss 1.151355 / 1.151356 vs 原版 1.151363 / 1.151347（同带）；0 回退、0 溢出。旧版只换数据面 +13.5–18%（F24）⇒ 去掉同步本身才是收益来源（F24 预判成立）。
- gg 臂（只换专家 GEMM、保留原 dispatcher）首轮被"每次调用行数变 → 新桶 → 中途建 hipBLASLt 计划 + triton 重编译"污染（iter4–6 5–11 s）；改固定行预算（OPUS_MOE_GG_MAXROWS=98304）重测（opus_moe_sfl_ab2）：原版 1532.7 / 1526.0，gg 1489.4 / 1479.9 ⇒ **−3.0%**，loss 同带；仍有 iter5–6 两步 ~7.7 s 的一次性尖峰（原因未查，sfl 臂没有）。⇒ 车辆 10.3% 的分解：专家侧免同步 ≈ 3%，dispatch/combine 免同步 ≈ 7%。
- 今日 MoE 8 卡用时约 35 分钟（≈ 4.7 卡时）。
- **撞车检查（11-02，related_work_1102/）**：
  - 免同步 a2a（C2）= **撞车**：DeepEP 低延迟（2025-02，最坏缓冲 + 原子领槽）、MORI-EP（recv 数留设备、src 位置编码原位合并 ≈ 我方 4B 标签）、Primus Turbo-DeepEP num_worst_token（2025-09，训练）、Megatron HybridEP/ECHO（2025-11）⇒ 论文中**不列为贡献**，作为系统组件 + DCU 实现。
  - 免同步分组专家 GEMM（C1）= 目标撞车、机制部分重叠：hipBLASLt UserArguments 官方示例、Primus-Turbo CK 分组 GEMM（设备侧 offs）、Megatron Core MoE 报告（2026-03）⇒ 只剩"虚拟分组覆盖任意路由（绕开 hipBLASLt 组数 host 固定）"这一移植技巧。
  - MoE 侧按形状选路（C3）：常规自动调参，并入预测器方法论，不单列。
  - 稠密 A：FiCCO（2025-12，AMD，DMA 重叠、不改 hipBLASLt、带启发式；明确不做 RS）+ PyTorch Async-TP（行块 + 拷贝引擎 + 置换序，但每步 barrier）⇒ 卖点收窄为"首个无屏障 DMA 推送的 GEMM+RS 在训练端到端兑现"，必须引用二者并加 FiCCO 对比节。
  - 预测器 B：控制面项（跨进程引擎拷贝的 host 阻塞）+"在线探针有偏"未见先例；须与 FiCCO 启发式（未见场景 81%）正面对比。
