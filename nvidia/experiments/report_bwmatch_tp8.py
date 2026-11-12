"""Produce an evidence-based progress table from the serial TP8 run logs."""
import json
import math
import re
import statistics
from pathlib import Path

RAW = Path('/root/data/stowaway-nvidia/results/raw')
OUT = Path('/root/Stowaway-NVIDIA/results/单机8卡原表对照_20261111.md')
VLLM = Path('/root/data/stowaway-nvidia/results/tp8_20261111')
RUN = re.compile(r'bwmatch_(27b|70b)_mbs(\d+)_(stock|row|all|row_qkv|null)_(a\d+)\.log$')
ITER = re.compile(r'iteration\s+(\d+)\s*/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+).*?lm loss:\s*([0-9.eE+-]+).*?grad norm:\s*([0-9.eE+-]+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)')
COUNT = re.compile(r'STOWAWAY_DIAGNOSTIC_COUNTS rank=(\d+) row=(\d+) column=(\d+)')

def fmt(value):
    return f'{value:.1f}' if value is not None else '—'

def main():
    status = {}
    path = RAW / 'bwmatch_tp8_queue_status.tsv'
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            parts = line.split('\t')
            if len(parts) == 7 and parts[5] == 'end':
                status[tuple(parts[1:5])] = parts[6]
    runs = {}
    for p in sorted(RAW.glob('bwmatch_*_a*.log')):
        match = RUN.fullmatch(p.name)
        if not match:
            continue
        model, mbs, arm, rep = match.groups()
        text = p.read_text(errors='replace')
        rows = ITER.findall(text)
        complete = bool(rows) and [int(r[0]) for r in rows] == list(range(1, int(rows[0][1])+1))
        healthy = complete and all(not int(r[5]) and not int(r[6]) and math.isfinite(float(r[3])) and math.isfinite(float(r[4])) for r in rows)
        rc = status.get((model, mbs, arm, rep))
        # The first baseline was started outside the queue; its successful exit
        # is recorded by the smoke wrapper driver log.
        driver = p.with_name(p.stem + '_driver.log')
        if rc is None and driver.exists() and '[after training is done]' in driver.read_text(errors='replace').lower():
            rc = '0'
        counts = COUNT.findall(text)
        median = statistics.median(float(r[2]) for r in rows[2:]) if healthy and rc == '0' else None
        runs[(model, int(mbs), arm, rep)] = dict(path=p, rows=rows, rc=rc, median=median, counts=counts, healthy=healthy)

    lines = ['# 端到端与预测器结果表（NVIDIA A800，单机 8 卡）', '',
             '本表由原始日志生成；只收录完整结束、逐步记录齐全、无跳步和 NaN 且退出码为 0 的训练结果。速度比为同批次原版 / 候选，取每臂第 3 步起中位。部分臂尚在排队；不跨硬件或不同载具求加速比。', '',
             '训练均为 mock 数据、随机初始化、BF16、TP8+SP、全重算、seq4096、GBS=8×MBS，8 个 microbatch/步。27B 为 H5120/FFN17408/40 头/KV8/32 层；70B 为 H8192/FFN28672/64 头/KV8/8 层。它们是 BW 表的 dense 载具，非完整模型，也不是实权重训练；上游 Megatron 与原 Megatron-das 存在实现差异。', '',
             '原表术语：base = 原版；fnull = 空对照；fbag = RS 两处；fbc = 四处；fcq = RS 两处 + QKV。以下 dense 载具表中以“fbag 对应前向”“fbc 对应前向”“fcq 对应前向”明确标注的测量仅覆盖前向，**不等同原表的完整 fbag/fbc/fcq**。后向 AG+dgrad 与独立 BRS 此后已集成到 Qwen 训练载具并开展实验，见 [Qwen 训练结果](端到端与预测器结果表.md)。fnull 若测出，也只初始化 NVSHMEM/包装而不预分配全部 DX 缓冲，并非严格空对照。', '',
             'Qwen3.6-27B 文本配置的 64 层随机初始化训练载具，seq4096、MBS1/GBS1，行 DX 两轮 speedup **1.024× / 1.012×**，合并 **1.018×**。该实验使用 mock 数据，未加载真实权重；与下面原表 dense 27B 载具不同。详见 [Qwen 训练结果](端到端与预测器结果表.md)。', '',
             '## 1. 总表与单机 8 卡：重叠方法', '',
             '| 载具 | MBS | 臂 | 首轮 ms | 次轮 ms | 首轮相对原版 | 次轮相对原版 | 实际覆盖和状态 |',
             '|---|---:|---|---:|---:|---:|---:|---|']
    for model, mbs in [('27b',8), ('27b',4), ('70b',8), ('70b',4), ('70b',2), ('70b',1)]:
        for arm in ('stock','null','row','all','row_qkv'):
            if arm == 'row_qkv' and (model,mbs) != ('27b',8):
                continue
            items = [runs.get((model,mbs,arm,rep)) for rep in ('a1','a2')]
            values = [x['median'] if x else None for x in items]
            if not any(items):
                continue
            ratios = []
            for index, rep in enumerate(('a1','a2')):
                base = runs.get((model,mbs,'stock',rep))
                b = base['median'] if base else None
                ratios.append(f'{b/values[index]:.3f}×' if b is not None and values[index] is not None else '—')
            states = []
            for rep, item in zip(('a1','a2'),items):
                if item and item['median'] is None:
                    states.append(f"{rep}: {len(item['rows'])} 步，rc={item['rc'] or '运行中/未记录'}")
                if item and item['median'] is not None and arm != 'stock':
                    counts = item['counts']
                    rank0 = next((c for c in counts if c[0] == '0'), None)
                    states.append(f"{rep}: {len(counts)} rank 命中记录" + (f", rank0 行/列={rank0[1]}/{rank0[2]}" if rank0 else ''))
            labels = {'stock':'base（原版）','null':'fnull（部分对应）','row':'fbag 对应前向','all':'fbc 对应前向','row_qkv':'fcq 对应前向'}
            lines.append(f'| {model} | {mbs} | {labels[arm]} | {fmt(values[0])} | {fmt(values[1])} | {ratios[0]} | {ratios[1]} | {"；".join(states) or "完成"} |')
    lines += ['', '## 3. 单机 8 卡：AG 两处的选择（预测器第二级实测点）', '',
              '上表 fbag/fbc 对应前向的同配置整步可用于初步比较 QKV、fc1 前向选择；原表的完整后向与独立 RS 未覆盖，不能作为第二级原表数据直接代入。', '',
              '## 6. 单机 8 卡 vLLM 推理（Llama3-70B prefill）', '',
              'BF16、TP8、80 层、eager，max batched tokens 16384。使用线性 RoPE 扩展；时延含生成调度和 1 token decode；吞吐为 16 请求总时间。按原表术语 stock 是原版，rccl 指序列并行且四处 NCCL（A800 使用 NCCL），dx 是四处 DX；plan（预测器）由 A800 第一级预测器选出，与 plan 对应位置（非预测器）分别记录。', '',
              '| 臂 | 轮次 | ttft ms | ttft2 ms | 16 请求 ms | speedup：ttft / ttft2 / 吞吐 | 结果文件 |', '|---|---|---:|---:|---:|---|---|']
    vresults = {}
    inference_files = sorted(VLLM.glob('vllm_bwmatch_*_a*.json'))
    baselines = {}
    for p in inference_files:
        d = json.loads(p.read_text())
        if d.get('arm') == 'stock':
            group = 'rope4' if d.get('rope_factor') else 'unscaled'
            baselines[(group, p.stem.rsplit('_',1)[1])] = d['summary_ms']
    for p in inference_files:
        try:
            d = json.loads(p.read_text())
            group = 'rope4' if d.get('rope_factor') else 'unscaled'
            if group == 'unscaled':
                continue
            vresults[(group, d['arm'], p.stem.rsplit('_',1)[1])] = d
            s=d['summary_ms']
            baseline = baselines.get((group, p.stem.rsplit('_',1)[1])) or baselines.get((group,'a1'))
            speedup = ' / '.join(f"{baseline[k]/s[k]:.3f}×" for k in ('ttft','ttft2','thru')) if baseline and group != 'unscaled' else '—（未扩展上下文 NaN）'
            ilabel = {'stock':'stock','stock_sp':'rccl（NCCL）','row':'plan 对应位置（非预测器）','all':'dx','plan':'plan（预测器）'}[d['arm']]
            lines.append(f"| {ilabel} ({group}) | {p.stem.rsplit('_',1)[1]} | {s['ttft']:.1f} | {s['ttft2']:.1f} | {s['thru']:.1f} | {speedup} | `{p}` |")
        except (ValueError, KeyError):
            lines.append(f'| 错误 | — | — | — | — | — | `{p}` 无法解析 |')
    lines += ['', '### 6.2 输出一致性（4×48 token）', '',
              '| 臂/轮次 | 可比较提示 | 一致 token / 总数 | 首 token 最大 logprob 差 | 状态 |', '|---|---:|---:|---:|---|']
    for (group,arm,rep), cand in sorted(vresults.items()):
        base = vresults.get((group, 'stock', 'a1'))
        if base:
            if arm == 'stock': continue
            pairs = list(zip(base['gates'],cand['gates']))
            equal_prompt = all(a['prompt_sha256'] == b['prompt_sha256'] for a,b in pairs)
            same = sum(x == y for a,b in pairs for x,y in zip(a['ids'], b['ids']))
            total = sum(max(len(a['ids']),len(b['ids'])) for a,b in pairs)
            diffs = []
            for a,b in pairs:
                token = str(a['ids'][0])
                if token in a['top5'][0] and token in b['top5'][0]:
                    diffs.append(abs(a['top5'][0][token] - b['top5'][0][token]))
            finite = all(math.isfinite(v) for a,b in pairs for gate in (a,b)
                         for lp in gate['top5'] for v in lp.values())
            diffs = [v for v in diffs if math.isfinite(v)]
            divergences = [next((i+1 for i,(x,y) in enumerate(zip(a['ids'],b['ids'])) if x!=y), None) for a,b in pairs]
            status = ('logprob 非有限，结果无效' if not finite else
                      ('token 全同' if same==total==192 else '有差异；首次分叉 ' + ','.join(str(v) if v else '无' for v in divergences)))
            ilabel = {'stock_sp':'rccl（NCCL）','row':'plan 对应位置（非预测器）','all':'dx','plan':'plan（预测器）'}[arm]
            lines.append(f'| {ilabel}/{rep} ({group}) | {len(pairs) if equal_prompt else 0}/4 | {same}/{total} | {max(diffs):.4f}'
                         f' ({len(diffs)}/4)' if diffs else f'| {ilabel}/{rep} ({group}) | {len(pairs) if equal_prompt else 0}/4 | {same}/{total} | —')
            lines[-1] += f' | {status} |'
    lines += ['', '首 token logprob 差只在原版选中 token 同时位于两臂 top5 时可计算；token 检查不替代逐层误差分析。未扩展 RoPE、强行超过原模型 8192 上下文的诊断结果产生 NaN，已从主表移除；原始文件保留在 `/root/data/stowaway-nvidia/results/tp8_20261111/vllm_bwmatch_*_a1.json`。', '',
              '## 4. 预测器', '',
              'A800 TP8 第一级 11 模型原表算子点已采集；第二级 5 个 E2E 点留一评估及训练 plan/alt 仍未完成；BW1000 的 F50 常数不能直接沿用。TP8 后向 AG+dgrad、独立 BRS 原型均已做 8 rank × 8 次算子检查，Qwen 64 层训练已运行后向 DX，独立 BRS 在 MBS1/128MB 门槛下命中 0 次。', '']
    original_cells = Path('/root/Stowaway-NVIDIA/planner/original_a800/data/cells.json')
    if original_cells.exists():
        oc=json.loads(original_cells.read_text())
        lines += [f"原始代价模型 A800 标定：**{len(oc)}/99 点，{len({c['model'] for c in oc})}/11 模型**；每点含原版/DX 的单次、16 连发和 GEMM/通信分量。", '']
    original_validation = Path('/root/Stowaway-NVIDIA/planner/original_a800/data/planner_validation.md')
    if original_validation.exists():
        lines += ['原代价模型留一模型结果（含覆盖点数和错误点）：[A800 留一模型验证](../planner/original_a800/data/planner_validation.md)。16 连发选对 **86/99（86.9%）**，几何平均 **1.057×**；候选仅原版与 DX，不包括原表拆批。', '']
    lines += ['实验队列与退出码：`/root/data/stowaway-nvidia/results/raw/bwmatch_tp8_queue_status.tsv`；每臂原始日志位于同目录，文件名 `bwmatch_<载具>_mbs<MBS>_<臂>_<轮次>.log`。队列截止时正在运行的实验可能被终止，不纳入表内有效数据。', '']
    OUT.write_text('\n'.join(lines))
    print(OUT)

if __name__ == '__main__':
    main()
