"""Incrementally document 8-GPU Megatron and vLLM results from saved logs."""
import json, re, statistics
from pathlib import Path
root=Path('/root/Stotaway/nvidia')
res=Path('/root/data/stowaway-nvidia/results')
raw=res/'raw'
rows=['# 单机 8 卡端到端与预测器结果（2026-11-13）','',
      'Qwen3.6-27B 用于 Megatron 训练，Llama3-70B 用于 vLLM 推理。每种候选按同批 stock 耗时计算 speedup；结果随任务结束自动更新。', '',
      '## Megatron：Qwen3.6-27B 文本配置', '',
      'TP8、BF16、64 层、seq4096、MBS2、GBS2、全重算、随机初始化和 mock 数据。每次 12 步，取第 4～12 步的中位整步耗时。', '',
      '| 方案 | 中位整步 ms | 对 stock | 路径命中 | 状态 |', '|---|---:|---:|---|---|']
arms={}
for arm in ('stock','async_tp','async_dx','auto_plan','td_baseline','td_auto_plan','sync_baseline','sync_auto','flash_baseline','flash_auto'):
 retry=raw/f'tp8_qwen36_{arm}_mbs2_fla_retry_{"20261114" if arm.startswith(("sync_", "flash_")) else "20261113"}.log'
 p=retry
 s=p.read_text(errors='replace') if p.exists() else ''
 times=[(int(a),float(b)) for a,b in re.findall(r'iteration\s+(\d+)\s*/\s*12.*?elapsed time per iteration \(ms\):\s*([\d.]+)',s)]
 valid=[b for a,b in times if 4<=a<=12]
 hits=re.findall(r'STOWAWAY_DIAGNOSTIC_COUNTS rank=0.*?(?=STOWAWAY_|\n|$)',s)
 arms[arm]={'median':statistics.median(valid) if len(valid)==9 else None,'hits':hits[-1] if hits else None,'state':'完成' if len(valid)==9 else ('运行中/未完成' if p.exists() else '未开始')}
base=arms['stock']['median']
for arm,label in [('stock','stock'),('async_tp','Async-TP'),('async_dx','Async-TP + DX（proj/fc2）'),('auto_plan','预测器自动方案'),('td_baseline','Triton-dist 基线'),('td_auto_plan','Triton-dist + DX 自动方案'),('sync_baseline','Syncopate 基线'),('sync_auto','Syncopate + DX 自动方案'),('flash_baseline','FlashOverlap 基线（仅 RS）'),('flash_auto','FlashOverlap + DX 自动方案')]:
 x=arms[arm];v=x['median']; speed=f'{base/v:.3f}×' if base and v else '—';hit=x['hits'].split('rank=0 ',1)[-1] if x['hits'] else '—'
 rows.append(f"| {label} | {v:.1f} | {speed} | `{hit}` | {x['state']} |" if v else f"| {label} | — | {speed} | `{hit}` | {x['state']} |")
if arms['async_tp']['median'] and arms['async_dx']['median']:
 rows += ['', f"Async-TP + DX 相对 Async-TP：{arms['async_tp']['median']/arms['async_dx']['median']:.3f}×。"]
if arms['async_tp']['median'] and arms['auto_plan']['median']:
 rows += ['', f"预测器自动方案相对 Async-TP：{arms['async_tp']['median']/arms['auto_plan']['median']:.3f}×。"]
 auto_log=raw/'tp8_qwen36_auto_plan_mbs2_fla_retry_20261113.log'
 matches=re.findall(r'STOWAWAY_AUTO_HIT rank=0 key=([^\n]+)',auto_log.read_text(errors='replace'))
 rows += ['', 'rank 0 自动计划实际命中：'] + [f'- `{hit}`' for hit in matches]
if arms['td_auto_plan']['median']:
 td_log=raw/'tp8_qwen36_td_auto_plan_mbs2_fla_retry_20261113.log'
 matches=list(dict.fromkeys(re.findall(r'STOWAWAY_MULTI_HIT rank=0 key=([^\n]+)',td_log.read_text(errors='replace'))))
 rows += ['', 'Triton-dist + DX 的 rank 0 自动计划实际命中：'] + [f'- `{hit}`' for hit in matches]
if arms['td_baseline']['median'] and arms['td_auto_plan']['median']:
 rows += ['', f"Triton-dist + DX 自动方案相对 Triton-dist 基线：{arms['td_baseline']['median']/arms['td_auto_plan']['median']:.3f}×。"]
if arms['sync_baseline']['median'] and arms['sync_auto']['median']:
 rows += ['', f"Syncopate + DX 自动方案相对 Syncopate 基线：{arms['sync_baseline']['median']/arms['sync_auto']['median']:.3f}×。"]
if arms['flash_baseline']['median'] and arms['flash_auto']['median']:
 rows += ['', f"FlashOverlap + DX 自动方案相对 FlashOverlap 基线：{arms['flash_baseline']['median']/arms['flash_auto']['median']:.3f}×。"]
rows += ['', '训练原始日志：`/root/data/stowaway-nvidia/results/raw/tp8_qwen36_<方案>_mbs2_fla_retry_20261113.log`。', '',
         '## vLLM：Llama3-70B', '',
         '真实模型权重、TP8、BF16、完整 80 层，M=16384、线性 RoPE 扩展 8 倍，记录 TTFT、两倍长度 TTFT、16 请求总耗时；speedup 为同批 stock 耗时除以候选耗时。', '',
         '| 方案 | TTFT ms | TTFT speedup | 2M TTFT ms | 2M speedup | 16 请求总耗时 ms | 吞吐 speedup | 路径命中 | 状态 |', '|---|---:|---:|---:|---:|---:|---:|---|---|']
values={}
for arm in ('stock','async_tp','async_dx','auto_plan','td_baseline','td_auto_plan','sync_baseline','sync_auto','flash_baseline','flash_auto'):
 p=res/'tp8_20261111'/f'vllm_bwmatch_{arm}_retry_{"20261114" if arm.startswith(("sync_", "flash_")) else "20261113"}.json'
 values[arm]=json.loads(p.read_text()) if p.exists() else None
base=values['stock']
for arm,label in [('stock','stock'),('async_tp','Async-TP'),('async_dx','Async-TP + DX（proj/fc2）'),('auto_plan','预测器自动方案'),('td_baseline','Triton-dist 基线'),('td_auto_plan','Triton-dist + DX 自动方案'),('sync_baseline','Syncopate 基线'),('sync_auto','Syncopate + DX 自动方案'),('flash_baseline','FlashOverlap 基线（仅 RS）'),('flash_auto','FlashOverlap + DX 自动方案')]:
 v=values[arm]
 if not v:
  rows.append(f'| {label} | — | — | — | — | — | — | — | 未完成 |');continue
 times=v['summary_ms']; speeds=[f"{base['summary_ms'][k]/times[k]:.3f}×" if base else '—' for k in ('ttft','ttft2','thru')]
 hits=json.dumps(v.get('backend_hits') or v.get('multi_hits'),ensure_ascii=False) if (v.get('backend_hits') or v.get('multi_hits')) else '—'
 rows.append(f"| {label} | {times['ttft']:.1f} | {speeds[0]} | {times['ttft2']:.1f} | {speeds[1]} | {times['thru']:.1f} | {speeds[2]} | `{hits}` | 完成 |")
if values['async_tp'] and values['async_dx']:
 a=values['async_tp']['summary_ms']; d=values['async_dx']['summary_ms']
 rows += ['', f"Async-TP + DX 相对 Async-TP：TTFT {a['ttft']/d['ttft']:.3f}×，2M TTFT {a['ttft2']/d['ttft2']:.3f}×，16 请求吞吐 {a['thru']/d['thru']:.3f}×。"]
if values['async_tp'] and values['auto_plan']:
 a=values['async_tp']['summary_ms']; d=values['auto_plan']['summary_ms']
 rows += ['', f"预测器自动方案相对 Async-TP：TTFT {a['ttft']/d['ttft']:.3f}×，2M TTFT {a['ttft2']/d['ttft2']:.3f}×，16 请求吞吐 {a['thru']/d['thru']:.3f}×。"]
 rows += ['', 'rank 0 自动计划实际命中：'] + [f'- `{key}`：{count}' for key,count in sorted((values['auto_plan'].get('auto_hits') or {}).items())]
if values['td_baseline'] and values['td_auto_plan']:
 a=values['td_baseline']['summary_ms']; d=values['td_auto_plan']['summary_ms']
 rows += ['', f"Triton-dist + DX 自动方案相对 Triton-dist 基线：TTFT {a['ttft']/d['ttft']:.3f}×，2M TTFT {a['ttft2']/d['ttft2']:.3f}×，16 请求吞吐 {a['thru']/d['thru']:.3f}×。"]
if values['sync_baseline'] and values['sync_auto']:
 a=values['sync_baseline']['summary_ms']; d=values['sync_auto']['summary_ms']
 rows += ['', f"Syncopate + DX 自动方案相对 Syncopate 基线：TTFT {a['ttft']/d['ttft']:.3f}×，2M TTFT {a['ttft2']/d['ttft2']:.3f}×，16 请求吞吐 {a['thru']/d['thru']:.3f}×。"]
if values['flash_baseline'] and values['flash_auto']:
 a=values['flash_baseline']['summary_ms']; d=values['flash_auto']['summary_ms']
 rows += ['', f"FlashOverlap + DX 自动方案相对 FlashOverlap 基线：TTFT {a['ttft']/d['ttft']:.3f}×，2M TTFT {a['ttft2']/d['ttft2']:.3f}×，16 请求吞吐 {a['thru']/d['thru']:.3f}×。"]
rows += ['', 'Syncopate 与 FlashOverlap 自动方案使用 FP16 外部后端数据及 BF16 DX 数据预测，属于跨数据类型的试验计划；模型实际运行均为 BF16。FlashOverlap 基线只替换 RS，并计入输出布局恢复。']
rows += ['', '推理原始记录：`/root/data/stowaway-nvidia/results/tp8_20261111/vllm_bwmatch_<方案>_retry_20261113.json`。', '',
         '三种推理方案的四条自然文本输出检查中，记录的 top-5 logprob 均为有限值。', '',
         'Async-TP + DX 的 proj/fc2 替换位置为手工固定；“预测器自动方案”使用单独的 JSON 计划。其他后端模型接线已增加，端到端验证状态见上表及进度文档。', '',
         '## 8 卡预测器与替换机制', '',
         'Async-TP 算子级留一模型评估结果及完整算子 speedup 见 [多后端算子表](单机8卡多后端算子结果_20261113.md)。',
         'TP8 三候选执行检查：每 rank 对 stock、Async-TP、DX 分别命中两次并通过对照；vLLM 合成模型中 Async-TP 与 Async-TP+DX 两种路径均通过 8 rank 检查。',
         '替换路径检查数据：`/root/data/stowaway-nvidia/results/tp8_replacement_coverage_20261113.json`；合成模型日志：`/root/data/stowaway-nvidia/results/tp8_vllm_synthetic_*_20261113.log`。', '',
         '既有训练与真实权重推理数据见 [原表对照](单机8卡原表对照_20261111.md) 和 [此前端到端结果](端到端与预测器结果表.md)。','']
(root/'results/单机8卡端到端与预测器结果_20261113.md').write_text('\n'.join(rows))
