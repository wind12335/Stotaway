import json,statistics,collections
from pathlib import Path
root=Path('/root/Stotaway/nvidia')
res=Path('/root/data/stowaway-nvidia/results')
old=json.loads((root/'planner/original_a800/data/cells.json').read_text())
def key(r):return (r['model'],r['op'],r['M'],r['N'],r['K'])
dx_retest={}
for r in old:
 p=res/'tp8_dx_bf16_retest_20261114'/f"{r['model']}_{r['op']}_{r['M']}_{r['N']}_{r['K']}.json"
 if not p.exists(): continue
 try:
  value=json.loads(p.read_text());op='rs' if r['op']=='rs' else 'ag'
  if value['world']==8 and value['batch']==16 and value['checks']>=4 and value['repeats']>=8:
   dx_retest[key(r)]=value['result'][op]['speedup']
 except (ValueError,KeyError,TypeError): pass
def load(pattern):
 d={}
 for p in sorted(res.glob(pattern)):
  for line in p.read_text().splitlines():
   if line.strip():
    try: r=json.loads(line)
    except json.JSONDecodeError: continue  # writer may be appending its current line
    d[tuple(r['identity'])]=r
 return d
asyncs=load('tp8_async_matrix_m*20261113.jsonl')
td=load('tp8_td_*20261113.jsonl')
sync={}
for op in ('ag','rs'):
 a=res/f'tp8_syncopate_{op}_stock_20261113.json';b=res/f'tp8_syncopate_{op}_syncopate_20261113.json'
 if a.exists() and b.exists():
  x=json.loads(a.read_text());y=json.loads(b.read_text());z=x['target']
  sync[(z['op'],z['M'],z['N'],z['K'])]=x['median_ms']/y['median_ms']
matrix=res/'tp8_sync_matrix_20261113'
for f in matrix.glob('*_stock.json') if matrix.exists() else []:
 g=f.with_name(f.name.replace('_stock.json','_syncopate.json'))
 if g.exists():
  try:
   x=json.loads(f.read_text());y=json.loads(g.read_text());z=x['target']
   sync[(z['op'],z['M'],z['N'],z['K'])]=x['median_ms']/y['median_ms']
  except (json.JSONDecodeError,KeyError,ZeroDivisionError): pass
flash={}
fpath=res/'tp8_flash_b16_20261113.json'
if fpath.exists():
 f=json.loads(fpath.read_text()); flash[('rs',f['M'],f['N'],f['K'])]=f['measurements']['stock']['median_ms']/f['measurements']['overlap']['median_ms']
flash_dir=res/'tp8_flash_matrix_20261113'
for f in flash_dir.glob('*.json') if flash_dir.exists() else []:
 if f.name.endswith('_check.json'): continue
 try:
  x=json.loads(f.read_text());flash[('rs',x['M'],x['N'],x['K'])]=x['measurements']['stock']['median_ms']/x['measurements']['overlap']['median_ms']
 except (json.JSONDecodeError,KeyError,ZeroDivisionError): pass
flash_dispatch={}
for f in (res/'tp8_flash_missing_dispatch_20261114').glob('*.json'):
 if f.name.endswith('_plan.json'): continue
 try:
  x=json.loads(f.read_text())['results'][0]; m,n,k=x['shape']
  flash_dispatch[('rs',m,n,k)]=x['speedup']
 except (json.JSONDecodeError,KeyError,IndexError,ValueError): pass
rows=['# 单机 8 卡 A800 多后端算子结果（2026-11-13）','','基准为对应测试内的串行 AG→GEMM 或 GEMM→RS，stock = 1.000×；speedup = stock 中位耗时 / 后端中位耗时。M 为全局行数，AG 每卡输入行数为 M/8。','',f'当前 Async-TP：{len(asyncs)}/99；Triton-distributed：{len(td)}/99；FlashOverlap：{len(set(flash)|set(flash_dispatch))}/33 RS 形状（原生 {len(flash)}，派发补测 {len(flash_dispatch)}）；Syncopate：{len(sync)}/99。DX 按形状取旧值与 2026-11-14 BF16 重测值中的较高值，来源保留。','', 'Async-TP 与 Triton-distributed 使用 BF16，每个点在同一进程比较 stock 和目标后端，预热 4 次，8 组、每组连续 16 次，取各 rank 最大耗时后取中位数。旧 DX 使用原记录中的 stock_b2b / dx_b2b，旧数据与新数据各自的 stock 独立计时。FlashOverlap 标注“派发”的 10 点为 FP16 同进程测量，包含 tile 恢复和跨 rank 转换；其余 FP16 点为原生内核计时，不包含独立布局恢复；两者计时口径不同。','', '| 模型 | 算子 | 全局 M | 每卡 N | K | stock | Async-TP | Triton-dist | FlashOverlap | Syncopate | DX（择高） |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
fmt=lambda x: f'{x:.3f}×'
for r in sorted(old,key=lambda x:(x['model'],x['op'],x['M'])):
 k=key(r); a=fmt(asyncs[k]['speedup']) if k in asyncs else '未测'; t=fmt(td[k]['speedup']) if k in td else '未测'; old_dx=r['stock_b2b']/r['dx_b2b'] if r.get('stock_b2b') and r.get('dx_b2b') else None; new_dx=dx_retest.get(k); dx=(fmt(new_dx)+'（重测）' if new_dx is not None and (old_dx is None or new_dx>old_dx) else fmt(old_dx) if old_dx is not None else '无数据'); sk=('ag' if r['op'].startswith('ag') else 'rs',r['M'],r['N'],r['K']); sy=fmt(sync[sk])+'（FP16）' if sk in sync else '未测'; fl=fmt(flash[sk])+'（FP16）' if sk in flash else fmt(flash_dispatch[sk])+'（FP16，派发）' if sk in flash_dispatch else '未测'
 rows.append(f"| {r['model']} | {r['op']} | {r['M']} | {r['N']} | {r['K']} | 1.000× | {a} | {t} | {fl} | {sy} | {dx} |")
paired_path=res/'tp8_async_dx_rs_paired_20261113.json'
if paired_path.exists():
 pair=json.loads(paired_path.read_text())
 rows += ['', '## Async-TP 与 DX 同进程对照', '',
          f"Llama3-8B RS，M={pair['M']}、N={pair['N']}、K={pair['K']}，BF16、TP8：Async-TP {pair['measurements']['async_tp']['median_ms']:.3f} ms，DX {pair['measurements']['dx']['median_ms']:.3f} ms，DX 相对 Async-TP {pair['dx_over_async_speedup']:.3f}×。", '',
          '原始数据：`/root/data/stowaway-nvidia/results/tp8_async_dx_rs_paired_20261113.json`。']
prediction_path=res/'tp8_async_predictor_lomo_20261113.json'
if prediction_path.exists():
 p=json.loads(prediction_path.read_text())['summary']
 rows += ['', '## Async-TP 算子预测器：按模型留出', '', '| 算子 | 正确 / 总数 | 正确率 | 选择后的几何平均 speedup |', '|---|---:|---:|---:|']
 for name in ('ag','rs','all'):
  x=p[name]; rows.append(f"| {name} | {x['correct']}/{x['count']} | {x['accuracy']*100:.2f}% | {x['speedup_gmean']:.3f}× |")
 rows += ['', '本评估为 stock 与 Async-TP 二选一；每次留出一个模型，训练其他模型，输入是全局 M、每卡 N 和 K。详见 `/root/data/stowaway-nvidia/results/tp8_async_predictor_lomo_20261113.json`。']
three_path=res/'tp8_auto_threeway_lomo_20261113.json'
if three_path.exists():
 p=json.loads(three_path.read_text())['summary']
 rows += ['', '## stock / Async-TP / DX 三选一：按模型留出', '', '| 算子 | 正确 / 总数 | 正确率 | 选择后的几何平均 speedup | 低于 stock |', '|---|---:|---:|---:|---:|']
 for name in ('ag','rs','all'):
  x=p[name]; rows.append(f"| {name} | {x['correct']}/{x['count']} | {x['accuracy']*100:.2f}% | {x['selected_gmean_speedup']:.3f}× | {x['selected_below_stock']} |")
 rows += ['', '开发评估：每种方法使用各自批次的 stock 归一化；DX 与 Async-TP 的测量来自不同批次，三候选最优标签属于近似对照。预测器自动方案的模型端到端正在另行验证。原始数据：`/root/data/stowaway-nvidia/results/tp8_auto_threeway_lomo_20261113.json`。']
td_three_path=res/'tp8_td_dx_threeway_lomo_20261113.json'
if td_three_path.exists():
 p=json.loads(td_three_path.read_text())['summary']
 rows += ['', '## stock / Triton-distributed / DX 三选一：按模型留出', '', '| 算子 | 正确 / 总数 | 正确率 | 选择后的几何平均 speedup | 低于 stock |', '|---|---:|---:|---:|---:|']
 for name in ('ag','rs','all'):
  x=p[name]; rows.append(f"| {name} | {x['correct']}/{x['count']} | {x['accuracy']*100:.2f}% | {x['selected_gmean_speedup']:.3f}× | {x['selected_below_stock']} |")
 rows += ['', '开发评估：两个后端各自批次单独使用 stock 归一化，部分 Triton-distributed 点运行参数不同，最优标签是近似对照；Triton-distributed + DX 的模型端到端结果见端到端表。原始数据：`/root/data/stowaway-nvidia/results/tp8_td_dx_threeway_lomo_20261113.json`。']
for method in ('syncopate','flashoverlap'):
 path=res/f'tp8_{method}_dx_fp16_lomo_20261114.json'
 if not path.exists(): continue
 data=json.loads(path.read_text())['summary']
 rows += ['',f'## stock / {method} / DX 三选一：FP16 按模型留出','', '| 算子 | 正确 / 总数 | 正确率 | 选择后的几何平均 speedup | 低于 stock |','|---|---:|---:|---:|---:|']
 for name in ('ag','rs','all'):
  if name not in data:continue
  x=data[name];rows.append(f"| {name} | {x['correct']}/{x['count']} | {x['accuracy']*100:.2f}% | {x['selected_gmean_speedup']:.3f}× | {x['selected_below_stock']} |")
 rows += ['', '开发评估，候选分别在不同批次测量，DX 使用 Syncopate 对应 stock 归一化；FlashOverlap 原生及含布局转换的派发计时口径不同。模型 BF16 自动方案使用此前跨精度计划。',f'原始记录：`{path}`。']
carrier_results=[]
for method in ('triton_dist','syncopate','flashoverlap','syncopate_bf16'):
 p=res/f'tp8_external_dispatch_{method}_20261113.json'
 if p.exists():
  payload=json.loads(p.read_text())
  for item in payload['results']:
   carrier_results.append((method,payload['dtype'],item))
if carrier_results:
 rows += ['', '## 指定外部后端与 DX 的统一派发器实测', '', '| 外部后端 | dtype | 位置 | 算子 | M | N | K | 实际路径 | speedup |', '|---|---|---|---|---:|---:|---:|---|---:|']
 for method,dtype,item in carrier_results:
  m,n,k=item['shape']
  rows.append(f"| {method} | {dtype} | {item['site']} | {item['op']} | {m} | {n} | {k} | {item['choice']} | {item['speedup']:.3f}× |")
 rows += ['', '同进程按路径测量，8 组、每组 16 连发；FlashOverlap 结果包含输出重排和跨 rank 布局转换。详细记录 `tp8_external_dispatch_<后端>_20261113.json`。']
rows += ['', '原始记录：', '', '- Async-TP: `/root/data/stowaway-nvidia/results/tp8_async_matrix_m8192_20261113.jsonl` 和 `tp8_async_matrix_m16384_32768_20261113.jsonl`。', '- FlashOverlap：`/root/data/stowaway-nvidia/results/tp8_flash_matrix_20261113/` 和 `tp8_flash_b16_20261113.json`，FP16、独立进程、8 卡配置搜索；输出布局校验见 `tp8_flash_correctness_20261113.json`，speedup 不包含独立布局还原开销。', '- Syncopate：`/root/data/stowaway-nvidia/results/tp8_sync_matrix_20261113/` 和 `tp8_syncopate_{ag,rs}_{stock,syncopate}_20261113.json`，使用 FP16 并在独立进程测量，标注 FP16。', '- Triton-distributed: `/root/data/stowaway-nvidia/results/tp8_td_*20261113.jsonl`。前 9 个点采用 CUDA_DEVICE_MAX_CONNECTIONS=1，后续采用 8；每个点的 stock 与目标后端使用同一设置，原始记录逐点保存。', '- 既有 DX：`planner/original_a800/data/cells.json`，其每行记录组件文件和重复次数；此前采集结果。33 个 RS 形状中 32 个已有 DX speedup 超过 1×，本次 Async-TP 的同 33 个形状均低于 1×；两批各自的 stock 独立计时。', '- 具体数据类型及采样信息见各原始记录；未测项目不代表没有加速。','']
rows += ['FlashOverlap 10 个派发补测的原始记录：`/root/data/stowaway-nvidia/results/tp8_flash_missing_dispatch_20261114/`。405B 最大形状使用从已有形状借用的候选及单分组，非完整性能调优。', '']
if dx_retest:
 rows += [f'DX BF16 重测目前 {len(dx_retest)}/99；逐形状比较重测与旧值，较高值标注“重测”，旧数据不删除。新旧批次计时可能有波动；完整结果记录于 `/root/data/stowaway-nvidia/results/tp8_dx_bf16_retest_20261114/`。', '']
(root/'results/单机8卡多后端算子结果_20261113.md').write_text('\n'.join(rows))
for op in ['ag_gu','ag_up','rs']:
 vals=[r['speedup'] for r in asyncs.values() if r['op']==op]
 print(op,'n=',len(vals),'median=',round(statistics.median(vals),3),'range=',round(min(vals),3),round(max(vals),3),'above1=',sum(x>1 for x in vals))
