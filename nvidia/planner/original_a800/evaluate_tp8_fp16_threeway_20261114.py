"""LOMO FP16 evaluation with existing external and newly measured DX timings.

Stock and DX use separate processes and batches. Flash has two timing contracts:
23 raw overlap measurements, 10 complete dispatcher measurements.
"""
import hashlib,json,math,statistics
from pathlib import Path
from .export_tp8_fp16_external_plan import make_route
from .export_tp8_auto_plan import ROOT
from .orthogonal_predictor import Sample
from .route_local_residual import RouteLocalResidual
from .route_policy import GroupedSample

inventory=Path(__file__).parent/'data/cells.json'
rows=json.loads(inventory.read_text()); matrix=ROOT/'tp8_sync_matrix_20261113'; dxdir=ROOT/'tp8_dx_fp16_matrix_20261114'
flash_raw=ROOT/'tp8_flash_matrix_20261113';flash_full=ROOT/'tp8_flash_missing_dispatch_20261114'
flash_one=ROOT/'tp8_flash_b16_20261113.json'

def load_json(p): return json.loads(p.read_text())

def evaluate(backend):
 samples=[]; paths=set()
 for r in rows:
  op='ag' if r['op'].startswith('ag') else 'rs'
  if backend=='flashoverlap' and op!='rs':continue
  stem=f"{r['model']}_{r['op']}_{r['M']}"
  stock=matrix/f'{stem}_stock.json'; dx=dxdir/f'{stem}_dx.json'
  if not stock.exists() or not dx.exists():continue
  s=load_json(stock)['median_ms']; d=load_json(dx)['median_ms']; paths.update([stock,dx])
  if backend=='syncopate':
   p=matrix/f'{stem}_syncopate.json'
   if not p.exists():continue
   e=s/load_json(p)['median_ms'];contract='generated_kernel'
  else:
   p=flash_raw/f"{r['model']}_{r['M']}_{r['N']}_{r['K']}.json"
   if not p.exists() and r['model']=='llama3-8b' and r['M']==8192:p=flash_one
   if p.exists():
    v=load_json(p)['measurements'];e=v['stock']['median_ms']/v['overlap']['median_ms'];contract='raw_overlap_no_layout'
   else:
    p=flash_full/f"{r['model']}_{r['M']}_{r['N']}_{r['K']}.json"
    if not p.exists():continue
    e=load_json(p)['results'][0]['speedup'];contract='dispatcher_with_layout'
  paths.add(p);samples.append({'model':r['model'],'op':r['op'],'kind':op,'shape':(r['M'],r['N'],r['K']), 'external':e,'dx':s/d,'contract':contract})
 result=[]
 for op in ('ag','rs'):
  data=[x for x in samples if x['kind']==op]
  if not data:continue
  for held in sorted({x['model'] for x in data}):
   train=[x for x in data if x['model']!=held]
   if len({x['model'] for x in train})<4:continue
   routes={method:make_route(method,op) for method in (backend,'dx')}
   models={method:RouteLocalResidual(routes[method]).fit([GroupedSample(x['model'],Sample(routes[method],x['shape'],1,1/x['external' if method==backend else 'dx'])) for x in train]) for method in (backend,'dx')}
   for x in [x for x in data if x['model']==held]:
    predicted={'stock':0.0,**{m:models[m].predict(routes[m],x['shape'])['predicted_log_speedup'] for m in models}}
    actual={'stock':1.0,backend:x['external'],'dx':x['dx']};choice=max(predicted,key=predicted.get)
    result.append({'identity':[x['model'],x['op'],*x['shape']],'choice':choice,'oracle_choice':max(actual,key=actual.get),'predicted_log_speedup':predicted,'actual_normalized_speedup':actual,'selected_speedup':actual[choice],'external_timing_contract':x['contract']})
 summary={}
 for op in ('ag','rs','all'):
  subset=[r for r in result if op=='all' or (('ag' if r['identity'][1].startswith('ag') else 'rs')==op)]
  if subset:
   summary[op]={'count':len(subset),'correct':sum(r['choice']==r['oracle_choice'] for r in subset),'accuracy':sum(r['choice']==r['oracle_choice'] for r in subset)/len(subset),'selected_gmean_speedup':math.exp(statistics.mean(math.log(r['selected_speedup']) for r in subset)),'selected_below_stock':sum(r['selected_speedup']<1 for r in subset)}
 out={'classification':'FP16 leave-one-model-out development evaluation','backend':backend,'limitation':'separate process/batch; DX normalized against separate stock, flash timing contracts differ for 23 raw and 10 dispatcher samples', 'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)],'summary':summary,'rows':result}
 path=ROOT/f'tp8_{backend}_dx_fp16_lomo_20261114.json'
 path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(backend,json.dumps(summary))

if __name__=='__main__':
 for name in ('syncopate','flashoverlap'):evaluate(name)
