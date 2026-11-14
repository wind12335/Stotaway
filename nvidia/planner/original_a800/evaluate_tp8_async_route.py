"""Leave-one-model-out 8-GPU Async-TP operator route evaluation."""
import hashlib, json, math, statistics
from pathlib import Path
from .orthogonal_predictor import Route, Sample
from .route_local_residual import RouteLocalResidual
from .route_policy import GroupedSample

root=Path('/root/data/stowaway-nvidia/results')
paths=sorted(root.glob('tp8_async_matrix_m*20261113.jsonl'))
rows=[json.loads(line) for p in paths for line in p.read_text().splitlines() if line]
results=[]
for op in ('ag','rs'):
    group=[r for r in rows if ('ag' if r['op'].startswith('ag') else 'rs') == op]
    route=Route('async_tp','pytorch_gemm','pytorch_symm_mem','symmetric_memory','stream',op,
                'torch.ops.symm_mem', 'A800-SXM4-80GB','NV8',8,'bfloat16',
                (('cuda','12.6'),('torch','2.7.0+cu126'),('triton','3.4.0')),
                'b2b16',4,'pytorch_direct')
    samples=[GroupedSample(r['model'],Sample(route,(r['M'],r['N'],r['K']),
                          r['measurements']['stock']['median_ms'],r['measurements']['async_tp']['median_ms'])) for r in group]
    for held in sorted({s.family for s in samples}):
        train=[s for s in samples if s.family!=held]
        pred=RouteLocalResidual(route).fit(train)
        for s in samples:
            if s.family!=held: continue
            p=pred.predict(route,s.sample.shape)
            actual=s.sample.stock_ms/s.sample.candidate_ms
            results.append({'model':held,'op':op,'shape':s.sample.shape,'actual_speedup':actual,'prediction':p})
summary={}
for op in ('ag','rs','all'):
    selected=[r for r in results if op=='all' or r['op']==op]
    correct=sum((r['prediction']['predicted_log_speedup']>0)==(r['actual_speedup']>1) for r in selected)
    actual=[r['actual_speedup'] if r['prediction']['predicted_log_speedup']>0 else 1 for r in selected]
    summary[op]={'count':len(selected),'correct':correct,'accuracy':correct/len(selected),
                 'speedup_gmean':math.exp(statistics.mean(map(math.log,actual))),'below_one':sum(x<1 for x in actual)}
out={'classification':'TP8 operator leave-one-model-out development evaluation',
     'source':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
     'note':'Stock vs Async-TP only; no DX replacement, unseen process, or E2E evaluation.',
     'summary':summary,'rows':results}
dest=root/'tp8_async_predictor_lomo_20261113.json'
with dest.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(summary,indent=2))
