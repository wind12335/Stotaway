"""Build BF16 model plans using the measured FP16 Flash model provisionally."""
import hashlib, json
from dataclasses import asdict
from pathlib import Path
from planner.original_a800.export_tp8_auto_plan import generate
from planner.original_a800.export_tp8_fp16_external_plan import make_route, read_flash
from planner.original_a800.orthogonal_predictor import Sample
from planner.original_a800.route_local_residual import RouteLocalResidual
from planner.original_a800.route_policy import GroupedSample

here=Path(__file__).parent
rows=read_flash()
route=make_route('flashoverlap','rs')
model=RouteLocalResidual(route).fit([GroupedSample(r['model'], Sample(route,(r['M'],r['N'],r['K']),r['stock'],r['candidate'])) for r in rows])
for name,workload in [('qwen36','qwen36_train'),('llama70','llama70_infer')]:
    auto=generate(workload,'triton_dist')
    auto['external_backend']='flashoverlap';auto['default']='stock'
    prov=auto['provenance']
    prov['method']='FP16 FlashOverlap route model transferred provisionally to BF16; BF16 DX route model'
    prov['limitation']='Flash measurements omit layout restoration and are FP16; model runs are BF16 and include layout restoration'
    files=sorted(Path('/root/data/stowaway-nvidia/results/tp8_flash_matrix_20261113').glob('*.json'))
    files=[p for p in files if not p.name.endswith('_check.json')]
    other=Path('/root/data/stowaway-nvidia/results/tp8_flash_b16_20261113.json')
    if other.exists():files.append(other)
    dx=here.parent/'planner/original_a800/data/cells.json'
    prov['sources']=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [*files,dx]]
    prov['routes']=[asdict(route)]+[r for r in prov['routes'] if r['method']=='dx']
    for e in auto['entries']:
        scores={'stock':0.0,'dx':e['predicted_log_speedup']['dx']}
        if e['op']=='rs': scores['flashoverlap']=model.predict(route,(e['M'],e['N'],e['K']))['predicted_log_speedup']
        e['predicted_log_speedup']=scores;e['choice']=max(scores,key=scores.get)
    for arm in ('baseline','auto'):
        out=json.loads(json.dumps(auto))
        if arm=='baseline':
            for e in out['entries']:e['choice']='flashoverlap' if e['op']=='rs' else 'stock'
            out['provenance']['method']='manual FlashOverlap on eligible RS, stock AG'
        path=here/f'tp8_{arm}_flash_{name}_20261114.json'
        if path.exists() and json.loads(path.read_text()) != out: raise ValueError(f'existing plan differs: {path}')
        if not path.exists():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        print(path,[(e['site'],e['M'],e['N'],e['K'],e['choice']) for e in out['entries']])
