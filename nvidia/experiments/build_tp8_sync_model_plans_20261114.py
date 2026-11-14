"""Build Syncopate baseline and provisional BF16 DX-replacement plans."""
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from planner.original_a800.export_tp8_auto_plan import generate
from planner.original_a800.export_tp8_fp16_external_plan import make_route, read_sync
from planner.original_a800.orthogonal_predictor import Sample
from planner.original_a800.route_local_residual import RouteLocalResidual
from planner.original_a800.route_policy import GroupedSample

root = Path(__file__).parent
rows = read_sync()
models = {}
for op in ('ag', 'rs'):
    route = make_route('syncopate', op)
    samples = [GroupedSample(r['model'], Sample(route, (r['M'], r['N'], r['K']), r['stock'], r['candidate']))
               for r in rows if r['op'] == op]
    models[op] = (route, RouteLocalResidual(route).fit(samples))
for model, workload in [('qwen36', 'qwen36_train'), ('llama70', 'llama70_infer')]:
    payload = generate(workload, 'triton_dist')
    payload['external_backend'] = 'syncopate'
    payload['default'] = 'stock'
    payload['provenance']['method'] = 'experimental FP16 Syncopate model transferred to BF16; BF16 DX route model'
    payload['provenance']['limitation'] = 'Syncopate model trained on FP16, DX on BF16; BF16 real-model runs required to validate'
    sync_dir = Path('/root/data/stowaway-nvidia/results/tp8_sync_matrix_20261113')
    sources = sorted(sync_dir.glob('*_stock.json')) + sorted(sync_dir.glob('*_syncopate.json'))
    dx_path = Path(__file__).resolve().parents[1] / 'planner/original_a800/data/cells.json'
    payload['provenance']['sources'] = [
        {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in [*sources, dx_path]
    ]
    payload['provenance']['routes'] = [asdict(models[op][0]) for op in ('ag', 'rs')] + [
        r for r in payload['provenance']['routes'] if r['method'] == 'dx'
    ]
    for e in payload['entries']:
        route, pred = models[e['op']]
        score = pred.predict(route, (e['M'], e['N'], e['K']))['predicted_log_speedup']
        scores = {'stock': 0.0, 'syncopate': score, 'dx': e['predicted_log_speedup']['dx']}
        e['predicted_log_speedup'] = scores
        e['choice'] = max(scores, key=scores.get)
    for arm in ('auto', 'baseline'):
        result = json.loads(json.dumps(payload))
        if arm == 'baseline':
            for e in result['entries']:
                e['choice'] = 'syncopate'
            result['provenance']['method'] = 'manual all eligible projections use Syncopate; stock fallback'
        path = root / f'tp8_{arm}_sync_{model}_20261114.json'
        if path.exists():
            if json.loads(path.read_text()) != result:
                raise ValueError(f'existing plan differs: {path}')
        else:
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(path, [(e['site'], e['M'], e['N'], e['K'], e['choice']) for e in result['entries']])
