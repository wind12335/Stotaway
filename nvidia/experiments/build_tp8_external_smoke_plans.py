"""Build explicit test plans; predictions are evaluated separately."""
import json
from pathlib import Path

root = Path(__file__).parent
for label, backend, dtype, entries in (
    ('triton_dist', 'triton_dist', 'bfloat16', [('qkv', 'ag', 8192, 1792, 4096, 'triton_dist'),
                                 ('proj', 'rs', 8192, 4096, 1792, 'dx'),
                                 ('fc2', 'rs', 8192, 4096, 1792, 'triton_dist')]),
    ('syncopate', 'syncopate', 'float16', [('qkv', 'ag', 8192, 1792, 4096, 'syncopate'),
                              ('proj', 'rs', 8192, 4096, 1792, 'dx'),
                              ('fc2', 'rs', 8192, 4096, 1792, 'syncopate')]),
    ('syncopate_bf16', 'syncopate', 'bfloat16', [('qkv', 'ag', 8192, 1792, 4096, 'syncopate'),
                                                   ('proj', 'rs', 8192, 4096, 1792, 'dx'),
                                                   ('fc2', 'rs', 8192, 4096, 1792, 'syncopate')]),
    ('flashoverlap', 'flashoverlap', 'float16', [('proj', 'rs', 8192, 4096, 1792, 'dx'),
                                 ('fc2', 'rs', 8192, 4096, 1792, 'flashoverlap')]),
    ('flashoverlap_bf16', 'flashoverlap', 'bfloat16', [('proj', 'rs', 8192, 4096, 1792, 'dx'),
                                 ('fc2', 'rs', 8192, 4096, 1792, 'flashoverlap')]),
):
    out = {'schema': 'stowaway-tp8-auto-plan-v1', 'workload': 'operator_smoke', 'tp': 8,
           'dtype': dtype, 'external_backend': backend, 'default': 'stock',
           'entries': [dict(site=site, op=op, M=m, N=n, K=k, choice=choice)
                       for site, op, m, n, k, choice in entries],
           'provenance': {'classification': 'manual dispatcher correctness test; not a predictor plan'}}
    (root / f'tp8_external_smoke_{label}_20261113.json').write_text(json.dumps(out, indent=2) + '\n')
