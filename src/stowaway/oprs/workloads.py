"""Call-site shape manifests of the BW end-to-end workloads and the operator inventory (CPU only).

Sites are framework call sites (qkv / proj / fc1 / fc2); operator-bench points are ag_gu / ag_up /
rs. A site's GEMM has op ag (column-parallel: gather the SP input shards, then GEMM) or rs
(row-parallel: GEMM, then scatter the sum). M = global rows = sequence length x micro-batch for
training and the prefill chunk for inference; N, K are the per-rank GEMM widths.

The operator table (99 points, 11 published dense models x {ag_gu, ag_up, rs} x M 8192/16384/32768)
does NOT contain the qkv / proj shapes: those are off-table targets and the plan says so.
"""
from pathlib import Path
from typing import Optional

from .predictor import OPS

SITES = ('qkv', 'proj', 'fc1', 'fc2')
SITE_OP = {'qkv': 'ag', 'proj': 'rs', 'fc1': 'ag', 'fc2': 'rs'}
SITE_TABLE_OP = {'qkv': 'ag_up', 'proj': 'rs', 'fc1': 'ag_gu', 'fc2': 'rs'}

# Mirrors planner/shapes.py DENSE (published model configs, checked 2026-10): key -> (hidden, ffn)
MODEL_FAMILIES = {
    'qwen3-8b': (4096, 12288), 'llama3-8b': (4096, 14336), 'qwen3-32b': (5120, 10240),
    'llama4-mav-d': (5120, 16384), 'gemma3-27b': (5376, 21504), 'mistral3-d': (7168, 16384),
    'dsv3-d': (7168, 18432), 'llama3-70b': (8192, 28672), 'qwen25-72b': (8192, 29568),
    'mistral-l-2407': (12288, 28672), 'llama31-405b': (16384, 53248),
}
TABLE_MS = (8192, 16384, 32768)
# Handoff section 2: the first batch to re-measure is the 7-model M=32768 subset.
FIRST_BATCH = ('llama3-8b', 'qwen3-32b', 'llama4-mav-d', 'mistral3-d', 'llama3-70b', 'qwen25-72b',
               'mistral-l-2407')
# Old anomalies to re-check (originals are kept untouched; re-measurements go to a new directory).
RECHECK = ('llama4-mav-d/ag_gu/M32768', 'qwen3-8b/rs/M32768')

# Dense GPT shape vehicles of e2e/megatron (random init, mock data). They are NOT Qwen3.6 models.
CARRIERS = {
    'bw27b': dict(hidden=5120, ffn=17408, heads=40, kv_groups=8, layers=32),
    'bw70b': dict(hidden=8192, ffn=28672, heads=64, kv_groups=8, layers=8),
}


def operator_inventory(tp: int, models=tuple(MODEL_FAMILIES), ms=TABLE_MS) -> list:
    """Operator-table points: rs (N=H, K=F/TP), ag_up (K=H, N=F/TP), ag_gu (K=H, N=2F/TP)."""
    points = []
    for model in models:
        hidden, ffn = MODEL_FAMILIES[model]
        for m in ms:
            points.append({'kind': 'operator', 'model': model, 'site': None, 'table_op': 'rs', 'op': 'rs',
                           'M': m, 'N': hidden, 'K': ffn // tp})
            points.append({'kind': 'operator', 'model': model, 'site': None, 'table_op': 'ag_up', 'op': 'ag',
                           'M': m, 'N': ffn // tp, 'K': hidden})
            points.append({'kind': 'operator', 'model': model, 'site': None, 'table_op': 'ag_gu', 'op': 'ag',
                           'M': m, 'N': 2 * ffn // tp, 'K': hidden})
    for point in points:
        point['point_id'] = f"{point['model']}/{point['table_op']}/M{point['M']}"
    return points


def dense_sites(hidden: int, ffn: int, heads: int, kv_groups: int, rows: int, tp: int,
                head_dim: Optional[int] = None) -> list:
    """Four TP call sites of a dense transformer at `rows` global rows."""
    head_dim = head_dim or hidden // heads
    widths = [(heads + 2 * kv_groups) * head_dim, heads * head_dim, 2 * ffn, ffn]
    if any(width % tp for width in widths):
        raise ValueError('projection widths cannot be partitioned by TP')
    specs = [('qkv', 'ag', 'ag_up', widths[0] // tp, hidden), ('proj', 'rs', 'rs', hidden, widths[1] // tp),
             ('fc1', 'ag', 'ag_gu', widths[2] // tp, hidden), ('fc2', 'rs', 'rs', hidden, widths[3] // tp)]
    return [{'site': site, 'origin': site, 'op': op, 'table_op': table_op, 'M': rows, 'N': n, 'K': k}
            for site, op, table_op, n, k in specs]


def _check(workload: dict) -> dict:
    seen = set()
    for site in workload['sites']:
        key = (site['site'], site['op'], site['M'], site['N'], site['K'])
        if (SITE_OP.get(site['site']) != site['op'] or site['op'] not in OPS or site['M'] % workload['tp']
                or any(not isinstance(site[d], int) or site[d] <= 0 for d in 'MNK')):
            raise ValueError(f'invalid site {key}')
        if key in seen:
            raise ValueError(f'duplicate site {key}')
        seen.add(key)
    return workload


def _rows_list(rows) -> list:
    """`rows` as a sorted list: one int, or '4096,8192' / [..] for several prefill chunk sizes."""
    if rows is None:
        return []
    if isinstance(rows, str):
        rows = [int(x) for x in rows.split(',') if x.strip()]
    if isinstance(rows, int):
        rows = [rows]
    if not rows or len(set(rows)) != len(rows):
        raise ValueError(f'rows must be one positive int or a list of distinct ints: {rows!r}')
    return sorted(int(r) for r in rows)


def build_workload(name: str, tp: int, *, mbs: int = 8, seq: int = 4096, rows=None,
                   config: Optional[str] = None) -> dict:
    """Site manifest of one end-to-end workload.

    bw27b_train / bw70b_train   Megatron dense vehicles, M = seq x mbs (default 4096 x 8 = 32768)
    custom_train                any dense GPT shape vehicle: needs --config
                                {"hidden": .., "ffn": .., "heads": .., "kv_groups": ..[,
                                 "head_dim": ..]} (e2e/megatron/oprs_inner.sh VH_*); M = seq x mbs
    llama70_infer               vLLM Llama3-70B prefill chunk, M = rows (default 16384); widths from the HF config
    hf_infer                    any HF Llama-style config (--config = its config.json); M = rows (default 16384)
    qwen36_train                shape parity with nvidia/planner/original_a800/qwen36_shapes.py: needs the real
                                Qwen3.6 text config (--config). NOT dispatchable on BW today: the Hygon
                                Megatron carrier is the dense bw27b vehicle and has no Qwen3.6 hook.

    `rows` may list several chunk sizes for inference workloads (one plan entry per site per M).
    """
    import json
    rows = rows or (16384 if name in ('llama70_infer', 'hf_infer') else seq * mbs)
    if name in ('bw27b_train', 'bw70b_train'):
        carrier = CARRIERS[name.split('_')[0]]
        sites = dense_sites(rows=rows, tp=tp, **{k: v for k, v in carrier.items() if k != 'layers'})
        note = ('dense GPT shape vehicle of e2e/megatron (random init, mock data), NOT a real Qwen3.6 model; '
                'forward + backward training step')
        source = 'e2e/megatron/run.sh VH_*'
    elif name == 'custom_train':
        if not config:
            raise ValueError('custom_train needs --config: {"hidden": .., "ffn": .., "heads": .., "kv_groups": ..}')
        shape = json.loads(Path(config).read_text()) if Path(config).is_file() else json.loads(config)
        missing = [key for key in ('hidden', 'ffn', 'heads', 'kv_groups') if key not in shape]
        if missing:
            raise ValueError(f'custom_train config lacks {missing}')
        sites = dense_sites(rows=rows, tp=tp, head_dim=shape.get('head_dim'),
                            **{key: shape[key] for key in ('hidden', 'ffn', 'heads', 'kv_groups')})
        note = ('custom dense GPT shape vehicle (random init, mock data): a single-card plumbing check or an '
                'ad-hoc shape, NOT a published model; forward + backward training step')
        source = config
    elif name in ('llama70_infer', 'hf_infer'):
        path = config or (str(Path(__file__).resolve().parents[3] / '3rdparty' / 'models' / 'LLAMA3-70B' / 'config.json')
                          if name == 'llama70_infer' else None)
        if not path:
            raise ValueError('hf_infer needs --config <HF config.json>')
        hf = json.loads(Path(path).read_text())
        sites = [site for m in _rows_list(rows) for site in dense_sites(
            hf['hidden_size'], hf['intermediate_size'], hf['num_attention_heads'],
            hf['num_key_value_heads'], m, tp, hf.get('head_dim'))]
        note, source = (f"vLLM prefill chunk(s) M = {', '.join(str(m) for m in _rows_list(rows))} of the HF model "
                        f'in {path}; forward only'), path
    elif name == 'qwen36_train':
        if not config:
            raise ValueError('qwen36_train needs --config <Qwen3.6 text config.json> (not in this repository)')
        sites = qwen36_sites(json.loads(Path(config).read_text()), rows, tp)
        note, source = ('static code-derived shapes of the Megatron Qwen3.6 hybrid vehicle; runtime verification '
                        'pending; NOT dispatchable on BW: no Hygon Megatron vehicle or hook for this '
                        'architecture exists'), config
    else:
        raise ValueError(f'unknown workload {name}')
    return _check({'workload': name, 'tp': tp, 'dtype': 'bfloat16', 'M': rows, 'sites': sites,
                   'source': source, 'note': note})


def qwen36_sites(config: dict, rows: int, tp: int = 8) -> list:
    """Port of nvidia/planner/original_a800/qwen36_shapes.build_manifest: static shapes of the Megatron
    Qwen3.6 text vehicle (hybrid gated-delta-net / attention / MLP layers), derived from its
    text_config. `site` is the hook call-site name (upstream `adapter_name`); `origin` keeps the
    layer-specific name, because gdn_in and mlp_fc1 both arrive at the fc1 hook (same for fc2)."""
    if rows <= 0 or tp <= 0 or rows % tp:
        raise ValueError('rows must be positive and divisible by TP')
    c = config['text_config']
    h, f = c['hidden_size'], c['intermediate_size']
    heads, kv, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    qk = c['linear_num_key_heads'] * c['linear_key_head_dim']
    value = c['linear_num_value_heads'] * c['linear_value_head_dim']
    gdn_input = 2 * qk + 2 * value + 2 * c['linear_num_value_heads']
    query = heads * dim
    attention_input = query + 2 * kv * dim + (query if c.get('attn_output_gate') else 0)
    dims = [h, f, value, gdn_input, attention_input, query, qk, heads, c['linear_num_value_heads']]
    if any(x % tp for x in dims):
        raise ValueError('projection dimensions cannot be partitioned by TP')
    from collections import Counter
    types = Counter(c['layer_types'])
    if sum(types.values()) != c['num_hidden_layers'] or set(types) != {'linear_attention', 'full_attention'}:
        raise ValueError('unsupported or inconsistent layer types')
    layers = c['num_hidden_layers']
    specs = [('qkv', 'full_qkv', 'ag_up', h, attention_input // tp, types['full_attention']),
             ('proj', 'full_proj', 'rs', query // tp, h, types['full_attention']),
             ('fc1', 'gdn_in', 'ag_up', h, gdn_input // tp, types['linear_attention']),
             ('fc2', 'gdn_out', 'rs', value // tp, h, types['linear_attention']),
             ('fc1', 'mlp_fc1', 'ag_gu', h, 2 * f // tp, layers),
             ('fc2', 'mlp_fc2', 'rs', f // tp, h, layers)]
    return [{'site': site, 'origin': origin, 'op': 'ag' if table_op.startswith('ag') else 'rs',
             'table_op': table_op, 'M': rows, 'N': n, 'K': k, 'layers': count}
            for site, origin, table_op, k, n, count in specs]
