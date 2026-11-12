"""Static shape manifest for the measured Megatron Qwen3.6 text vehicle.

Derived from its text_config and Megatron 571370c local GDN/attention/MLP
constructors. It is not a runtime trace or a complete HF weight converter.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def build_manifest(config, rows, tp=8):
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
    types = Counter(c['layer_types'])
    if sum(types.values()) != c['num_hidden_layers'] or set(types) != {'linear_attention', 'full_attention'}:
        raise ValueError('unsupported or inconsistent layer types')
    def site(key, layer, label, op, k, n, count):
        return {'site': key, 'layer_type': layer, 'adapter_name': label, 'op': op,
                'M': rows, 'K': k, 'N': n, 'layers': count,
                'payload_MB': rows * (n if op == 'rs' else k) * 2 / 1e6,
                'status': 'static code-derived shape; runtime verification pending'}
    layer_count = c['num_hidden_layers']
    sites = [
        site('full_qkv', 'full_attention', 'qkv', 'ag_up', h, attention_input // tp, types['full_attention']),
        site('full_proj', 'full_attention', 'proj', 'rs', query // tp, h, types['full_attention']),
        site('gdn_in', 'linear_attention', 'fc1', 'ag_up', h, gdn_input // tp, types['linear_attention']),
        site('gdn_out', 'linear_attention', 'fc2', 'rs', value // tp, h, types['linear_attention']),
        site('mlp_fc1', 'all', 'fc1', 'ag_gu', h, 2 * f // tp, layer_count),
        site('mlp_fc2', 'all', 'fc2', 'rs', f // tp, h, layer_count),
    ]
    return {'model': 'Qwen3.6-27B-text-megatron-local-vehicle', 'tp': tp, 'M': rows,
            'dtype': 'bfloat16', 'layer_types': dict(types), 'sites': sites,
            'assumptions': ['BF16', 'sequence parallel', 'SwiGLU', 'local Megatron hybrid spec',
                            'global M = sequence length times microbatch', 'forward only'],
            'classification': 'static manifest; not runtime verified; not predictor calibration'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('/root/eb-public/huggingface-models/Qwen/Qwen3.6-27B/config.json'))
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.config.read_text()), args.rows, args.tp)
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as f:
            f.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
