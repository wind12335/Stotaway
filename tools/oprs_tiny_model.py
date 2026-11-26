"""A tiny Llama-style model directory for the single-card plumbing check of the vLLM entry.

  python3 tools/oprs_tiny_model.py <new dir> [--from-model DIR]

Writes config.json (hidden 1024, intermediate 3584, 8 heads, 8 KV heads, head_dim 64, 4 layers,
Llama-3 rope, vocab 128256) and copies the tokenizer files and the natural-text files of the gate
(README.md, LICENSE, USE_POLICY.md) from a real Llama model directory (default: $SW_VLLM_MODEL, else
3rdparty/models/LLAMA3-70B). There are no weights: run it with VS_LOAD_FORMAT=dummy.

A plumbing vehicle only -- random weights of a shape that is nobody's model. It shows that the
entry script, the hook, the plan and the report run on one card; none of its numbers (time or
output) says anything about a real model, and with random weights the natural-text gate has no
"confident copy" to compare against.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    'architectures': ['LlamaForCausalLM'], 'attention_bias': False, 'attention_dropout': 0.0,
    'bos_token_id': 128000, 'eos_token_id': 128001, 'hidden_act': 'silu', 'hidden_size': 1024,
    'initializer_range': 0.02, 'intermediate_size': 3584, 'max_position_embeddings': 131072,
    'mlp_bias': False, 'model_type': 'llama', 'num_attention_heads': 8, 'num_hidden_layers': 4,
    'num_key_value_heads': 8, 'pretraining_tp': 1, 'rms_norm_eps': 1e-05,
    'rope_scaling': {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
                     'original_max_position_embeddings': 8192, 'rope_type': 'llama3'},
    'rope_theta': 500000.0, 'tie_word_embeddings': False, 'torch_dtype': 'bfloat16',
    'use_cache': True, 'vocab_size': 128256, 'head_dim': 64,
}
COPIES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json',
          'README.md', 'LICENSE', 'USE_POLICY.md')


def main(argv=None):
    parser = argparse.ArgumentParser(description='tiny Llama-style model directory for a single-card plumbing check')
    parser.add_argument('out', type=Path, help='NEW directory')
    parser.add_argument('--from-model', help='real Llama model directory (tokenizer + text files)')
    args = parser.parse_args(argv)
    source = Path(args.from_model or os.environ.get('SW_VLLM_MODEL') or ROOT / '3rdparty' / 'models' / 'LLAMA3-70B')
    if args.out.exists():
        sys.exit(f'refusing to overwrite {args.out}')
    missing = [name for name in COPIES if not (source / name).is_file()]
    if missing:
        sys.exit(f'{source} lacks {missing}: give --from-model <Llama model directory>')
    args.out.mkdir(parents=True)
    with (args.out / 'config.json').open('x') as handle:
        json.dump(CONFIG, handle, indent=1)
        handle.write('\n')
    for name in COPIES:
        shutil.copy2(source / name, args.out / name)
    with (args.out / 'NOTE.txt').open('x') as handle:
        handle.write('tiny plumbing vehicle made by tools/oprs_tiny_model.py: no weights (run with '
                     'VS_LOAD_FORMAT=dummy); tokenizer and text files copied from '
                     f'{source}\n')
    print(f'{args.out}: hidden {CONFIG["hidden_size"]}, ffn {CONFIG["intermediate_size"]}, '
          f'{CONFIG["num_hidden_layers"]} layers, vocab {CONFIG["vocab_size"]}; tokenizer + text from {source}')


if __name__ == '__main__':
    main()
