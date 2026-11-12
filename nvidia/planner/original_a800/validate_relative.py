"""CPU-only exploratory outer leave-one-model evaluation; never changes old results."""
import hashlib
import json
import math
from pathlib import Path
import argparse

import numpy as np
from costmodel import CostModel
from relative_model import RelativeModel

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'cells.json'
OUT = HERE / 'data' / 'relative_exploratory.json'
REPORT = HERE / 'data' / 'relative_exploratory.md'


def summarize(rows):
    gm = lambda values: math.exp(sum(math.log(v) for v in values) / len(values))
    return {'n': len(rows), 'correct': sum(r['pick'] == r['oracle'] for r in rows),
            'speedup_gmean': gm([r['stock'] / r['chosen'] for r in rows]),
            'mean_regret_pct': 100 * np.mean([r['chosen'] / r['best'] - 1 for r in rows]),
            'max_regret_pct': 100 * max(r['chosen'] / r['best'] - 1 for r in rows),
            'slower_gt_0_5pct': sum(r['chosen'] > r['stock'] * 1.005 for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-data', type=Path, help='independently measured single-GPU split GEMM ratios')
    parser.add_argument('--output-prefix', type=Path, default=HERE / 'data' / 'relative_exploratory')
    args = parser.parse_args()
    raw = DATA.read_bytes()
    cells = [c for c in json.loads(raw) if not c.get('anomaly')]
    split_hash = None
    if args.split_data:
        split_raw = args.split_data.read_bytes()
        split_hash = hashlib.sha256(split_raw).hexdigest()
        data = json.loads(split_raw)
        if data['metadata']['scope'] != 'single GPU, full vs 8-slice GEMM only, no communication':
            raise ValueError('unexpected split GEMM measurement')
        key = lambda c: tuple(c[k] for k in ('model', 'op', 'M', 'K', 'N'))
        lookup = {key(c): c['sliced_over_full'] for c in data['results']}
        if len(lookup) != len(data['results']) or any(key(c) not in lookup for c in cells):
            raise ValueError('split measurement is incomplete or has duplicate shapes')
        cells = [{**c, 'split_gemm_ratio': lookup[key(c)]} for c in cells]
    models = sorted({c['model'] for c in cells})
    out = {'data_sha256': hashlib.sha256(raw).hexdigest(), 'validation': 'outer leave-one-model-out',
           'split_data_sha256': split_hash,
           'status': 'exploratory; tuning on this dataset precludes a final independent claim', 'modes': {}}
    for mode in ('b2b', 'single'):
        rows = {'original': [], 'relative': []}
        if args.split_data:
            rows['relative_split'] = []
        for held in models:
            train = [c for c in cells if c['model'] != held]
            cm = CostModel(mode=mode).fit(train)
            rm = RelativeModel(mode=mode).fit(train)
            sm = RelativeModel(mode=mode, split_feature=True).fit(train) if args.split_data else None
            for c in (c for c in cells if c['model'] == held):
                stock, dx = c['stock_' + mode], c['dx_' + mode]
                picks = [('original', cm.choose(c['op'], c['G'], c['B'], allowed={'dx'})[0]),
                         ('relative', rm.choose(c)[0])]
                if sm:
                    picks.append(('relative_split', sm.choose(c)[0]))
                for name, pick in picks:
                    rows[name].append({'model': held, 'op': c['op'], 'M': c['M'], 'K': c['K'], 'N': c['N'],
                                       'pick': pick, 'oracle': 'dx' if dx < stock else 'stock',
                                       'stock': stock, 'dx': dx, 'chosen': dx if pick == 'dx' else stock,
                                       'best': min(stock, dx)})
        out['modes'][mode] = {name: {'summary': summarize(items), 'rows': items} for name, items in rows.items()}
    args.output_prefix.with_suffix('.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    lines = ['# 相对收益模型：A800 既有算子数据探索性验证', '',
             '仅 CPU 重用既有 TP8 测量；候选为 stock 与 DX，16 连发为主。外层留一模型；',
             '相对模型的安全门在外层训练模型内部，再按模型留一生成预测残差。',
             '这份数据已经用于研发；下表属于探索结果，不是独立最终检验。', '',
             '| 计时 | 模型 | 选对 | 几何平均 speedup | 平均 / 最差损失 | 比原版慢 >0.5% |',
             '|---|---|---:|---:|---:|---:|']
    for mode, tests in out['modes'].items():
        for name, entry in tests.items():
            s = entry['summary']
            lines.append(f"| {mode} | {name} | {s['correct']}/{s['n']} | {s['speedup_gmean']:.3f}× | "
                         f"{s['mean_regret_pct']:.2f}% / {s['max_regret_pct']:.2f}% | {s['slower_gt_0_5pct']} |")
    lines += ['', f"数据 SHA256：`{out['data_sha256']}`。",
              f"分块数据 SHA256：`{split_hash}`。" if split_hash else '未使用分块探针。',
              f'完整逐点结果见 `{args.output_prefix.name}.json`。']
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
