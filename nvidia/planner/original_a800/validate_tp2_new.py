"""TP2 out-of-shape-family evaluation; CPU-only, no GPU benchmark runs.

All M values and repeated occurrences with the same (direction,K,N) are held out
together. Uses same-run GEMM G as an input; communication curve and guard are
fit only from training families. Keeps matched runtime signatures separate.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

try:
    from .relative_model import RelativeModel
except ImportError:
    from relative_model import RelativeModel


def load(paths):
    cells, signatures, seen = [], set(), set()
    sources = []
    for path in paths:
        raw = path.read_bytes()
        report = json.loads(raw)
        meta = report['metadata']
        keys = ('tp', 'dtype', 'gpu', 'torch', 'triton', 'cuda_build', 'cuda_device_max_connections', 'batch')
        method = meta.get('method', 'dx')
        signatures.add((method,) + tuple(meta[k] for k in keys) +
                       (meta.get('triton_dist_path', ''), meta.get('autotune', ''), meta.get('warmup')))
        if meta.get('rs_actual_path', 'direct_exchange') != 'direct_exchange':
            raise ValueError('scratch RS requires its own route-specific evaluation; original DX models are incompatible')
        if meta['tp'] != 2 or meta['batch'] != 16 or meta['checks'] < 1:
            raise ValueError(f'not a checked TP2 16-call batch: {path}')
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        for e in report['results']:
            op = 'rs' if e['op'] == 'rs' else {'qkv': 'ag_up', 'fc1': 'ag_gu'}[e['site']]
            unique = (path, e['model'], e['site'], e['M'], e['K'], e['N'])
            if unique in seen:
                raise ValueError('duplicate records')
            seen.add(unique)
            measure = e['measurements']
            # qkv and fc1 can use an identical physical AG shape; keep both
            # out of training when testing either one.
            family = f"{e['op']}:{e['K']}:{e['N']}"
            cells.append({'model': family, 'source_model': e['model'], 'site': e['site'], 'op': op,
                          'M': e['M'], 'K': e['K'], 'N': e['N'], 'B': e['payload_MB'],
                          'G': measure['gemm']['median_ms'], 'C': measure['comm']['median_ms'],
                          'stock_b2b': measure['stock']['median_ms'],
                          'dx_b2b': measure['dx' if method == 'dx' else 'overlap']['median_ms']})
    if len(signatures) != 1:
        raise ValueError('runtime signature mismatch across source files')
    return cells, list(signatures)[0], sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cells, signature, sources = load(args.inputs)
    families = sorted({c['model'] for c in cells})
    output = []
    for held in families:
        model = RelativeModel(mode='b2b').fit([c for c in cells if c['model'] != held])
        for c in (c for c in cells if c['model'] == held):
            try:
                choice, prediction = model.choose(c)
            except ValueError:
                choice, prediction = 'stock', None
            oracle = 'dx' if c['dx_b2b'] < c['stock_b2b'] else 'stock'
            chosen = c['dx_b2b'] if choice == 'dx' else c['stock_b2b']
            output.append({**c, 'choice': choice, 'oracle': oracle, 'prediction': prediction,
                           'speedup': c['stock_b2b']/chosen})
    def summary(rows):
        gm = lambda v: math.exp(statistics.mean(math.log(x) for x in v))
        return {'count': len(rows), 'correct': sum(x['choice'] == x['oracle'] for x in rows),
                'speedup_gmean': gm([x['speedup'] for x in rows]),
                'slower': sum(x['speedup'] < 1 for x in rows),
                'always_dx_correct': sum(x['oracle'] == 'dx' for x in rows),
                'always_dx_gmean': gm([x['stock_b2b']/x['dx_b2b'] for x in rows]),
                'always_stock_correct': sum(x['oracle'] == 'stock' for x in rows),
                'oracle_gmean': gm([max(1, x['stock_b2b']/x['dx_b2b']) for x in rows])}
    result = {'description': 'TP2 exploratory leave-(direction,K,N)-out; duplicate shapes stay together',
              'signature': signature, 'sources': sources, 'families': len(families),
              'summary': summary(output), 'by_op': {op: summary([r for r in output if r['op'] == op])
              for op in sorted({r['op'] for r in output})}, 'rows': output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('families', 'summary', 'by_op')}, indent=2))


if __name__ == '__main__':
    main()
