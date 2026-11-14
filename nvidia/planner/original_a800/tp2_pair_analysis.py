"""Compare independent TP2 batches without relabeling individual timings."""
import argparse
import json
import math
import statistics
from pathlib import Path


def read(path):
    data = json.loads(path.read_text())
    meta = data['metadata']
    keys = ('tp', 'dtype', 'gpu', 'torch', 'triton', 'cuda_build', 'cuda_device_max_connections', 'batch', 'rows')
    rows = {(x['model'], x['site'], x['M'], x['K'], x['N']): x for x in data['results']}
    if len(rows) != len(data['results']):
        raise ValueError('duplicate keys')
    return tuple(meta[k] for k in keys), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('first', type=Path)
    ap.add_argument('repeat', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sig1, first = read(args.first)
    sig2, repeat = read(args.repeat)
    if sig1 != sig2 or first.keys() != repeat.keys():
        raise ValueError('mismatched signatures or cells')
    result = []
    for key in sorted(first):
        a, b = first[key]['measurements'], repeat[key]['measurements']
        ratios = [p['stock']['median_ms']/p['dx']['median_ms'] for p in (a, b)]
        result.append({'key': key, 'first_speedup': ratios[0], 'repeat_speedup': ratios[1],
                       'winner_changed': (ratios[0] > 1) != (ratios[1] > 1),
                       'stock_change': b['stock']['median_ms']/a['stock']['median_ms'],
                       'dx_change': b['dx']['median_ms']/a['dx']['median_ms']})
    data = {'source_first': str(args.first), 'source_repeat': str(args.repeat), 'signature': sig1,
            'count': len(result), 'winner_changed': sum(x['winner_changed'] for x in result),
            'median_absolute_speedup_change': statistics.median(abs(x['repeat_speedup']-x['first_speedup']) for x in result),
            'geomean_stock_drift': math.exp(statistics.mean(math.log(x['stock_change']) for x in result)),
            'geomean_dx_drift': math.exp(statistics.mean(math.log(x['dx_change']) for x in result)), 'rows': result}
    with args.output.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k,v in data.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
