"""Extract bounded, traceable 4-GPU operator records from supplied PDF text.

Only completed benchmark blocks are included; errors are recorded separately.
This preserves per-rank measurements, which are not independent repetitions.
"""
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/research_sources'
DEST = Path(__file__).resolve().parent / 'data/reference_pdf_records.json'
NUMBER = r'([0-9]+(?:\.[0-9]+)?)'


def flash(kind):
    name = 'flashoverlap_' + kind
    s = (SOURCE / (name + '.txt')).read_text()
    blocks = re.split(r'(?=\bItem\s+Value\s+)', s)
    rows = []
    for block in blocks[1:]:
        values = {}
        for field in ('m', 'n', 'k', 'tile_num', 'gemm_dur', 'comm_dur', 'baseline_dur', 'overlap_dur', 'speedup'):
            match = re.search(r'(?m)^\s*' + field + r'(?:\s*\(ms\))?\s+' + NUMBER, block)
            if match:
                values[field] = float(match.group(1))
        if len(values) != 9:
            raise ValueError('incomplete FlashOverlap table: ' + name + ' ' + repr(values))
        if abs(values['baseline_dur']/values['overlap_dur'] - values['speedup']) > 0.001:
            raise ValueError('speedup mismatch')
        offset = s.index(block)
        rows.append({'method': 'flashoverlap', 'op': kind, 'M': int(values['m']), 'N': int(values['n']),
                     'K': int(values['k']), 'tp': 4, 'dtype': 'float16', 'gemm_ms': values['gemm_dur'],
                     'comm_ms': values['comm_dur'], 'stock_ms': values['baseline_dur'],
                     'overlap_ms': values['overlap_dur'], 'speedup_reported': values['speedup'],
                     'tile_count': int(values['tile_num']), 'source': '/root/data/' + name + '.pdf',
                     'pdf_page': s[:offset].count('\f') + 1, 'repeat_kind': 'separate test invocation'})
    return rows


def flux(kind):
    name = 'flux_' + kind
    s = (SOURCE / (name + '.txt')).read_text()
    chunks = re.split(r'(?=\./launch\.sh\s+test/python/)', s)
    rows = []
    for block in chunks[1:]:
        args = re.search(r'test_(?:ag_kernel|gemm_rs)\.py\s+(\d+)\s+(\d+)\s+(\d+)', block)
        if not args:
            raise ValueError('missing command in ' + name)
        m, global_n, global_k = map(int, args.groups())
        pattern = r'(torch|flux)\s*#(\d+):(?:(?!\b(?:torch|flux)\s*#).)*?total\s+([0-9.]+)\s*ms'
        records = re.findall(pattern, block, flags=re.DOTALL)
        timings = {arm: {int(rank): float(ms) for arm0, rank, ms in records if arm0 == arm}
                   for arm in ('torch', 'flux')}
        if any(set(timings[arm]) != {0, 1, 2, 3} for arm in timings):
            raise ValueError('missing per-rank total: ' + name + str(args.groups()))
        offset = s.index(block)
        rows.append({'method': 'flux', 'op': kind, 'command_M': m, 'command_N': global_n,
                     'command_K': global_k, 'M': m, 'K': global_k if kind == 'ag' else global_k // 4,
                     'N': global_n // 4 if kind == 'ag' else global_n,
                     'tp': 4, 'dtype': 'float16', 'iters': 1 if kind == 'ag' else 10,
                     'stock_by_rank_ms': timings['torch'], 'overlap_by_rank_ms': timings['flux'],
                     'stock_max_rank_ms': max(timings['torch'].values()),
                     'overlap_max_rank_ms': max(timings['flux'].values()),
                     'speedup_max_rank': max(timings['torch'].values()) / max(timings['flux'].values()),
                     'rank0_speedup': timings['torch'][0] / timings['flux'][0],
                     'allclose_text': 'flux check passed' in block,
                     'bitwise_failure_text': 'bitwise check failed' in block or 'is bitwise match: False' in block,
                     'source': '/root/data/' + name + '.pdf', 'pdf_page': s[:offset].count('\f') + 1,
                     'repeat_kind': 'one invocation; ranks are not repeats'})
    return rows


def main():
    rows = flash('ar') + flash('rs') + flux('ag') + flux('rs')
    sources = {}
    for name in ('flashoverlap_ar', 'flashoverlap_rs', 'flux_ag', 'flux_rs'):
        path = Path('/root/data') / (name + '.pdf')
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = {'classification': 'supplied 4-GPU operator logs; GPU and topology unknown',
            'tp_provenance': 'user confirmed all four PDF files are 4-GPU results',
            'warning': ('not model E2E, not TP8, FP16; Flux AG iters=1; no cross-platform transfer; '
                        'Flux RS command argument order used even where nearby heading conflicts'),
            'sources_sha256': sources, 'rows': rows,
            'failures': [{'method': 'flashoverlap', 'op': 'ar', 'M': 8192, 'N': 29568, 'K': 8192,
                          'reason': 'NCCL unhandled CUDA error during search'},
                         {'method': 'flashoverlap', 'op': 'ar', 'M': 16384, 'N': 29568, 'K': 8192,
                          'reason': 'NCCL unhandled CUDA error during search'},
                         {'method': 'flashoverlap', 'op': 'rs', 'M': 8192, 'N': 29568, 'K': 8192,
                          'reason': 'partial tile count assertion in search'}]}
    DEST.write_text(json.dumps(data, indent=2) + '\n')
    print('extracted', len(rows), 'rows', {name: sum(r['method'] == name for r in rows) for name in ('flux','flashoverlap')})
    for method, op in (('flashoverlap','ar'), ('flashoverlap','rs'), ('flux','ag'), ('flux','rs')):
        subset = [r for r in rows if r['method'] == method and r['op'] == op]
        print(method, op, len(subset))
        for r in subset:
            print(' ', (r['M'], r['N'], r['K']), round(r.get('speedup_reported',r.get('speedup_max_rank')), 4))


if __name__ == '__main__':
    main()
