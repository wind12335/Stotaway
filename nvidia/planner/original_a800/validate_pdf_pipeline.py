"""CPU-only leave-one-shape-out exploratory baseline on FlashOverlap PDF rows.

Each run's separately measured GEMM and communication serve as cheap inputs;
the equal-chunk approximation is not the repository's actual wave grouping.
No device identity is assumed and no Flux/DX data are mixed into this fit.
"""
import json
import math
import statistics
from pathlib import Path

from pipeline_subset import pipeline_makespan, fit_linear_residual

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / 'data/reference_pdf_records.json').read_text())
    result = {}
    for kind in ('ar', 'rs'):
        rows = [r for r in data['rows'] if r['method'] == 'flashoverlap' and r['op'] == kind]
        groups = {}
        for row in rows:
            groups.setdefault((row['M'], row['N'], row['K']), []).append(row)
        if len(groups) < 3:
            result[kind] = {'status': 'insufficient distinct shapes for leave-one-shape-out', 'shapes': len(groups)}
            continue
        entries = []
        for held, trials in groups.items():
            train = [r for key, vals in groups.items() if key != held for r in vals]
            # Fixed J=4 = TP count as a crude baseline; not inferred as the
            # actual FlashOverlap wave grouping and not tuned on held-out data.
            def x(r):
                return pipeline_makespan([r['gemm_ms']/4]*4, [r['comm_ms']/4]*4)
            fit = fit_linear_residual([x(r) for r in train], [r['overlap_ms'] for r in train])
            train_err = sorted(r['overlap_ms'] - (fit['scale']*x(r) + fit['fixed_ms']) for r in train)
            guard = max(0.0, train_err[math.ceil(.8*len(train_err))-1])
            stock = statistics.median(r['stock_ms'] for r in trials)
            measured = statistics.median(r['overlap_ms'] for r in trials)
            pred = fit['scale'] * statistics.median(x(r) for r in trials) + fit['fixed_ms']
            choice = 'overlap' if pred + guard < stock else 'stock'
            entries.append({'shape_MNK': held, 'repeats': len(trials), 'stock_ms': stock,
                            'overlap_ms': measured, 'pred_overlap_ms': pred, 'guard_ms': guard,
                            'choice': choice, 'oracle': 'overlap' if measured < stock else 'stock',
                            'calibration': fit})
        result[kind] = {'status': 'exploratory; only 3 shapes, no independent generalization claim',
                        'shapes': len(entries), 'correct': sum(e['choice'] == e['oracle'] for e in entries),
                        'entries': entries}
    dest = HERE / 'data/pdf_pipeline_exploratory.json'
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
