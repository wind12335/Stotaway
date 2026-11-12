"""Create an exploratory Qwen text-vehicle operator plan, not a training plan.

Uses measured single-GPU G inputs and existing A800 operator calibration.
GDN and MLP share current adapter switches; distinct per-shape choices here
must not be applied to Megatron until a shape-aware adapter is implemented and
the training step model has been measured.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .costmodel import CostModel
    from .relative_model import RelativeModel
except ImportError:
    from costmodel import CostModel
    from relative_model import RelativeModel


def build_report(manifest, probes, cells):
    if probes.get('classification') != 'single-gpu-gemm-probe' or probes.get('tp') != manifest['tp']:
        raise ValueError('expected matching single-GPU GEMM probe')
    if probes.get('M') != manifest['M'] or probes.get('dtype') != manifest['dtype']:
        raise ValueError('probe shape/dtype mismatch')
    if (probes.get('model') != manifest.get('model') or probes.get('gpu') != 'NVIDIA A800-SXM4-80GB'
            or probes.get('batch') != 16 or probes.get('repeats', 0) < 8):
        raise ValueError('probe platform, model or sampling mismatch')
    index = {r['site']: r for r in probes['results']}
    if len(index) != len(probes['results']):
        raise ValueError('duplicate probe site')
    original = CostModel(mode='b2b').fit(cells)
    relative = RelativeModel(mode='b2b').fit(cells)
    results = []
    for site in manifest['sites']:
        p = index.get(site['site'])
        if p is None or any(p[key] != site[key] for key in ('M', 'K', 'N')):
            raise ValueError('missing or mismatched probe for ' + site['site'])
        if any(not math.isfinite(p[k]) or p[k] <= 0 for k in ('single_median_ms', 'b2b_median_ms')):
            raise ValueError('invalid probe timings for ' + site['site'])
        measured = {'op': site['op'], 'M': site['M'], 'K': site['K'], 'N': site['N'],
                    'G': p['single_median_ms'], 'B': site['payload_MB']}
        orig_pick, original_times = original.choose(site['op'], measured['G'], measured['B'], allowed={'dx'})
        rel_pick, rel_prediction = relative.choose(measured)
        group = [c for c in cells if c['op'] == site['op'] and not c.get('anomaly')]
        ranges = {key: (min(float(c[key]) for c in group), max(float(c[key]) for c in group))
                  for key in ('M', 'K', 'N', 'G', 'B')}
        outside = [key for key, (lo, hi) in ranges.items() if not lo <= float(measured[key]) <= hi]
        results.append({**site, 'G_single_ms': measured['G'], 'G_b2b_ms': p['b2b_median_ms'], 'original_choice': orig_pick,
                        'original_pred_ms': original_times, 'relative_choice': rel_pick,
                        'relative_prediction': rel_prediction, 'outside_calibration_ranges': outside,
                        'cautious_choice': 'stock' if outside or orig_pick != rel_pick else orig_pick})
    return {'classification': 'exploratory operator-only; not an E2E training plan',
            'model': manifest['model'], 'tp': manifest['tp'], 'M': manifest['M'],
            'warnings': ['GDN uses the same adapter names as MLP; per-shape choices not yet deployable',
                         'no backward/step calibration; do not interpret as training speedup',
                         'G probe is single GPU; existing G is single call over TP8 ranks; windows differ',
                         'cautious_choice is diagnostic agreement and range filtering, not a safety guarantee'],
            'results': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--gemm-probe', type=Path, required=True)
    p.add_argument('--cells', type=Path, default=Path(__file__).parent / 'data' / 'cells.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = build_report(json.loads(a.manifest.read_text()), json.loads(a.gemm_probe.read_text()),
                          json.loads(a.cells.read_text()))
    report['calibration_sha256'] = hashlib.sha256(a.cells.read_bytes()).hexdigest()
    report['gemm_probe_sha256'] = hashlib.sha256(a.gemm_probe.read_bytes()).hexdigest()
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
