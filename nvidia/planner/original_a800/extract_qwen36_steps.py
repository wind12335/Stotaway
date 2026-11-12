"""Index existing Qwen3.6 TP8 step logs with provenance; no new GPU work.

Cross-window pairs are explicitly marked exploratory. These observations do
not form a complete or independent training-step calibration dataset.
"""
import hashlib
import json
import re
import statistics
from pathlib import Path

RAW = Path('/root/data/stowaway-nvidia/results/raw')
OUT = Path(__file__).resolve().parent / 'data/qwen36_step_inventory.json'

# Arm assignments follow the existing experiment report, not filename guesses.
RUNS = {
    'mbs1_forward': {'base': ['tp8_qwen36_tune12_stock_r0_20261111.log', 'tp8_qwen36_tune12_stock_r1_20261111.log'],
                     'row': ['tp8_qwen36_tune12_dx_r0_20261111.log', 'tp8_qwen36_tune12_dx_r1_20261111.log']},
    'mbs1_full': {'base': ['qwen36_full_dx_base_mbs1_b1_20261111.log', 'qwen36_full_dx_base_mbs1_b2_20261111.log'],
                  'fbag': ['qwen36_full_dx_fbag_mbs1_b1_20261111.log', 'qwen36_full_dx_fbag_mbs1_b2_20261111.log']},
    'mbs2': {'base': ['qwen36_mbs2_base_a1_20261111.log', 'qwen36_mbs2_base_a2_20261111.log',
                      'qwen36_mbs2_full_base_b1_20261111.log'],
             'row': ['qwen36_mbs2_row_a1_20261111.log', 'qwen36_mbs2_row_a2_20261111.log'],
             'fbag': ['qwen36_mbs2_fbag_a1_20261111.log', 'qwen36_mbs2_fbag_a2_20261111.log'],
             'fbc': ['qwen36_mbs2_fbc_a1_20261111.log', 'qwen36_mbs2_fbc_a2_20261111.log']},
    'mbs4': {'base': ['qwen36_mbs4_base_a1_20261111.log', 'qwen36_mbs4_base_a2_20261111.log'],
             'row': ['qwen36_mbs4_row_a1_20261111.log', 'qwen36_mbs4_row_a2_20261111.log'],
             'fbag': ['qwen36_mbs4_fbag_a1_20261111.log'],
             'fbc': ['qwen36_mbs4_fbc_a1_20261111.log']},
    'mbs8': {'base': ['qwen36_mbs8_base_a1_expand_20261111.log', 'qwen36_mbs8_base_a2_expand_20261111.log'],
             'row': ['qwen36_mbs8_row_a1_expand_20261111.log', 'qwen36_mbs8_row_a2_expand_20261111.log']},
}


def read_log(name):
    path = RAW / name
    raw = path.read_bytes()
    s = raw.decode(errors='replace')
    timings = [float(x) for x in re.findall(r'elapsed time per iteration \(ms\):\s*([0-9.]+)', s)]
    skipped = [int(x) for x in re.findall(r'number of skipped iterations:\s*(\d+)', s)]
    nan = [int(x) for x in re.findall(r'number of nan iterations:\s*(\d+)', s)]
    if len(timings) != 12 or len(skipped) != 12 or len(nan) != 12 or any(skipped + nan) or '[after training is done]' not in s:
        raise ValueError('incomplete or invalid log: ' + name)
    dx = re.search(r'STOWAWAY_DIAGNOSTIC_COUNTS rank=0 row=(\d+) column=(\d+)', s)
    brs = re.search(r"STOWAWAY_BRS_COUNTS rank=0 hits=\{'layers':\s*(\d+), 'mappings':\s*(\d+)\}", s)
    return {'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'steps': len(timings), 'warmup_excluded': 3, 'median_ms': statistics.median(timings[3:]),
            'rank0_dx_row_column': list(map(int, dx.groups())) if dx else None,
            'rank0_brs_layers_mappings': list(map(int, brs.groups())) if brs else None}


def main():
    result = {'classification': 'inventory only; not matched training predictor calibration',
              'model': 'Qwen3.6 text hybrid vehicle', 'tp': 8,
              'warning': 'runs and baselines may be from different windows; see source report',
              'runs': {group: {arm: [read_log(path) for path in paths]
                               for arm, paths in arms.items()} for group, arms in RUNS.items()}}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for group, arms in result['runs'].items():
        print(group, {arm: [r['median_ms'] for r in runs] for arm, runs in arms.items()})


if __name__ == '__main__':
    main()
