import copy
import json
import unittest
from pathlib import Path

from planner.original_a800.qwen36_shapes import build_manifest
from planner.original_a800.qwen36_plan import build_report
from tests.test_qwen36_shapes import config


class TestQwenPlan(unittest.TestCase):
    def setUp(self):
        self.manifest = build_manifest(config(), 8192)
        self.probe = {'classification': 'single-gpu-gemm-probe', 'tp': 8, 'M': 8192,
                      'dtype': 'bfloat16', 'model': self.manifest['model'],
                      'gpu': 'NVIDIA A800-SXM4-80GB', 'batch': 16, 'repeats': 8,
                      'results': [{**{k: s[k] for k in ('site', 'M', 'K', 'N')},
                                   'single_median_ms': 1.0, 'b2b_median_ms': 0.9}
                                  for s in self.manifest['sites']]}
        path = Path(__file__).resolve().parents[1] / 'planner/original_a800/data/cells.json'
        self.cells = json.loads(path.read_text())

    def test_qwen_output_remains_non_deployable(self):
        report = build_report(self.manifest, self.probe, self.cells)
        self.assertEqual(len(report['results']), 6)
        self.assertIn('not an E2E training plan', report['classification'])
        self.assertEqual(report['results'][0]['G_single_ms'], 1.0)

    def test_wrong_shape_or_runtime_is_rejected(self):
        invalid = copy.deepcopy(self.probe)
        invalid['results'][0]['K'] += 1
        with self.assertRaises(ValueError):
            build_report(self.manifest, invalid, self.cells)
        invalid = copy.deepcopy(self.probe)
        invalid['tp'] = 2
        with self.assertRaises(ValueError):
            build_report(self.manifest, invalid, self.cells)


if __name__ == '__main__':
    unittest.main()
