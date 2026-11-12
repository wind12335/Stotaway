import json
import tempfile
import unittest
from pathlib import Path

from stowaway_nvidia.shape_plan import ShapePlan


class TestShapePlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'plan.json'
        self.runtime = {'tp': 8, 'dtype': 'bfloat16', 'gpu': 'A800', 'torch_version': '2.7',
                        'triton_version': '3.3', 'cuda_build': '12.6', 'schedule': 'shift', 'batch_ack': True,
                        'backward': True, 'brs': False, 'brs_min_bytes': 0, 'brs_slots': 1,
                        'cuda_device_max_connections': '1'}
        self.data = {'schema': 'stowaway-dx-shape-plan-v1', 'classification': 'diagnostic-manual',
                     'backend': 'dx', 'runtime': {**self.runtime, 'torch': self.runtime['torch_version'],
                                                  'triton': self.runtime['triton_version']},
                     'entries': [{'site': 'fc2', 'M': 8192, 'K': 768, 'N': 5120, 'choice': 'dx'},
                                 {'site': 'fc2', 'M': 8192, 'K': 2176, 'N': 5120, 'choice': 'stock'}]}
        del self.data['runtime']['torch_version']
        del self.data['runtime']['triton_version']

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        self.path.write_text(json.dumps(self.data))
        return ShapePlan(self.path, **self.runtime)

    def test_distinguishes_same_site_name_by_shape(self):
        p = self.load()
        self.assertEqual(p.choose('fc2', 8192, 768, 5120, 'bfloat16'), 'dx')
        self.assertEqual(p.choose('fc2', 8192, 2176, 5120, 'bfloat16'), 'stock')
        self.assertEqual(p.choose('fc2', 4096, 768, 5120, 'bfloat16'), 'stock')
        self.assertEqual(p.choose('fc2', 8192, 768, 5120, 'float16'), 'stock')

    def test_rejects_runtime_and_duplicate_conflicts(self):
        self.data['runtime']['brs'] = True
        with self.assertRaises(ValueError):
            self.load()
        self.data['runtime']['brs'] = False
        self.data['entries'].append(dict(self.data['entries'][0]))
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()
