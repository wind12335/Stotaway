import json
import tempfile
import unittest
from pathlib import Path

from planner.original_a800.validate_tp2_new import load
from planner.original_a800.validate_orthogonal_tp2 import route_for


class ProtocolTest(unittest.TestCase):
    def test_missing_warmup_is_unknown_and_cannot_mix_with_known(self):
        data = {'metadata': {'method': 'dx', 'tp': 2, 'dtype': 'bfloat16', 'gpu': 'a800',
                             'torch': '2', 'triton': '3', 'cuda_build': '12',
                             'cuda_device_max_connections': '1', 'batch': 16, 'checks': 1},
                'results': []}
        with tempfile.TemporaryDirectory() as root:
            old, new = Path(root)/'old.json', Path(root)/'new.json'
            old.write_text(json.dumps(data))
            data['metadata']['warmup'] = 3
            new.write_text(json.dumps(data))
            _, unknown_signature, _ = load([old])
            _, known_signature, _ = load([new])
            self.assertIsNone(route_for(unknown_signature, 'rs').warmup)
            self.assertEqual(route_for(known_signature, 'rs').warmup, 3)
            with self.assertRaises(ValueError):
                load([old, new])


if __name__ == '__main__':
    unittest.main()
