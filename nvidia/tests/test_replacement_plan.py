"""Schema checks for three-candidate replacement plans."""
import json
import tempfile
import unittest
from pathlib import Path

from stowaway_nvidia.replacement_plan import _load


class ReplacementPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'plan.json'
        self.plan = {'schema': 'stowaway-tp2-replacement-v1', 'external_backend': 'async_tp',
                     'selection_source': 'manual_coverage', 'runtime': {}, 'entries': [
                         {'site': 'qkv', 'op': 'ag', 'M': 128, 'N': 64, 'K': 32, 'choice': 'dx'},
                         {'site': 'o_proj', 'op': 'rs', 'M': 128, 'N': 64, 'K': 32, 'choice': 'async_tp'}]}

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        self.path.write_text(json.dumps(self.plan))
        return _load(self.path)

    def test_valid_plan_and_digest(self):
        plan, entries, digest = self.load()
        self.assertEqual(plan['external_backend'], 'async_tp')
        self.assertEqual(entries[('qkv', 'ag', 128, 64, 32)], 'dx')
        self.assertEqual(len(digest), 64)

    def test_rejects_duplicate_shape_and_foreign_backend(self):
        self.plan['entries'].append(dict(self.plan['entries'][0]))
        with self.assertRaises(ValueError):
            self.load()
        self.plan['entries'].pop()
        self.plan['external_backend'] = 'flashoverlap'
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_unknown_choice_source_and_invalid_dimensions(self):
        self.plan['entries'][0]['choice'] = 'flux'
        with self.assertRaises(ValueError):
            self.load()
        self.plan['entries'][0]['choice'] = 'dx'
        self.plan['entries'][0]['M'] = 127
        with self.assertRaises(ValueError):
            self.load()
        self.plan['entries'][0]['M'] = 128
        self.plan['selection_source'] = 'predictor'
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()
