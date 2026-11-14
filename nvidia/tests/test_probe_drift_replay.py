import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.probe_drift_replay import replay

class DriftReplayTest(unittest.TestCase):
    def test_only_prior_blocks_in_decision(self):
        row={'model':'v','site':'qkv','op':'ag','M':1024,'N':4096,'K':4096,'batch':16,
             'warmup':32,'gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
             'cuda_device_max_connections':'unset','blocks':6,'seed':1,
             'block_results':[{'stock_ms':1.,'dx_ms':.8} for _ in range(5)]+[{'stock_ms':1.,'dx_ms':2.}]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'r.json';p.write_text(json.dumps(row))
            r=replay([p])['reports'][0]['rows'][0]
            self.assertEqual(r['past_blocks'],[2,3,4]);self.assertEqual(r['choice'],'dx')
            self.assertLess(r['selected_ratio'],1.)
            with self.assertRaisesRegex(ValueError,'duplicate seed'):replay([p,p])
