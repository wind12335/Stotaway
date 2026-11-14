import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.compare_measurement_orders import compare


class MeasurementOrderTest(unittest.TestCase):
    def test_preserves_flip_and_checks_signature(self):
        report={'model':'v','site':'qkv','op':'ag','M':1024,'N':4096,'K':4096,'batch':16,
                'warmup':32,'gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
                'cuda_device_max_connections':'unset','blocks':4,'seed':1,
                'block_results':[{'stock_ms':.9,'dx_ms':1.}]*2+[{'stock_ms':1.2,'dx_ms':1.}]*2}
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.json';a.write_text(json.dumps(report))
            result=compare([a]);self.assertEqual(result['direction_flips'],1)
            self.assertEqual(len(result['rows'][0]['all_speedups']),4)
            report['seed']=2;report['warmup']=3;b=Path(tmp)/'b.json';b.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'contexts'):compare([a,b])


if __name__=='__main__':unittest.main()
