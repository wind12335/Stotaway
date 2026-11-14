import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.evaluate_state_probe import evaluate


class StateProbeTest(unittest.TestCase):
    def test_holdout_and_context_isolation(self):
        def report(seed):
            return {'classification':'interleaved TP2 direct-exchange operator probe','checks':2,
                    'model':'v','site':'qkv','op':'ag','M':1024,'N':4096,'K':4096,'batch':16,
                    'warmup':32,'gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
                    'cuda_device_max_connections':'unset','blocks':2,'seed':seed,
                    'block_results':[{'stock_ms':1.,'dx_ms':1.2 if seed==3 else .8}]*2,
                    'pre_components':{'gemm':{'median_ms':.1+seed*.001},
                                      'comm':{'median_ms':.2},'wall_ms':25.}}
        with tempfile.TemporaryDirectory() as root:
            paths=[]
            for i in range(4):
                p=Path(root)/f'{i}.json';p.write_text(json.dumps(report(i)));paths.append(p)
            result=evaluate(paths)
            self.assertEqual(result['summary']['stock']['correct'],1)
            self.assertEqual(result['summary']['dx']['slower'],1)
            self.assertEqual(result['summary']['state']['total'],4)
            with self.assertRaisesRegex(ValueError,'duplicate'):evaluate(paths+[paths[0]])
            changed=report(3);changed['M']=2048;paths[3].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'mixed route'):evaluate(paths)


if __name__=='__main__':unittest.main()
