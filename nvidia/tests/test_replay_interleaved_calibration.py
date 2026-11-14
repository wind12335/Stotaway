import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.replay_interleaved_calibration import replay


class SequentialReplayTest(unittest.TestCase):
    def test_no_future_information_and_signature_guard(self):
        def report(seed, dx):
            rows=[{'stock_ms':1.,'dx_ms':dx} for _ in range(4)]
            return {'classification':'interleaved TP2 direct-exchange operator probe','checks':2,
                    'model':'v','site':'qkv','op':'ag','M':1024,'N':4096,'K':4096,'batch':16,
                    'warmup':32,'gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
                    'cuda_device_max_connections':'unset','blocks':4,'seed':seed,
                    'block_results':rows,'point_wall_ms':10.,'collector_wall_ms':20.}
        with tempfile.TemporaryDirectory() as root:
            paths=[]
            for i,dx in enumerate((.8,.8,1.2)):
                p=Path(root)/f'{i}.json';p.write_text(json.dumps(report(i,dx)));paths.append(p)
            result=replay(paths,horizon=1000)
            self.assertEqual(result['decisions'][0]['choice'],'optimized')
            self.assertEqual(result['summary']['slower_runs'],1)
            self.assertEqual(result['decisions'][0]['prior_seeds'],[0,1])
            with self.assertRaisesRegex(ValueError,'duplicate'):replay(paths[:2]+paths[:1])
            bad=report(2,1.2);bad['warmup']=3;paths[2].write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'signature mismatch'):replay(paths)


if __name__=='__main__':unittest.main()
