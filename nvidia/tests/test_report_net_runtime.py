import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.report_net_runtime import report

class NetRuntimeReportTest(unittest.TestCase):
    def test_block_pairing_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);design={'shapes':[{'model':'shape_vehicle_h1','site':'fc2','M':1024}],
              'blocks_per_shape':2,'arms':['stock','adaptive'],'order':['stock-adaptive','adaptive-stock'],'calls':16}
            d=root/'design.json';d.write_text(json.dumps(design))
            base={'classification':'x','model':'shape_vehicle_h1','site':'fc2','op':'rs','M':1024,'N':1,'K':2,
                  'calls':16,'batch':16,'warmup':32,'probe_blocks':4,'gpu':'gpu','torch':'t','triton':'tr',
                  'cuda_build':'c','cuda_device_max_connections':'unset','checks':2,
                  'probe_max_rank_wall_ms':0.,'chosen_run_max_rank_wall_ms':100.}
            for i in (1,2):
                for arm in design['arms']:
                    choice='stock' if arm=='stock' else 'dx'
                    row={**base,'arm':arm,'choice':choice,'rank_choices':[choice,choice],
                         'total_max_rank_wall_ms':100. if arm=='stock' else 90.}
                    (root/f'tp2_net_h1_fc2_1024_{arm}_b{i}_20261113.json').write_text(json.dumps(row))
            result=report(d,root);self.assertAlmostEqual(result['groups'][0]['geomean_net_speedup'],100/90)
            target=root/'tp2_net_h1_fc2_1024_adaptive_b2_20261113.json';v=json.loads(target.read_text());v['calls']=32;target.write_text(json.dumps(v))
            with self.assertRaises(ValueError):report(d,root)
