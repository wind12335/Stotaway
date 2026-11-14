import json
import tempfile
import unittest
from pathlib import Path
from hashlib import sha256
from planner.original_a800.evaluate_neighbor_guard import evaluate


class NeighborGuardTest(unittest.TestCase):
    def test_guard_and_evaluation_provenance(self):
        pred={'predicted_log_speedup':.03,'neighbors':[{'family':'a','residual':-.1},
              {'family':'b','residual':0.},{'family':'c','residual':.1}]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dev=root/'dev.json';dec=root/'dec.json';ev=root/'ev.json'
            dev.write_text(json.dumps({'rows':[{'prediction':pred,'actual_speedup':.9}]}))
            dec.write_text(json.dumps({'predictions':[{'identity':['shape'], 'prediction':pred}]}))
            ev.write_text(json.dumps({'frozen_sha256':sha256(dec.read_bytes()).hexdigest(),
                                      'rows':[{'identity':['shape'],'actual_speedup':.9}]}))
            result=evaluate(dev,dec,[ev])
            self.assertEqual(result['summary']['evaluation_1']['0.0']['slower'],1)
            self.assertEqual(result['summary']['evaluation_1']['0.5']['slower'],0)
            data=json.loads(ev.read_text());data['frozen_sha256']='bad';ev.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'digest mismatch'):evaluate(dev,dec,[ev])


if __name__=='__main__':unittest.main()
