import json
import unittest
from planner.original_a800.reduced_probe_policy import ReducedProbePolicy


class ReducedProbeTest(unittest.TestCase):
    def test_predict_without_target_baseline_or_label(self):
        rows=[{'family':str(i),'op':'rs','M':1024*(i+1),'N':4096,'K':4096,'full_ms':.2*(i+1),'split_ms':.22*(i+1),
               'comm_ms':.04*(i+1),'stock_ms':.24*(i+1),'overlap_ms':.2*(i+1)} for i in range(8)]
        p=ReducedProbePolicy().fit(rows,op='rs');q={k:v for k,v in rows[0].items() if k not in ('comm_ms','stock_ms','overlap_ms')}
        result=p.predict(q);self.assertEqual(result['choice'],'overlap')
        restored=ReducedProbePolicy.from_dict(json.loads(json.dumps(p.to_dict())))
        self.assertEqual(restored.predict(q),result)
        with self.assertRaises(ValueError):p.predict({**q,'op':'ag_up'})


if __name__=='__main__':unittest.main()
