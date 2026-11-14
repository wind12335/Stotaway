import json
import unittest
from planner.original_a800.piecewise_gain_policy import PiecewiseGainPolicy


class PiecewiseGainTest(unittest.TestCase):
    def test_no_label_in_query_and_serialization(self):
        rows=[{'M':1024*(i+1),'N':4096,'K':4096,'full_ms':1.,'split_ms':1.1,
               'comm_ms':.3,'stock_ms':1.2,'overlap_ms':1.} for i in range(8)]
        model=PiecewiseGainPolicy().fit(rows)
        q={k:v for k,v in rows[0].items() if k!='overlap_ms'}
        self.assertEqual(model.predict(q)['choice'],'overlap')
        restored=PiecewiseGainPolicy.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(restored.predict(q),model.predict(q))
        with self.assertRaises(ValueError):restored.predict({**q,'full_ms':0})

    def test_scale_floor_round_trip_and_legacy(self):
        rows=[{'M':1024*(i+1),'N':4096+i,'K':4096,'full_ms':1.,'split_ms':1.1,
               'comm_ms':.3,'stock_ms':1.2,'overlap_ms':1.} for i in range(8)]
        model=PiecewiseGainPolicy(scale_floor=.05).fit(rows)
        self.assertGreaterEqual(min(model.params[1]), .05)
        restored=PiecewiseGainPolicy.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(restored.scale_floor,.05)
        legacy=model.to_dict();legacy.pop('scale_floor')
        self.assertEqual(PiecewiseGainPolicy.from_dict(legacy).scale_floor,0.)


if __name__=='__main__':unittest.main()
