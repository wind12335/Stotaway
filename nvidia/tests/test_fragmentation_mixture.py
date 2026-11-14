import json
import unittest
from planner.original_a800.fragmentation_mixture import FragmentationMixture


class FragmentationMixtureTest(unittest.TestCase):
    def test_unlabelled_prediction_and_roundtrip(self):
        rows=[{'M':1024*(i+1),'N':4096,'K':4096,'full_ms':1.,'split_ms':1.1,'comm_ms':.2,
               'stock_ms':1.2,'overlap_ms':1.} for i in range(5)]
        model=FragmentationMixture().fit(rows)
        q={k:v for k,v in rows[0].items() if k!='overlap_ms'}
        self.assertEqual(model.predict(q)['choice'],'overlap')
        restored=FragmentationMixture.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(restored.predict(q),model.predict(q))
        with self.assertRaises(ValueError):model.predict({**q,'comm_ms':0})


if __name__=='__main__':unittest.main()
