import json
import unittest

from planner.original_a800.reduced_family_ensemble import ReducedFamilyEnsemble


class EnsembleTest(unittest.TestCase):
    def test_roundtrip_and_reject_mixed_operation(self):
        rows=[]
        for family in range(5):
            for m in (1024,4096):
                rows.append({'family':str(family),'op':'rs','M':m,'N':1024+family*256,'K':2048,
                             'full_ms':.1*m/1024,'split_ms':.11*m/1024,
                             'comm_ms':.03*m/1024,'stock_ms':.14*m/1024,'overlap_ms':.12*m/1024})
        model=ReducedFamilyEnsemble().fit(rows,op='rs')
        query={k:v for k,v in rows[0].items() if k not in ('comm_ms','stock_ms','overlap_ms')}
        expected=model.predict(query)
        restored=ReducedFamilyEnsemble.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(expected,restored.predict(query))
        with self.assertRaises(ValueError):restored.predict({**query,'op':'ag_up'})
        with self.assertRaises(ValueError):ReducedFamilyEnsemble().fit(rows[:6],op='rs')


if __name__=='__main__':unittest.main()
