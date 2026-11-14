import json
import unittest

from planner.original_a800.local_residual_policy import LocalResidualPolicy


class LocalResidualTest(unittest.TestCase):
    def test_serialization_and_distinct_neighbor_families(self):
        rows=[]
        for family in range(5):
            for m in (1024,4096):
                rows.append({'family':str(family),'op':'rs','M':m,'N':1024+family*256,'K':2048,
                             'full_ms':.1*m/1024,'split_ms':.11*m/1024,
                             'comm_ms':.03*m/1024,'stock_ms':.14*m/1024,'overlap_ms':.12*m/1024})
        policy=LocalResidualPolicy().fit(rows,op='rs')
        query={k:v for k,v in rows[0].items() if k not in ('comm_ms','stock_ms','overlap_ms')}
        result=policy.predict(query)
        self.assertEqual(len({n['family'] for n in result['neighbors']}),3)
        restored=LocalResidualPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(result,restored.predict(query))
        with self.assertRaises(ValueError):restored.predict({**query,'op':'ag_up'})


if __name__=='__main__':unittest.main()
