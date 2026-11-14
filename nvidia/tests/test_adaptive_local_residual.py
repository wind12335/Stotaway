import json
import unittest
from planner.original_a800.adaptive_local_residual import AdaptiveLocalResidual


class AdaptiveLocalTest(unittest.TestCase):
    def test_fit_and_roundtrip(self):
        rows=[]
        for i in range(6):
            for m in (1024,4096):
                rows.append({'family':str(i),'op':'rs','M':m,'N':1024+i*256,'K':2048,
                             'full_ms':.1*m/1024,'split_ms':.11*m/1024,'comm_ms':.03*m/1024,
                             'stock_ms':.14*m/1024,'overlap_ms':.12*m/1024})
        model=AdaptiveLocalResidual().fit(rows,op='rs')
        q={k:v for k,v in rows[0].items() if k not in ('comm_ms','stock_ms','overlap_ms')}
        restored=AdaptiveLocalResidual.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(model.predict(q),restored.predict(q))
        self.assertIn(model.alpha,model.CANDIDATES)
        with self.assertRaises(ValueError):AdaptiveLocalResidual().fit(rows[:8],op='rs')


if __name__=='__main__':unittest.main()
