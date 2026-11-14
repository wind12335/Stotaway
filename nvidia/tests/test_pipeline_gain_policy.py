import unittest
from planner.original_a800.pipeline_gain_policy import PipelineGainPolicy


class PipelineGainTest(unittest.TestCase):
    def test_unlabeled_query(self):
        rows=[{'model':str(i),'M':1024*(i+1),'N':4096,'K':4096,'G':1.,'C':.2,'stock_b2b':1.,'dx_b2b':.8} for i in range(4)]
        model=PipelineGainPolicy().fit(rows)
        q={'M':2048,'N':4096,'K':4096,'G':1.,'C':.2}
        self.assertEqual(model.predict(q)['choice'],'overlap')
        with self.assertRaises(ValueError):PipelineGainPolicy().fit(rows[:2])


if __name__=='__main__':unittest.main()
