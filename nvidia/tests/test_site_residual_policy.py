import unittest
import json
from planner.original_a800.site_residual_policy import SiteResidualPolicy,PipelineCorrectedSitePolicy


class SiteResidualTest(unittest.TestCase):
    def test_site_and_shape_only(self):
        rows=[{'site':s,'M':1024*(i+1),'N':4096,'K':4096,'stock_b2b':1.,'dx_b2b':(.8 if s=='fc2' else 1.2)} for i in range(4) for s in ('fc2','o_proj')]
        p=SiteResidualPolicy().fit(rows)
        self.assertEqual(p.predict({'site':'fc2','M':2048,'N':4096,'K':4096})['choice'],'overlap')
        self.assertEqual(p.predict({'site':'unknown','M':2048,'N':4096,'K':4096})['choice'],'stock')
        restored=SiteResidualPolicy.from_dict(json.loads(json.dumps(p.to_dict())))
        self.assertEqual(restored.predict({'site':'fc2','M':2048,'N':4096,'K':4096}),p.predict({'site':'fc2','M':2048,'N':4096,'K':4096}))
        q=PipelineCorrectedSitePolicy(restored)
        self.assertEqual(q.predict({'site':'fc2','M':2048,'N':4096,'K':4096},full_ms=1,split_ms=2)['choice'],'stock')


if __name__=='__main__':unittest.main()
