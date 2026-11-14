import unittest
import json
from planner.original_a800.hybrid_route_policy import HybridRoutePolicy
from planner.original_a800.orthogonal_predictor import Sample
from planner.original_a800.route_policy import GroupedSample
from tests.test_predictor_registry import route
from planner.original_a800.predictor_registry import PredictorRegistry,CachedRouteSelector
from planner.original_a800.orthogonal_predictor import ContextRequirement


class HybridTest(unittest.TestCase):
    def test_exact_and_unknown(self):
        r=route('flashoverlap','cutlass_nccl_wave','rs')
        p=HybridRoutePolicy(r).fit([GroupedSample('a',Sample(r,(1024,4096,4096),1,.8))])
        self.assertEqual(p.predict(r,(1024,4096,4096))['choice'],'flashoverlap')
        self.assertEqual(p.predict(r,(2048,4096,4096))['choice'],'stock')
        q=HybridRoutePolicy.from_dict(json.loads(json.dumps(p.to_dict())))
        reg=PredictorRegistry();reg.register(q);shape=(1024,4096,4096)
        self.assertEqual(CachedRouteSelector(reg).predict(r,shape,capability=ContextRequirement(r,shape,'supported','test'))['choice'],'flashoverlap')


if __name__=='__main__':unittest.main()
