import unittest
import json
from dataclasses import replace
from planner.original_a800.orthogonal_predictor import Sample,ContextRequirement
from planner.original_a800.predictor_registry import PredictorRegistry,CachedRouteSelector
from planner.original_a800.route_policy import GroupedSample, RoutePolicy
from tests.test_predictor_registry import route


class RoutePolicyTest(unittest.TestCase):
    def test_fixed_gain_and_route_isolation(self):
        r=route('flashoverlap','cutlass_nccl_wave','rs')
        rows=[GroupedSample(str(i),Sample(r,(1024*(i+1),4096,4096),1,.8)) for i in range(4)]
        policy=RoutePolicy(r).fit(rows)
        self.assertEqual(policy.predict(r,(2048,4096,4096))['choice'],'flashoverlap')
        with self.assertRaises(ValueError):policy.predict(replace(r,transport='nvshmem'),(2048,4096,4096))
        with self.assertRaises(ValueError):RoutePolicy(r).fit(rows[:2])
        restored=RoutePolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(restored.predict(r,(2048,4096,4096)),policy.predict(r,(2048,4096,4096)))
        registry=PredictorRegistry();registry.register(restored)
        shape=(2048,4096,4096);cap=ContextRequirement(r,shape,'supported','test')
        self.assertEqual(CachedRouteSelector(registry).predict(r,shape,capability=cap)['choice'],'flashoverlap')
        payload=restored.to_dict();payload['threshold']=float('nan');payload['policy']='threshold'
        with self.assertRaises(ValueError):RoutePolicy.from_dict(payload)


if __name__=='__main__':unittest.main()
