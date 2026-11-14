import json
import unittest

from planner.original_a800.orthogonal_predictor import ContextRequirement, Sample
from planner.original_a800.predictor_registry import PredictorRegistry
from planner.original_a800.route_local_residual import RouteLocalResidual
from planner.original_a800.route_policy import GroupedSample
from tests.test_predictor_registry import route


class RouteLocalTest(unittest.TestCase):
    def test_serialization_registry_and_route_isolation(self):
        r=route('triton_dist','nonpersistent_ag_all2all_pull','ag')
        rows=[GroupedSample(str(i),Sample(r,(1024,4096+i*256,4096),1,.8)) for i in range(5)]
        model=RouteLocalResidual(r).fit(rows)
        copy=RouteLocalResidual.from_dict(json.loads(json.dumps(model.to_dict())))
        shape=(1024,4096,4096)
        self.assertEqual(model.predict(r,shape),copy.predict(r,shape))
        reg=PredictorRegistry();reg.register(model)
        cap=ContextRequirement(r,shape,'supported','test')
        self.assertEqual(reg.predict(r,shape,capability=cap)['choice'],'triton_dist')
        self.assertEqual(reg.predict(r,shape,capability=ContextRequirement(r,shape,'unknown','test'))['choice'],'stock')
        with self.assertRaises(ValueError):model.predict(route('dx','direct_exchange','ag'),shape)


if __name__=='__main__':unittest.main()
