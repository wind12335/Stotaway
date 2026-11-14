import unittest
from dataclasses import replace

from planner.original_a800.dependency_residual import Observation,Query,DependencyResidual
from tests.test_predictor_registry import route


class TestDependencyResidual(unittest.TestCase):
    def test_group_calibration_and_route_isolation(self):
        r=route('dx','direct_exchange','ag')
        samples=[Observation(r,str(i),(1024*(i+1),4096,4096),1.,.8,.6,.2) for i in range(5)]
        model=DependencyResidual(r).fit(samples)
        q=Query(r,(4096,4096,4096),.6,.2)
        self.assertEqual(model.predict(q)['choice'],'dx')
        with self.assertRaises(ValueError):model.predict(replace(q,route=replace(r,transport='nccl')))
        with self.assertRaises(ValueError):DependencyResidual(r).fit(samples[:3])

    def test_requires_positive_features(self):
        r=route('dx','direct_exchange','ag')
        with self.assertRaises(ValueError):
            DependencyResidual(r).fit([Observation(r,str(i),(1024,4096,4096),1.,.8,0,.2) for i in range(4)])

    def test_payload_proxy_rejects_unsupported_dtype_or_collective(self):
        r=route('dx','direct_exchange','ag')
        for changed in (replace(r,dtype='float32'),replace(r,direction='ar')):
            with self.assertRaises(ValueError):
                DependencyResidual(changed).fit([Observation(changed,str(i),(1024,4096,4096),1.,.8,.6,.2) for i in range(4)])


if __name__=='__main__':unittest.main()
