import unittest
from dataclasses import replace

from planner.original_a800.orthogonal_predictor import ContextRequirement, RouteRegressor, Sample
from planner.original_a800.predictor_registry import PredictorRegistry
from planner.original_a800.unified_step import Callsite, choose_step
from planner.original_a800.guarded_hybrid_route import GuardedHybridRoute
from tests.test_predictor_registry import route


class UnifiedStepTest(unittest.TestCase):
    def test_exact_shape_guard_cannot_be_overridden_by_step_interaction(self):
        r=route('flashoverlap','cutlass_nccl_wave','rs')
        shape=(1024,4096,4096)
        model=GuardedHybridRoute(r).fit([('f',shape,[1.]*8,[.99]*8)])
        registry=PredictorRegistry();registry.register(model)
        call=Callsite('proj','fwd',r,shape,1,1.,ContextRequirement(r,shape,'supported','test'),0)
        result=choose_step(registry,[call],'step',common_ms=-1,pairs_ms={},calibration_provenance='test')
        self.assertEqual(result['best']['sites'],[])
        self.assertEqual(result['unavailable_sites'],['proj'])
        self.assertTrue(result['diagnostics'][0]['forbid_step_selection'])

    def test_route_model_integrates_with_known_step_costs(self):
        ag=route('dx','direct_exchange','ag');rs=route('dx','direct_exchange','rs')
        registry=PredictorRegistry()
        for r in (ag,rs):
            registry.register(RouteRegressor(r).fit([Sample(r,(1024,4096,4096),1,.8),Sample(r,(2048,4096,4096),2,1.6)]))
        shape=(1024,4096,4096)
        calls=[Callsite('qkv','fwd',ag,shape,2,1.,ContextRequirement(ag,shape,'supported','test'),0),
               Callsite('proj','fwd',rs,shape,1,1.,ContextRequirement(rs,shape,'supported','test'),0)]
        self.assertEqual(choose_step(registry,calls,'step')['best']['sites'],[])
        single=choose_step(registry,calls,'step',common_ms=0,pairs_ms={},calibration_provenance='test')
        self.assertEqual(single['best']['sites'],['qkv'])
        both=choose_step(registry,calls,'step',common_ms=0,pairs_ms={('proj','qkv'):0},calibration_provenance='test')
        self.assertEqual(both['best']['sites'],['proj','qkv'])
        corrected=choose_step(registry,calls,'step',common_ms=0,pairs_ms={('proj','qkv'):0},
                              site_residual_ms={'qkv':1.0},calibration_provenance='test')
        self.assertEqual(corrected['best']['sites'],['proj'])
        unknown=replace(calls[0],capability=ContextRequirement(ag,shape,'unknown','test'))
        self.assertEqual(choose_step(registry,[unknown,calls[1]],'step',common_ms=0,pairs_ms={},
                                     calibration_provenance='test')['best']['sites'],['proj'])
        other=replace(rs,method='flux')
        with self.assertRaises(ValueError):
            choose_step(registry,[calls[0],replace(calls[1],route=other)],'step')



if __name__=='__main__':unittest.main()
