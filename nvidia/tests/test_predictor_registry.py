import unittest
from dataclasses import replace

from planner.original_a800.predictor_registry import detect, bind, PredictorRegistry, UnsupportedRoute, CachedRouteSelector
from planner.original_a800.orthogonal_predictor import ContextRequirement, RouteRegressor, Sample


def route(method, path, direction):
    return bind({'method': method, 'actual_path': path, 'direction': direction, 'fallback': False},
                gpu='A800', topology='nv8', tp=2, dtype='bf16', versions=(('torch', '2.7'),),
                timing_mode='b2b16', warmup=3, config_family='default')


class RegistryTest(unittest.TestCase):
    def test_recognizes_compute_and_communication_independently(self):
        flash = route('flashoverlap', 'cutlass_nccl_wave', 'rs')
        triton = route('triton_dist', 'nonpersistent_ag_all2all_pull', 'ag')
        dx = route('dx', 'direct_exchange', 'ag')
        self.assertEqual((flash.compute, flash.transport), ('cutlass', 'nccl'))
        self.assertEqual((triton.compute, triton.transport, triton.memory),
                         ('triton', 'cuda_peer_copy', 'nvshmem_symmetric'))
        self.assertEqual((dx.compute, dx.transport, dx.memory),
                         ('vendor_gemm', 'cuda_peer_copy', 'nvshmem_symmetric'))

    def test_unknown_and_fallback_do_not_guess(self):
        for metadata in ({'method':'flux','actual_path':'unknown','direction':'ag','fallback':False},
                         {'method':'dx','actual_path':'direct_exchange','direction':'ag'},
                         {'method':'dx','actual_path':'direct_exchange','direction':'ag','fallback':True}):
            with self.assertRaises(UnsupportedRoute):
                detect(metadata)

    def test_scratch_route_cannot_reuse_original_calibration(self):
        original=route('dx','direct_exchange','rs')
        scratch=route('dx','direct_exchange_rs_scratch','rs')
        shape=(1024,4096,4096)
        reg=PredictorRegistry()
        reg.register(RouteRegressor(original).fit([Sample(original,shape,1.,.8),
                     Sample(original,(2048,4096,4096),2.,1.6)]))
        cap=ContextRequirement(scratch,shape,'supported','checked')
        self.assertEqual(reg.predict(scratch,shape,capability=cap)['choice'],'stock')
        with self.assertRaises(UnsupportedRoute):route('dx','direct_exchange_rs_scratch','ag')

    def test_dispatch_and_unregistered_fallback(self):
        r = route('dx', 'direct_exchange', 'ag')
        model = RouteRegressor(r).fit([Sample(r,(1024,4096,4096),1, .8), Sample(r,(2048,4096,4096),2,1.5)])
        registry = PredictorRegistry()
        registry.register(model)
        shape = (1024,4096,4096)
        evidence = ContextRequirement(r, shape, 'supported', 'verified adapter')
        self.assertEqual(registry.predict(r,shape,capability=evidence)['choice'], 'dx')
        other = replace(r, transport='nccl')
        self.assertEqual(registry.predict(other,shape,capability=replace(evidence,route=other))['choice'], 'stock')
        self.assertEqual(registry.predict(r,shape,capability=replace(evidence,support='unknown'))['choice'], 'stock')
        with self.assertRaises(ValueError):
            registry.predict(other,shape,capability=evidence)
        with self.assertRaises(ValueError):
            registry.register(model)

    def test_cached_selector_separates_capabilities_and_features(self):
        r=route('dx','direct_exchange','ag');shape=(1024,4096,4096)
        reg=PredictorRegistry()
        reg.register(RouteRegressor(r).fit([Sample(r,shape,1,.8),Sample(r,(2048,4096,4096),2,1.6)]))
        selector=CachedRouteSelector(reg,capacity=2)
        cap=ContextRequirement(r,shape,'supported','verified')
        self.assertEqual(selector.predict(r,shape,capability=cap)['choice'],'dx')
        self.assertEqual(selector.predict(r,shape,capability=replace(cap,support='unknown'))['choice'],'stock')
        value=selector.predict(r,shape,capability=cap);value['choice']='tampered'
        self.assertEqual(selector.predict(r,shape,capability=cap)['choice'],'dx')
        self.assertEqual(selector.predict(r,shape,capability=cap,margin_log=1.)['choice'],'stock')
        with self.assertRaises(ValueError):selector.predict(r,shape,capability=cap,gemm_ms=float('nan'))

    def test_cached_missing_model_invalidated_on_registration(self):
        r=route('dx','direct_exchange','ag');shape=(1024,4096,4096)
        reg=PredictorRegistry();selector=CachedRouteSelector(reg)
        cap=ContextRequirement(r,shape,'supported','verified')
        self.assertEqual(selector.predict(r,shape,capability=cap)['choice'],'stock')
        reg.register(RouteRegressor(r).fit([Sample(r,shape,1,.8),Sample(r,(2048,4096,4096),2,1.6)]))
        self.assertEqual(selector.predict(r,shape,capability=cap)['choice'],'dx')

    def test_registered_model_is_a_snapshot(self):
        r=route('dx','direct_exchange','ag');shape=(1024,4096,4096)
        model=RouteRegressor(r).fit([Sample(r,shape,1,.8),Sample(r,(2048,4096,4096),2,1.6)])
        reg=PredictorRegistry();reg.register(model)
        model.fit([Sample(r,shape,1,1.5),Sample(r,(2048,4096,4096),2,3.)])
        cap=ContextRequirement(r,shape,'supported','verified')
        self.assertEqual(reg.predict(r,shape,capability=cap)['choice'],'dx')


if __name__ == '__main__':
    unittest.main()
