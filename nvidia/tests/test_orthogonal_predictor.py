import unittest
import json
from dataclasses import replace

from planner.original_a800.orthogonal_predictor import Route, Sample, RouteRegressor


class OrthogonalRouteTest(unittest.TestCase):
    def setUp(self):
        self.route = Route('flashoverlap', 'cutlass', 'nccl', 'cuda', 'wave_signal', 'rs',
                           'cutlass_nccl_wave', 'a800', 'nvlink', 2, 'float16',
                           (('cuda', '12.6'), ('torch', '2.7')), 'b2b16', 32, 'predictive_search')
        self.samples = [Sample(self.route, (1024, 4096, 4096), 1., 1.1),
                        Sample(self.route, (4096, 4096, 4096), 2., 1.7)]

    def test_route_isolation(self):
        model = RouteRegressor(self.route).fit(self.samples)
        for other in (replace(self.route, transport='nvshmem'), replace(self.route, compute='triton'),
                      replace(self.route, memory='nvshmem_symmetric'), replace(self.route, warmup=4),
                      replace(self.route, actual_path='fallback_nccl')):
            with self.assertRaises(ValueError):
                model.predict(other, (2048, 4096, 4096))
            with self.assertRaises(ValueError):
                model.fit(self.samples + [replace(self.samples[0], route=other)])

    def test_unknown_warmup_does_not_match_known_warmup(self):
        unknown = replace(self.route, warmup=None)
        samples = [replace(s, route=unknown) for s in self.samples]
        model = RouteRegressor(unknown).fit(samples)
        restored = RouteRegressor.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertIsNone(restored.route.warmup)
        with self.assertRaises(ValueError):
            restored.predict(self.route, (1024, 4096, 4096))

    def test_pipeline_can_use_measured_components_or_missing(self):
        model = RouteRegressor(self.route).fit(self.samples)
        self.assertEqual(model.choose(self.route, (4096, 4096, 4096))['choice'], 'flashoverlap')
        component_model = RouteRegressor(self.route, use_components=True)
        with self.assertRaises(ValueError):
            component_model.fit(self.samples)
        comp = [replace(s, gemm_ms=.5, comm_ms=.2) for s in self.samples]
        component_model.fit(comp)
        with self.assertRaises(ValueError):
            component_model.predict(self.route, (1024, 4096, 4096))

    def test_invalid_routes(self):
        with self.assertRaises(ValueError):
            replace(self.route, transport='')
        with self.assertRaises(ValueError):
            replace(self.route, warmup=0)

    def test_json_round_trip_and_invalid_parameters(self):
        model = RouteRegressor(self.route).fit(self.samples)
        saved = json.loads(json.dumps(model.to_dict()))
        restored = RouteRegressor.from_dict(saved)
        shape = (2048,4096,4096)
        self.assertAlmostEqual(model.predict(self.route,shape),restored.predict(self.route,shape))
        saved['sd'][0] = 0
        with self.assertRaises(ValueError):
            RouteRegressor.from_dict(saved)


if __name__ == '__main__':
    unittest.main()
