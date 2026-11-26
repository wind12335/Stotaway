"""Per-route models: ridge predictor, family-holdout local residual, route policy, registry."""
import pytest

from oprs_synthetic import FAMILIES, GAIN, VERSIONS, law, records
from stowaway.oprs.platform import ASYNC_TP, DEO, TRITON_DIST, bw_route
from stowaway.oprs.predictor import SERIAL, ContextRequirement, Route, RouteRegressor
from stowaway.oprs.registry import (PATHS, CachedRouteSelector, PredictorRegistry, UnsupportedRoute, detect)
from stowaway.oprs.route_local_residual import RouteLocalResidual
from stowaway.oprs.route_policy import GroupedSample, RoutePolicy


def test_route_validation():
    route = bw_route(DEO, 'ag', tp=8, topology='BW8', versions=VERSIONS)
    assert route.method == DEO and route.direction == 'ag' and route.actual_path == 'direct_exchange'
    with pytest.raises(ValueError):
        Route(**{**route.__dict__, 'method': SERIAL})
    with pytest.raises(ValueError):
        Route(**{**route.__dict__, 'versions': (('b', '1'), ('a', '1'))})           # unsorted
    with pytest.raises(ValueError):
        Route(**{**route.__dict__, 'direction': 'allreduce'})
    with pytest.raises(ValueError):
        Route(**{**route.__dict__, 'tp': 0})
    with pytest.raises(ValueError):
        bw_route('flux', 'ag', tp=8, topology='BW8', versions=VERSIONS)            # not an integrated method
    with pytest.raises(ValueError):
        bw_route(DEO, 'a2a', tp=8, topology='BW8', versions=VERSIONS)


def test_ridge_is_right_up_to_shrinkage():
    rows = records(DEO, 'rs')
    model = RouteRegressor(rows[0].sample.route).fit([r.sample for r in rows])
    errors = [abs(model.predict(model.route, r.shape) - law(DEO, *r.shape)) for r in rows]
    assert max(errors) < 0.05
    assert model.predict(model.route, (8192, 4096, 1536)) < model.predict(model.route, (32768, 4096, 1536))
    with pytest.raises(ValueError):                             # never transfer across routes
        model.predict(bw_route(ASYNC_TP, 'rs', tp=8, topology='BW8', versions=VERSIONS), (8192, 4096, 1536))
    with pytest.raises(ValueError):
        RouteRegressor(rows[0].sample.route).fit([rows[0].sample])     # at least two samples


def test_a_dimension_no_sample_varies_gets_no_slope():
    """The single-card smoke set has M=8192 only: the fitted model must not invent an M trend (a 12
    standard-deviation extrapolation turned a measured -1.0 into a predicted +0.2 before this law)."""
    rows = records(TRITON_DIST, 'ag', ms=(8192,))
    model = RouteRegressor(rows[0].sample.route).fit([r.sample for r in rows])
    at = lambda m: model.predict(model.route, (m, 1536, 4096))              # noqa: E731
    assert at(4096) == at(8192) == at(32768) == at(1 << 20)                # exactly: no slope in a flat dimension
    assert at(8192) == pytest.approx(law(TRITON_DIST, 8192, 1536, 4096), abs=1e-2)   # and the level is right


def test_local_residual_needs_four_families_and_corrects_the_shrinkage():
    rows = records(DEO, 'ag')
    model = RouteLocalResidual(rows[0].sample.route).fit(rows)
    for r in rows:
        out = model.predict(model.route, r.shape)
        assert out['choice'] == DEO and out['predicted_log_speedup'] == pytest.approx(law(DEO, *r.shape), abs=1e-2)
    slow = RouteLocalResidual(records(ASYNC_TP, 'ag')[0].sample.route).fit(records(ASYNC_TP, 'ag'))
    assert slow.predict(slow.route, (8192, 1536, 4096))['choice'] == SERIAL        # a loss is not chosen
    with pytest.raises(ValueError):
        RouteLocalResidual(rows[0].sample.route).fit(records(DEO, 'ag', families=FAMILIES[:3]))
    with pytest.raises(ValueError):
        model.predict(bw_route(DEO, 'ag', tp=4, topology='BW4', versions=VERSIONS), (8192, 1536, 4096))


def test_model_round_trips_through_json():
    for method in (DEO, ASYNC_TP, TRITON_DIST):
        for op in ('ag', 'rs'):
            rows = records(method, op)
            model = RouteLocalResidual(rows[0].sample.route).fit(rows)
            again = RouteLocalResidual.from_dict(model.to_dict())
            assert again.predict(again.route, (16384, 512, 2048)) == model.predict(model.route, (16384, 512, 2048))
            assert len(model.to_dict()['support']) == len(rows)
    with pytest.raises(ValueError):
        RouteLocalResidual.from_dict({'schema': 'other'})
    with pytest.raises(RuntimeError):
        RouteLocalResidual(rows[0].sample.route).to_dict()


def test_route_policy():
    rows = [GroupedSample(r.family, r.sample) for r in records(DEO, 'rs')]
    policy = RoutePolicy(rows[0].sample.route).fit(rows)
    assert policy.predict(policy.route, (8192, 4096, 1536))['choice'] == DEO
    losing_rows = [GroupedSample(r.family, r.sample) for r in records(ASYNC_TP, 'rs')]
    losing = RoutePolicy(losing_rows[0].sample.route).fit(losing_rows)
    assert losing.predict(losing.route, (8192, 4096, 1536))['choice'] == SERIAL
    again = RoutePolicy.from_dict(policy.to_dict())
    assert again.predict(again.route, (8192, 4096, 1536)) == policy.predict(policy.route, (8192, 4096, 1536))
    with pytest.raises(ValueError):
        RoutePolicy(rows[0].sample.route).fit(rows[:6])


def test_registry_detects_explicit_paths_only():
    key = {'method': DEO, 'actual_path': 'direct_exchange', 'direction': 'ag', 'fallback': False}
    assert detect(key).compute == 'vendor_gemm' and len(PATHS) == 6
    with pytest.raises(UnsupportedRoute):
        detect({**key, 'fallback': True})
    with pytest.raises(UnsupportedRoute):
        detect({**key, 'fallback': None})                     # unknown is not "explicitly not a fallback"
    with pytest.raises(UnsupportedRoute):
        detect({**key, 'actual_path': 'some_other_path'})


def test_registry_dispatches_by_exact_route():
    rows = records(DEO, 'rs')
    model = RouteLocalResidual(rows[0].sample.route).fit(rows)
    registry = PredictorRegistry()
    registry.register(model)
    with pytest.raises(ValueError):
        registry.register(model)                              # one model per route
    good = ContextRequirement(model.route, (8192, 4096, 1536), 'supported', 'test')
    assert registry.predict(model.route, good.shape, capability=good)['choice'] == DEO
    unknown = ContextRequirement(model.route, (8192, 4096, 1536), 'unknown', 'test')
    assert registry.predict(model.route, unknown.shape, capability=unknown) == {
        'choice': SERIAL, 'status': 'shape capability unknown', 'predicted_log_speedup': None}
    other = ContextRequirement(model.route, (8192, 4096, 1536), 'supported', 'test')
    with pytest.raises(ValueError):                           # evidence must match the asked route / shape
        registry.predict(bw_route(DEO, 'ag', tp=8, topology='BW8', versions=VERSIONS), other.shape, capability=other)
    selector = CachedRouteSelector(registry, capacity=2)
    first = selector.predict(model.route, good.shape, capability=good)
    assert selector.predict(model.route, good.shape, capability=good) == first
    assert GAIN[DEO] > 0 and first['predicted_log_speedup'] == pytest.approx(law(DEO, *good.shape), abs=1e-2)
    selector.clear()
