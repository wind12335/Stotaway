"""Route-local nested threshold policy, independent of backend names.

Hygon port of nvidia/planner/original_a800/route_policy.py (method unchanged; ``stock`` -> ``serial``).
Each complete execution Route is trained independently. Grouping is supplied by the caller and
should identify a whole physical shape family. This is an empirical heuristic; it neither promises
80% accuracy nor measures end-to-end. Kept for parity with the NVIDIA study: the OPRS plan is
produced by route_local_residual.RouteLocalResidual, this policy is an ablation baseline.
"""
from dataclasses import asdict, dataclass
import json
import math
import statistics

from .predictor import SERIAL, Route, RouteRegressor, Sample


@dataclass(frozen=True)
class GroupedSample:
    family: str
    sample: Sample

    def __post_init__(self):
        if not isinstance(self.family, str) or not self.family:
            raise ValueError('nonempty family required')


class RoutePolicy:
    THRESHOLDS = (-.12, -.06, -.03, 0., .03, .06, .12)

    def __init__(self, route: Route, *, penalty=2., ridge=2.):
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError('invalid penalty')
        self.route, self.penalty, self.ridge = route, penalty, ridge
        self.model = self.policy = self.threshold = self.audit = None

    def fit(self, rows):
        rows = tuple(rows)
        if any(r.sample.route != self.route for r in rows):
            raise ValueError('mixed execution routes')
        families = sorted({r.family for r in rows})
        if len(families) < 3:
            raise ValueError('at least three independent families required')
        oof = []
        for held in families:
            train = [r.sample for r in rows if r.family != held]
            model = RouteRegressor(self.route, ridge=self.ridge).fit(train)
            for r in rows:
                if r.family == held:
                    s = r.sample
                    oof.append((model.predict(self.route, s.shape), math.log(s.stock_ms / s.candidate_ms)))
        options = [(SERIAL, None), ('overlap', None)] + [('threshold', t) for t in self.THRESHOLDS]

        def score(option):
            kind, threshold = option
            gains = [actual if kind == 'overlap' or (kind == 'threshold' and predicted > threshold) else 0.
                     for predicted, actual in oof]
            return statistics.mean(g - self.penalty * max(0., -g) for g in gains)

        best = max(options, key=lambda option: (score(option), option[0] == SERIAL))
        self.policy, self.threshold = best
        self.model = RouteRegressor(self.route, ridge=self.ridge).fit([r.sample for r in rows])
        self.audit = {'families': len(families), 'samples': len(rows), 'oof_utility': score(best),
                      'penalty': self.penalty, 'policy': self.policy, 'threshold': self.threshold}
        return self

    def predict(self, route, shape, *, gemm_ms=None, comm_ms=None, margin_log=0.):
        if self.model is None:
            raise RuntimeError('policy is not fitted')
        if route != self.route:
            raise ValueError('execution route mismatch')
        if gemm_ms is not None or comm_ms is not None or margin_log != 0:
            raise ValueError('route policy accepts shape-only inputs and its frozen decision rule')
        gain = self.model.predict(route, shape)
        enabled = self.policy == 'overlap' or (self.policy == 'threshold' and gain > self.threshold)
        return {'choice': route.method if enabled else SERIAL, 'predicted_log_speedup': gain,
                'policy': self.policy, 'threshold': self.threshold,
                'status': 'experimental grouped route policy', 'estimate_valid': self.policy == 'threshold'}

    def choose(self, route, shape, **features):
        return self.predict(route, shape, **features)

    def to_dict(self):
        if self.model is None:
            raise RuntimeError('policy is not fitted')
        return {'schema': 'stowaway-route-policy-v1', 'route': asdict(self.route),
                'model': self.model.to_dict(), 'policy': self.policy, 'threshold': self.threshold,
                'penalty': self.penalty, 'ridge': self.ridge, 'audit': self.audit}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get('schema') != 'stowaway-route-policy-v1':
            raise ValueError('unsupported policy schema')
        model = RouteRegressor.from_dict(payload['model'])
        if json.dumps(asdict(model.route), sort_keys=True) != json.dumps(payload['route'], sort_keys=True):
            raise ValueError('route/model mismatch')
        result = cls(model.route, penalty=payload['penalty'], ridge=payload['ridge'])
        policy, threshold = payload['policy'], payload['threshold']
        if policy not in (SERIAL, 'overlap', 'threshold'):
            raise ValueError('invalid policy')
        if ((policy == 'threshold' and (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                                       or not math.isfinite(threshold))) or
                (policy != 'threshold' and threshold is not None)):
            raise ValueError('invalid threshold')
        if result.ridge != model.ridge:
            raise ValueError('model ridge mismatch')
        result.model, result.policy, result.threshold, result.audit = model, policy, threshold, payload['audit']
        return result
