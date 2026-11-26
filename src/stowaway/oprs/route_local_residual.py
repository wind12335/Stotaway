"""Shape-only family-holdout residual correction for a single execution route.

Hygon port of nvidia/planner/original_a800/route_local_residual.py (method unchanged):
  * base  = ridge regression of log(serial/candidate) on (log M, log N, log K), fitted on ALL rows
            of this exact route;
  * support = leave-one-family-out residual of every training row (model fitted on the OTHER
            families), stored per row;
  * predict = base + 0.5 * weighted mean of the <=3 nearest support residuals that come from
            DIFFERENT model families (weight 1/(0.25+d), distance in per-dimension-standardized
            log-shape space, scale = max(std, 0.15)).
"""
import json
import math
from dataclasses import asdict

import numpy as np

from .predictor import SERIAL, Route, RouteRegressor, _features


class RouteLocalResidual:
    def __init__(self, route: Route, *, neighbors: int = 3, shrink: float = .5):
        if not isinstance(neighbors, int) or isinstance(neighbors, bool) or neighbors < 1 or not 0 <= shrink <= 1:
            raise ValueError('invalid settings')
        self.route, self.neighbors, self.shrink = route, neighbors, shrink
        self.model, self.scale, self.support = None, None, []

    def fit(self, rows):
        rows = tuple(rows)
        families = sorted({r.family for r in rows})
        if len(families) < 4 or any(r.sample.route != self.route for r in rows):
            raise ValueError('four same-route families required')
        self.model = RouteRegressor(self.route).fit([r.sample for r in rows])
        features = np.stack([_features(r.sample.shape, None, None, False) for r in rows])
        self.scale = np.maximum(features.std(0), .15)
        self.support = []
        for family in families:
            model = RouteRegressor(self.route).fit([r.sample for r in rows if r.family != family])
            for r in rows:
                if r.family == family:
                    s = r.sample
                    residual = math.log(s.stock_ms / s.candidate_ms) - model.predict(self.route, s.shape)
                    self.support.append((family, _features(s.shape, None, None, False), residual))
        return self

    def predict(self, route, shape):
        if route != self.route or self.model is None:
            raise ValueError('route mismatch or unfitted')
        base = self.model.predict(route, shape)
        q = _features(shape, None, None, False)
        ordered = sorted((float(np.linalg.norm((q - x) / self.scale)), f, r) for f, x, r in self.support)
        seen, near = set(), []
        for d, f, r in ordered:
            if f not in seen:
                near.append((d, f, r))
                seen.add(f)
            if len(near) == self.neighbors:
                break
        w = np.array([1 / (.25 + d) for d, _, _ in near])
        delta = self.shrink * float(w @ np.array([r for _, _, r in near]) / w.sum())
        gain = base + delta
        return {'choice': route.method if gain > 0 else SERIAL, 'predicted_log_speedup': gain,
                'base_log_speedup': base, 'local_delta': delta,
                'status': 'experimental route-local residual', 'estimate_valid': True}

    def choose(self, route, shape, *, gemm_ms=None, comm_ms=None, margin_log=0.):
        if gemm_ms is not None or comm_ms is not None or margin_log != 0:
            raise ValueError('shape-only route model requires its frozen decision rule')
        return self.predict(route, shape)

    def to_dict(self):
        if self.model is None:
            raise RuntimeError('unfitted model')
        return {'schema': 'stowaway-route-local-residual-v1', 'route': asdict(self.route),
                'model': self.model.to_dict(), 'neighbors': self.neighbors, 'shrink': self.shrink,
                'scale': self.scale.tolist(),
                'support': [{'family': f, 'features': x.tolist(), 'residual': r} for f, x, r in self.support]}

    @classmethod
    def from_dict(cls, payload):
        if payload.get('schema') != 'stowaway-route-local-residual-v1':
            raise ValueError('invalid schema')
        model = RouteRegressor.from_dict(payload['model'])
        obj = cls(model.route, neighbors=payload['neighbors'], shrink=payload['shrink'])
        # JSON turns tuples into lists; compare canonical JSON (as the NVIDIA original does).
        if json.dumps(asdict(model.route), sort_keys=True) != json.dumps(payload['route'], sort_keys=True):
            raise ValueError('route/model mismatch')
        obj.model, obj.scale = model, np.asarray(payload['scale'], dtype=float)
        if obj.scale.shape != (3,) or not np.all(np.isfinite(obj.scale)) or np.any(obj.scale <= 0):
            raise ValueError('invalid scale')
        for item in payload['support']:
            family, x, residual = item['family'], np.asarray(item['features'], dtype=float), item['residual']
            if (not isinstance(family, str) or not family or x.shape != (3,)
                    or not np.all(np.isfinite(x)) or not math.isfinite(residual)):
                raise ValueError('invalid support')
            obj.support.append((family, x, float(residual)))
        if len({f for f, _, _ in obj.support}) < 4:
            raise ValueError('insufficient support')
        return obj
