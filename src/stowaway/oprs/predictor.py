"""Per-route ridge predictor (Hygon port of nvidia/planner/original_a800/orthogonal_predictor.py).

One calibrated model per complete route (method x compute x transport x memory x synchronization x
actual path x device x topology x TP x dtype x software versions x timing regime x config family).
Features and the ridge mechanism are shared; no fitted coefficient, profile or choice is shared
between routes. The math is the NVIDIA code unchanged; only the choice vocabulary differs:
the no-overlap choice is called ``serial`` here (the handoff's name for NVIDIA's ``stock``).
This module does not claim E2E correctness or performance.
"""
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

SERIAL = 'serial'          # the reference path: RCCL collective, then / around the vendor GEMM
OPS = ('ag', 'rs')


@dataclass(frozen=True)
class Route:
    method: str
    compute: str
    transport: str
    memory: str
    synchronization: str
    direction: str
    actual_path: str
    gpu: str
    topology: str
    tp: int
    dtype: str
    versions: Tuple[Tuple[str, str], ...]
    timing_mode: str
    warmup: Optional[int]  # None means the historical measurement did not record it
    config_family: str

    def __post_init__(self):
        fields = (self.method, self.compute, self.transport, self.memory, self.synchronization,
                  self.direction, self.actual_path, self.gpu, self.topology, self.dtype,
                  self.timing_mode, self.config_family)
        if any(not isinstance(x, str) or not x for x in fields):
            raise ValueError('all route dimensions must be explicit')
        if self.direction not in ('ag', 'rs', 'ar', 'a2a'):
            raise ValueError('unknown collective')
        if self.method == SERIAL:
            raise ValueError('route must describe one overlap method')
        if not isinstance(self.tp, int) or isinstance(self.tp, bool) or self.tp < 1:
            raise ValueError('invalid tensor parallel size')
        if self.warmup is not None and (not isinstance(self.warmup, int) or isinstance(self.warmup, bool) or self.warmup < 1):
            raise ValueError('warmup must be positive or explicitly unknown')
        if (not isinstance(self.versions, tuple) or not self.versions or
            any(not isinstance(p, tuple) or len(p) != 2 or not all(isinstance(x, str) and x for x in p)
                for p in self.versions) or len(dict(self.versions)) != len(self.versions) or
                tuple(sorted(self.versions)) != self.versions):
            raise ValueError('versions must be sorted unique pairs')


@dataclass(frozen=True)
class Sample:
    route: Route
    shape: Tuple[int, int, int]  # M,N,K; semantics must be identical inside route
    stock_ms: float              # median of the serial reference groups (same process / batch)
    candidate_ms: float          # median of the candidate groups
    gemm_ms: Optional[float] = None
    comm_ms: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or len(self.shape) != 3 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.shape
        ):
            raise ValueError('shape must be positive integer MNK')
        for x in (self.stock_ms, self.candidate_ms, self.gemm_ms, self.comm_ms):
            if x is not None and (not math.isfinite(x) or x <= 0):
                raise ValueError('timings must be finite and positive')


@dataclass(frozen=True)
class ContextRequirement:
    """Shape-specific evidence supplied by the backend integration adapter."""
    route: Route
    shape: Tuple[int, int, int]
    support: str  # supported | unsupported | unknown
    evidence: str

    def __post_init__(self):
        if self.support not in ('supported', 'unsupported', 'unknown') or not self.evidence:
            raise ValueError('explicit capability state and evidence required')
        if not isinstance(self.shape, tuple) or len(self.shape) != 3 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.shape
        ):
            raise ValueError('shape must contain positive integer MNK')


def _features(shape, gemm_ms, comm_ms, use_components):
    values = list(shape)
    if use_components:
        if gemm_ms is None or comm_ms is None or min(gemm_ms, comm_ms) <= 0:
            raise ValueError('both calibrated component timings are required')
        values.extend([gemm_ms, comm_ms])
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('features must be finite and positive')
    return np.log(np.asarray(values, dtype=float))


class RouteRegressor:
    """Predict log(serial/candidate), with parameters isolated per exact route."""
    def __init__(self, route: Route, use_components=False, ridge=2.0):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError('ridge must be positive')
        self.route, self.use_components, self.ridge = route, use_components, ridge
        self.parameters = None

    def fit(self, samples):
        samples = list(samples)
        if len(samples) < 2 or any(s.route != self.route for s in samples):
            raise ValueError('at least two samples from the exact route required')
        x = np.stack([_features(s.shape, s.gemm_ms, s.comm_ms, self.use_components) for s in samples])
        y = np.asarray([math.log(s.stock_ms / s.candidate_ms) for s in samples])
        mean, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        z = (x - mean) / sd
        beta = np.linalg.solve(z.T @ z + self.ridge * np.eye(z.shape[1]), z.T @ (y - y.mean()))
        self.parameters = (mean, sd, beta, float(y.mean()))
        return self

    def predict(self, route: Route, shape, gemm_ms=None, comm_ms=None):
        if route != self.route:
            raise ValueError('route mismatch: never transfer transport or compute calibration')
        if self.parameters is None:
            raise RuntimeError('route is uncalibrated')
        mean, sd, beta, intercept = self.parameters
        x = _features(shape, gemm_ms, comm_ms, self.use_components)
        return float(intercept + ((x - mean) / sd) @ beta)

    def choose(self, route, shape, gemm_ms=None, comm_ms=None, margin_log=0.0):
        if not math.isfinite(margin_log) or margin_log < 0:
            raise ValueError('margin must be finite and nonnegative')
        gain = self.predict(route, shape, gemm_ms, comm_ms)
        return {'choice': self.route.method if gain > margin_log else SERIAL,
                'predicted_log_speedup': gain, 'status': 'exploratory operator prediction'}

    def to_dict(self):
        """Export numeric model parameters without executable serialization."""
        if self.parameters is None:
            raise RuntimeError('route is uncalibrated')
        mean, sd, beta, intercept = self.parameters
        return {'schema': 'stowaway-route-regression-v1', 'route': self.route.__dict__,
                'use_components': self.use_components, 'ridge': self.ridge,
                'mean': mean.tolist(), 'sd': sd.tolist(), 'beta': beta.tolist(), 'intercept': intercept}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or data.get('schema') != 'stowaway-route-regression-v1':
            raise ValueError('unsupported model schema')
        try:
            fields = dict(data['route'])
            fields['versions'] = tuple(tuple(p) for p in fields['versions'])
            route = Route(**fields)
            use_components = data['use_components']
            if not isinstance(use_components, bool):
                raise ValueError('use_components must be bool')
            model = cls(route, use_components=use_components, ridge=float(data['ridge']))
            mean, sd, beta = (np.asarray(data[k], dtype=float) for k in ('mean', 'sd', 'beta'))
            intercept = float(data['intercept'])
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('invalid model payload') from exc
        count = 5 if use_components else 3
        if (any(v.shape != (count,) or not np.all(np.isfinite(v)) for v in (mean, sd, beta))
                or np.any(sd <= 0) or not math.isfinite(intercept)):
            raise ValueError('invalid model parameters')
        model.parameters = (mean, sd, beta, intercept)
        return model
