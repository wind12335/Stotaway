"""Experimental within-method combination selection from measured E2E deltas.

Every measured switch set is eligible. Unmeasured combinations are disabled
by default: a linear model cannot prove its own omitted interactions are zero.
Opt-in extrapolation only rejects combinations not identifiable under the
specified interaction model; it is not proof of E2E performance.
"""
from dataclasses import dataclass
from itertools import combinations, product
import math
import numpy as np


@dataclass(frozen=True)
class Observation:
    """Measured candidate minus matched baseline in ms, with exact switches.

    context identifies the *same* platform, model, workload, implementation,
    batch and measurement window. It is supplied by the collector.
    """
    context: str
    switches: tuple
    delta_ms: float


class CombinationPlanner:
    def __init__(self, backend, sites, operator_delta_ms, context,
                 interaction_order=1, min_gain_ms=0.0):
        if not backend or not context or not sites or len(set(sites)) != len(sites):
            raise ValueError('backend, context and unique sites are required')
        if backend == 'stock':
            raise ValueError('backend must be a single overlap method')
        if interaction_order not in (1, 2):
            raise ValueError('interaction_order must be 1 or 2')
        if not math.isfinite(min_gain_ms) or min_gain_ms < 0:
            raise ValueError('min_gain_ms must be finite and non-negative')
        if set(operator_delta_ms) != set(sites):
            raise ValueError('one operator contribution is required per site')
        values = [float(operator_delta_ms[s]) for s in sites]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('operator contributions must be finite')
        self.backend, self.sites, self.context = backend, tuple(sites), context
        self.op_delta = np.asarray(values)
        self.min_gain_ms = min_gain_ms
        self.terms = [(i,) for i in range(len(sites))]
        if interaction_order == 2:
            self.terms += list(combinations(range(len(sites)), 2))
        self.params, self.row_basis, self.observations = None, None, []

    def _switches(self, switches):
        switches = tuple(switches)
        if len(switches) != len(self.sites) or any(x not in (0, 1) for x in switches):
            raise ValueError('switches must contain one 0/1 per site')
        return switches

    def _features(self, switches):
        return np.asarray([math.prod(switches[i] for i in term) for term in self.terms], dtype=float)

    def fit(self, observations):
        observations = list(observations)
        if not observations:
            raise ValueError('E2E observations required; operator-only is not E2E calibration')
        x, y = [], []
        measured_by_switch = {}
        for obs in observations:
            if obs.context != self.context:
                raise ValueError('measurement context does not match')
            switches = self._switches(obs.switches)
            if not math.isfinite(obs.delta_ms):
                raise ValueError('non-finite E2E measurement')
            if not any(switches) and abs(obs.delta_ms) > 1e-9:
                raise ValueError('all-stock baseline must have zero delta')
            x.append(self._features(switches))
            y.append(obs.delta_ms - float(self.op_delta @ switches))
            measured_by_switch.setdefault(switches, []).append(float(obs.delta_ms))
        x, y = np.asarray(x), np.asarray(y)
        self.params = np.linalg.lstsq(x, y, rcond=None)[0]
        _, singular, vt = np.linalg.svd(x, full_matrices=False)
        tol = max(x.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0)
        self.row_basis = vt[singular > tol]
        self.observations = observations
        self.measured_deltas = {switches: float(np.median(values))
                                for switches, values in measured_by_switch.items()}
        self.measured_deltas[(0,) * len(self.sites)] = 0.0
        return self

    def estimate(self, switches):
        if self.params is None:
            raise RuntimeError('E2E model is uncalibrated')
        switches = self._switches(switches)
        x = self._features(switches)
        projected = self.row_basis.T @ (self.row_basis @ x)
        if np.linalg.norm(x - projected) > 1e-8 * max(1.0, np.linalg.norm(x)):
            return None
        return float(self.op_delta @ switches + x @ self.params)

    def select(self, allowed=None, allow_model_extrapolation=False):
        if self.params is None:
            raise RuntimeError('E2E model is uncalibrated')
        allowed = set(self.sites if allowed is None else allowed)
        if not allowed <= set(self.sites):
            raise ValueError('unknown allowed site')
        options = []
        measured = {self._switches(o.switches) for o in self.observations}
        measured.add((0,) * len(self.sites))
        for switches in product((0, 1), repeat=len(self.sites)):
            if any(v and s not in allowed for s, v in zip(self.sites, switches)):
                continue
            if not allow_model_extrapolation and switches not in measured:
                continue
            # Compare directly measured combinations by their measurements;
            # a lower-order fit can be misspecified and smooth away a real
            # interaction. Only unmeasured combinations use extrapolation.
            delta = self.measured_deltas.get(switches)
            if delta is None and allow_model_extrapolation:
                delta = self.estimate(switches)
            if delta is not None:
                options.append((delta, switches))
        best_delta, best_switches = min(options, key=lambda v: (v[0], sum(v[1]), v[1]))
        if best_delta >= -self.min_gain_ms:
            best_delta, best_switches = 0.0, (0,) * len(self.sites)
        return {'backend': self.backend, 'context': self.context,
                'choices': {s: (self.backend if enabled else 'stock')
                            for s, enabled in zip(self.sites, best_switches)},
                'estimated_delta_ms': best_delta, 'estimable_options': len(options),
                'total_eligible_options': 2 ** len(allowed),
                'extrapolates': allow_model_extrapolation,
                'assumption': 'specified interaction order; no unidentified combinations'}
