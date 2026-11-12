"""One-backend, dependency-aware operator and step-combination predictor.

Times are milliseconds. Transfer/GEMM stage times must be externally measured
or fitted on the *matching* backend, TP, dtype and device. Nothing here claims
that an unmeasured prediction is accurate; callers own calibration/validation.
"""
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence
import math


def pipeline_makespan(producer_ms: Sequence[float], consumer_ms: Sequence[float]) -> float:
    """Two serial resources, each chunk consumes only after its producer finishes.

    For AG, producer=communication, consumer=GEMM; for RS, producer=GEMM,
    consumer=communication. The two lists describe the same ordered chunks.
    Chunk delays include any signal/wait overhead assigned to that stage.
    """
    if not producer_ms or len(producer_ms) != len(consumer_ms):
        raise ValueError('producer and consumer require equal nonempty chunks')
    p = c = 0.0
    for a, b in zip(producer_ms, consumer_ms):
        if not math.isfinite(a) or not math.isfinite(b) or min(a, b) < 0:
            raise ValueError('stage times must be finite and nonnegative')
        p += a
        c = max(c, p) + b
    return c


def fit_linear_residual(pipeline_predictions_ms, measured_overlap_ms):
    """Fit T = nonnegative_scale * T_pipeline + nonnegative fixed overhead.

    This low-capacity calibration is a baseline, not a model of arbitrary
    interference. At least two distinct training points are required.
    """
    x = [float(v) for v in pipeline_predictions_ms]
    y = [float(v) for v in measured_overlap_ms]
    if len(x) < 2 or len(x) != len(y) or any(not math.isfinite(v) or v <= 0 for v in x + y):
        raise ValueError('at least two positive matched pipeline/overlap points required')
    if max(x) - min(x) <= 1e-9:
        raise ValueError('calibration inputs must vary')
    xbar, ybar = sum(x)/len(x), sum(y)/len(y)
    slope = sum((a-xbar)*(b-ybar) for a,b in zip(x,y)) / sum((a-xbar)**2 for a in x)
    intercept = ybar - slope*xbar
    # Exact 2-variable nonnegative least squares via interior/boundaries.
    candidates = [(max(0.0, sum(a*b for a,b in zip(x,y))/sum(a*a for a in x)), 0.0),
                  (0.0, max(0.0, ybar))]
    if slope >= 0 and intercept >= 0:
        candidates.append((slope, intercept))
    slope, intercept = min(candidates, key=lambda p: sum((p[0]*a+p[1]-b)**2 for a,b in zip(x,y)))
    return {'scale': slope, 'fixed_ms': intercept}


@dataclass(frozen=True)
class Operation:
    """One invocation shared between two possible implementations."""
    site: str
    phase: str
    stock_ms: float
    overlap_ms: float
    count: int
    eligible: bool = True


def predict_combinations(backend: str, operations: Sequence[Operation],
                         interaction_ms: Mapping[tuple[str, str], float] | None = None,
                         common_overhead_ms: float = 0.0,
                         pessimism_ms: Mapping[str, float] | None = None,
                         margin_ms: float = 0.0):
    """Enumerate per-site on/off masks for *one* backend.

    Input overlap_ms and overheads are injected by calibration; uncertainty
    is a nonnegative per-invocation upper error in ms. Interactions are whole
    E2E deltas per pair, not multiplied by operation counts. Common overhead is
    applied once if any site is enabled. No measured truth is read at decision
    time. Missing labels must be supplied as eligible=False by the caller.
    """
    if not backend or backend == 'stock' or not operations:
        raise ValueError('one overlap backend and operations are required')
    if not math.isfinite(common_overhead_ms) or not math.isfinite(margin_ms) or margin_ms < 0:
        raise ValueError('invalid common overhead or margin')
    sites = sorted({op.site for op in operations})
    if len(sites) > 16:
        raise ValueError('enumeration limited to 16 logical sites')
    if any(not o.site or not o.phase or not isinstance(o.count, int) or isinstance(o.count, bool)
           or o.count <= 0 or not math.isfinite(o.stock_ms) or not math.isfinite(o.overlap_ms)
           or min(o.stock_ms, o.overlap_ms) <= 0 for o in operations):
        raise ValueError('invalid operation')
    # A site switch controls every occurrence of that named site; do not enable
    # it if any occurrence has unknown availability.
    availability = {s: all(o.eligible for o in operations if o.site == s) for s in sites}
    errors = dict(pessimism_ms or {})
    if set(errors) - set(sites) or any(not math.isfinite(v) or v < 0 for v in errors.values()):
        raise ValueError('invalid per-site uncertainty')
    pairs = {}
    for pair, value in (interaction_ms or {}).items():
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(sites) or not math.isfinite(value):
            raise ValueError('invalid interaction')
        ordered = tuple(sorted(pair))
        if ordered in pairs:
            raise ValueError('duplicate interaction')
        pairs[ordered] = value
    candidates = []
    for bits in product((0, 1), repeat=len(sites)):
        selected = {site for site, bit in zip(sites, bits) if bit}
        if any(not availability[s] for s in selected):
            continue
        raw = sum(o.count * (o.overlap_ms - o.stock_ms) for o in operations if o.site in selected)
        raw += common_overhead_ms if selected else 0.0
        raw += sum(v for pair, v in pairs.items() if set(pair) <= selected)
        penalty = sum(o.count * errors.get(o.site, 0.0) for o in operations if o.site in selected)
        candidates.append({'mask': bits, 'sites': sorted(selected), 'delta_ms': raw,
                           'conservative_delta_ms': raw + penalty})
    best = min(candidates, key=lambda row: (row['conservative_delta_ms'], len(row['sites']), row['mask']))
    if best['conservative_delta_ms'] >= -margin_ms:
        best = next(row for row in candidates if not row['sites'])
    return {'backend': backend, 'sites_order': sites, 'best': best, 'candidates': candidates,
            'status': 'experimental model; requires matched backend/step calibration'}
