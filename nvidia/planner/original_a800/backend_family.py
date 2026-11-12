"""Experimental fixed-method planner with explicit execution provenance.

Calibration values are supplied by the caller; this module does not train a
cost model, check GPU kernel support, or establish end-to-end performance.
Unknown interaction terms are never silently treated as measured zeros.
"""
from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping, Optional, Tuple
import math


@dataclass(frozen=True)
class Environment:
    method: str
    family: str
    arch: str
    gpu: str
    topology: str
    world_size: int
    dtype: str
    versions: Tuple[Tuple[str, str], ...]

    def __post_init__(self):
        if self.family not in {'cuda_cutlass', 'triton', 'hybrid'}:
            raise ValueError('unknown backend family')
        if not all((self.method, self.arch, self.gpu, self.topology, self.dtype)) or self.method == 'stock':
            raise ValueError('complete, non-stock environment required')
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int) or self.world_size < 1:
            raise ValueError('world_size must be a positive integer')
        if not isinstance(self.versions, tuple) or not self.versions or any(
            not isinstance(p, tuple) or len(p) != 2 or not all(isinstance(x, str) and x for x in p)
            for p in self.versions
        ) or len(dict(self.versions)) != len(self.versions) or tuple(sorted(self.versions)) != self.versions:
            raise ValueError('versions must be sorted unique (name, version) tuples')


@dataclass(frozen=True)
class Signature:
    environment: Environment
    compute: str
    communication: str
    direction: str
    config: Tuple[Tuple[str, str], ...]
    execution_path: str

    def __post_init__(self):
        if self.direction not in {'ag', 'rs', 'ar', 'a2a'}:
            raise ValueError('unknown direction')
        if not self.compute or not self.communication or not self.execution_path:
            raise ValueError('compute, communication and actual execution path required')
        if not isinstance(self.config, tuple) or any(
            not isinstance(p, tuple) or len(p) != 2 or not all(isinstance(x, str) and x for x in p)
            for p in self.config
        ) or len(dict(self.config)) != len(self.config) or tuple(sorted(self.config)) != self.config:
            raise ValueError('config must be sorted unique (name, value) tuples')


@dataclass(frozen=True)
class MeasurementKey:
    signature: Signature
    phase: str
    shape: Tuple[int, int, int]  # M, N, K, local semantics set by collector
    timing_mode: str  # e.g. b2b16; never silently substitute single-call

    def __post_init__(self):
        if not self.phase or not self.timing_mode or not isinstance(self.shape, tuple) or len(self.shape) != 3 or any(
            not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.shape
        ):
            raise ValueError('phase, positive MNK and timing mode required')


@dataclass(frozen=True)
class OperatorEstimate:
    key: MeasurementKey
    stock_ms: float
    overlap_ms: float
    uncertainty_ms: float
    provenance: str
    supported: bool = True

    def __post_init__(self):
        if not self.provenance or not all(math.isfinite(x) for x in (self.stock_ms, self.overlap_ms, self.uncertainty_ms)):
            raise ValueError('finite estimates and provenance required')
        if self.stock_ms <= 0 or self.overlap_ms <= 0 or self.uncertainty_ms < 0:
            raise ValueError('invalid estimates')
        if not isinstance(self.supported, bool):
            raise ValueError('supported must be bool')


@dataclass(frozen=True)
class Occurrence:
    site: str
    key: MeasurementKey
    count: int

    def __post_init__(self):
        if not self.site or not isinstance(self.count, int) or isinstance(self.count, bool) or self.count <= 0:
            raise ValueError('site and positive count required')


@dataclass(frozen=True)
class StepCalibration:
    environment: Environment
    workload: str
    signatures: Tuple[Signature, ...]
    common_ms: float
    pairs_ms: Mapping[Tuple[str, str], float]
    provenance: str

    def __post_init__(self):
        if not self.workload or not self.provenance or not math.isfinite(self.common_ms):
            raise ValueError('workload, finite common overhead and provenance required')
        if not isinstance(self.signatures, tuple) or not self.signatures or len(set(self.signatures)) != len(self.signatures) or any(
            s.environment != self.environment for s in self.signatures
        ):
            raise ValueError('unique matching implementation signatures required')
        for pair, value in self.pairs_ms.items():
            if not isinstance(pair, tuple) or len(pair) != 2 or pair[0] >= pair[1] or not math.isfinite(value):
                raise ValueError('interaction keys must be sorted distinct site pairs')


def plan(environment: Environment, workload: str, occurrences, estimates,
         step: Optional[StepCalibration], margin_ms: float = 0.0):
    """Select known-safe-to-model masks; unmatched data forces stock at that site.

    A multi-site mask requires an explicit value, possibly zero, for every
    included pair. No step calibration means stock only. Estimates are upper
    error allowances, not statistical guarantees. Timing is predicted, not
    verified. All records must refer to one exact environment.
    """
    occurrences, estimates = tuple(occurrences), tuple(estimates)
    if not workload or not occurrences or not math.isfinite(margin_ms) or margin_ms < 0:
        raise ValueError('workload, occurrences and nonnegative margin required')
    if len({o.site for o in occurrences}) > 16:
        raise ValueError('at most 16 logical sites')
    if any(o.key.signature.environment != environment for o in occurrences):
        raise ValueError('occurrence environment mismatch')
    if any(e.key.signature.environment != environment for e in estimates):
        raise ValueError('estimate environment mismatch')
    if step is not None and (step.environment != environment or step.workload != workload):
        raise ValueError('step calibration environment or workload mismatch')
    if step is not None and set(step.signatures) != {o.key.signature for o in occurrences}:
        raise ValueError('step calibration implementation signatures mismatch')
    lookup = {}
    for e in estimates:
        if e.key in lookup:
            raise ValueError('duplicate estimate key')
        lookup[e.key] = e
    sites = tuple(sorted({o.site for o in occurrences}))
    if step is not None and any(not set(pair) <= set(sites) for pair in step.pairs_ms):
        raise ValueError('interaction references an unknown site')
    missing = sorted({o.site for o in occurrences if o.key not in lookup or not lookup[o.key].supported})
    available = set(sites) - set(missing)
    candidates = []
    uncalibrated_masks = 0
    for bits in product((0, 1), repeat=len(sites)):
        selected = tuple(s for s, b in zip(sites, bits) if b)
        if not set(selected) <= available:
            continue
        if selected and (step is None or any(p not in step.pairs_ms for p in combinations(selected, 2))):
            uncalibrated_masks += 1
            continue
        delta = sum(o.count * (lookup[o.key].overlap_ms - lookup[o.key].stock_ms)
                    for o in occurrences if o.site in selected)
        penalty = sum(o.count * lookup[o.key].uncertainty_ms for o in occurrences if o.site in selected)
        if selected:
            delta += step.common_ms + sum(step.pairs_ms[p] for p in combinations(selected, 2))
        candidates.append({'sites': list(selected), 'delta_ms': delta, 'conservative_delta_ms': delta + penalty})
    best = min(candidates, key=lambda c: (c['conservative_delta_ms'], len(c['sites']), c['sites']))
    if best['conservative_delta_ms'] >= -margin_ms:
        best = next(c for c in candidates if not c['sites'])
    return {'method': environment.method, 'family': environment.family, 'workload': workload,
            'best': best, 'candidates': candidates, 'unavailable_sites': missing,
            'uncalibrated_masks': uncalibrated_masks,
            'status': 'experimental prediction; not a measured end-to-end result'}
