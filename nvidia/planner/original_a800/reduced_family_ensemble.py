"""Family-jackknife disagreement as a route-local decision diagnostic.

Models exclude one training family each; no independence or coverage guarantee
is implied. This class does not observe the target stock or overlap time.
"""
import math
import statistics

from .reduced_probe_policy import ReducedProbePolicy


class ReducedFamilyEnsemble:
    def __init__(self, *, scale_floor=.05):
        if not math.isfinite(scale_floor) or scale_floor < 0:
            raise ValueError('invalid scale floor')
        self.scale_floor = scale_floor
        self.op = None
        self.full = None
        self.folds = ()

    def fit(self, rows, *, op):
        rows = tuple(rows)
        families = sorted({r['family'] for r in rows})
        if len(families) < 4 or any(r['op'] != op for r in rows):
            raise ValueError('one operation and four physical families required')
        self.op = op
        self.full = ReducedProbePolicy(scale_floor=self.scale_floor).fit(rows, op=op)
        self.folds = tuple((family, ReducedProbePolicy(scale_floor=self.scale_floor).fit(
            [r for r in rows if r['family'] != family], op=op)) for family in families)
        return self

    def predict(self, query):
        if self.full is None or query['op'] != self.op:
            raise ValueError('unfitted ensemble or operation mismatch')
        point = self.full.predict(query)['predicted_log_speedup']
        gains = [model.predict(query)['predicted_log_speedup'] for _, model in self.folds]
        return {'full_log_speedup': point, 'median_log_speedup': statistics.median(gains),
                'min_log_speedup': min(gains), 'max_log_speedup': max(gains),
                'positive_votes': sum(g > 0 for g in gains), 'votes': len(gains),
                'family_gains': [{'family': family, 'gain': gain}
                                 for (family, _), gain in zip(self.folds, gains)]}

    def to_dict(self):
        if self.full is None:
            raise RuntimeError('unfitted ensemble')
        return {'schema': 'stowaway-reduced-family-ensemble-v1', 'op': self.op,
                'scale_floor': self.scale_floor, 'full': self.full.to_dict(),
                'folds': [{'family': family, 'model': model.to_dict()} for family, model in self.folds]}

    @classmethod
    def from_dict(cls, payload):
        if payload.get('schema') != 'stowaway-reduced-family-ensemble-v1':
            raise ValueError('invalid ensemble schema')
        instance = cls(scale_floor=payload['scale_floor'])
        instance.op = payload['op']
        instance.full = ReducedProbePolicy.from_dict(payload['full'])
        instance.folds = tuple((item['family'], ReducedProbePolicy.from_dict(item['model']))
                               for item in payload['folds'])
        if (len(instance.folds) < 4 or instance.full.op != instance.op or
                len({family for family, _ in instance.folds}) != len(instance.folds) or
                any(not isinstance(family, str) or not family or model.op != instance.op
                    for family, model in instance.folds)):
            raise ValueError('invalid ensemble family inventory')
        return instance
