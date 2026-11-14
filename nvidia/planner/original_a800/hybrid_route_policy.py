"""Exact-shape calibration with route-local grouped prediction fallback."""
import math
from .route_policy import RoutePolicy
from .orthogonal_predictor import Route
from dataclasses import asdict


class HybridRoutePolicy:
    """Use a same-shape gain when observed; otherwise use grouped policy.

    This initial nearest-exact rule intentionally has no tuned margin. It does
    not claim statistical protection against batch drift.
    """
    def __init__(self,route):
        self.route=route;self.exact={};self.fallback=None

    def fit(self,rows):
        rows=tuple(rows)
        if any(r.sample.route!=self.route for r in rows):raise ValueError('mixed routes')
        grouped={}
        for r in rows:
            s=r.sample
            grouped.setdefault(s.shape,[]).append(math.log(s.stock_ms/s.candidate_ms))
        import statistics
        self.exact={shape:statistics.median(gains) for shape,gains in grouped.items()}
        try:self.fallback=RoutePolicy(self.route).fit(rows)
        except ValueError:self.fallback=None
        return self

    def predict(self,route,shape):
        if route!=self.route:raise ValueError('route mismatch')
        if shape in self.exact:
            gain=self.exact[shape]
            return {'choice':route.method if gain>0 else 'stock','source':'exact_shape','log_speedup':gain}
        if self.fallback:
            return {**self.fallback.predict(route,shape),'source':'grouped_model'}
        return {'choice':'stock','source':'uncalibrated'}

    def choose(self,route,shape,*,gemm_ms=None,comm_ms=None,margin_log=0.):
        if gemm_ms is not None or comm_ms is not None or margin_log!=0:
            raise ValueError('hybrid policy requires frozen shape-only rule')
        result=self.predict(route,shape)
        if 'predicted_log_speedup' not in result:
            result['predicted_log_speedup']=result.get('log_speedup')
        result['estimate_valid']=result['source']=='exact_shape' or result.get('policy')=='threshold'
        result['status']='experimental hybrid route policy'
        return result

    def to_dict(self):
        return {'schema':'stowaway-hybrid-route-v1','route':asdict(self.route),
                'exact':[{'shape':list(shape),'log_speedup':gain} for shape,gain in sorted(self.exact.items())],
                'fallback':self.fallback.to_dict() if self.fallback else None}

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or data.get('schema')!='stowaway-hybrid-route-v1':raise ValueError('invalid schema')
        fields=dict(data['route']);fields['versions']=tuple(tuple(p) for p in fields['versions']);route=Route(**fields)
        result=cls(route)
        for row in data['exact']:
            shape=tuple(row['shape']);gain=row['log_speedup']
            if len(shape)!=3 or any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in shape):
                raise ValueError('invalid shape')
            if not isinstance(gain,(int,float)) or isinstance(gain,bool) or not math.isfinite(gain) or shape in result.exact:
                raise ValueError('invalid exact measurement')
            result.exact[shape]=float(gain)
        if data['fallback'] is not None:
            result.fallback=RoutePolicy.from_dict(data['fallback'])
            if result.fallback.route!=route:raise ValueError('fallback route mismatch')
        return result
