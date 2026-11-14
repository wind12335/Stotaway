"""Measured exact-shape quantile guard with a route-local fallback model.

Input arm distributions are not paired: each arm may run in a separate
process. Quantile margins are a heuristic rather than statistical coverage.
"""
import math
import statistics
from dataclasses import asdict

from .route_local_residual import RouteLocalResidual
from .route_policy import GroupedSample
from .orthogonal_predictor import Sample


def quantile(values,q):
    xs=sorted(float(x) for x in values)
    if not xs or any(not math.isfinite(x) or x<=0 for x in xs) or not 0<=q<=1:raise ValueError('invalid samples')
    p=q*(len(xs)-1);i=int(p);j=min(i+1,len(xs)-1)
    return xs[i]+(xs[j]-xs[i])*(p-i)


class GuardedHybridRoute:
    def __init__(self,route,*,margin=.02,lower=.2,upper=.8):
        if not 0<=margin<1 or not 0<=lower<upper<=1:raise ValueError('invalid guard')
        self.route=route;self.margin=margin;self.lower=lower;self.upper=upper;self.exact={};self.fallback=None

    def fit(self,rows):
        """Each row is (family, shape, stock_samples, overlap_samples)."""
        rows=tuple(rows);grouped={};samples=[]
        for family,shape,stock,overlap in rows:
            if len(stock)<4 or len(overlap)<4:raise ValueError('insufficient repeated samples')
            s=Sample(self.route,tuple(shape),statistics.median(stock),statistics.median(overlap))
            samples.append(GroupedSample(family,s))
            item=grouped.setdefault(tuple(shape),{'stock':[],'overlap':[]})
            item['stock'].extend(stock);item['overlap'].extend(overlap)
        self.exact={}
        for shape,item in grouped.items():
            low=quantile(item['stock'],self.lower);high=quantile(item['overlap'],self.upper)
            self.exact[shape]={'stock_lower_ms':low,'overlap_upper_ms':high,
                               'gain':math.log(statistics.median(item['stock'])/statistics.median(item['overlap'])),
                               'choice':self.route.method if high<low*(1-self.margin) else 'stock'}
        if len({r.family for r in samples})>=4:self.fallback=RouteLocalResidual(self.route).fit(samples)
        if not self.exact:raise ValueError('no measured shapes')
        return self

    def choose(self,route,shape,*,gemm_ms=None,comm_ms=None,margin_log=0.):
        if route!=self.route or gemm_ms is not None or comm_ms is not None or margin_log!=0:
            raise ValueError('route mismatch or incompatible features')
        if shape in self.exact:
            row=self.exact[shape]
            return {'choice':row['choice'],'predicted_log_speedup':row['gain'],'status':'guarded exact-shape timing',
                    'estimate_valid':True,'source':'exact',
                    'forbid_step_selection':row['choice']=='stock',**row}
        if self.fallback is not None:
            return {**self.fallback.choose(route,shape),'source':'route_local_fallback'}
        return {'choice':'stock','predicted_log_speedup':None,'status':'no calibrated shape','source':'stock',
                'estimate_valid':False}

    def to_dict(self):
        return {'schema':'stowaway-guarded-hybrid-route-v1','route':asdict(self.route),'margin':self.margin,
                'lower':self.lower,'upper':self.upper,
                'exact':[{'shape':list(k),**v} for k,v in sorted(self.exact.items())],
                'fallback':self.fallback.to_dict() if self.fallback else None}

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='stowaway-guarded-hybrid-route-v1':raise ValueError('invalid schema')
        from .orthogonal_predictor import Route
        fields=dict(data['route']);fields['versions']=tuple(tuple(x) for x in fields['versions']);route=Route(**fields)
        obj=cls(route,margin=data['margin'],lower=data['lower'],upper=data['upper'])
        for row in data['exact']:
            shape=tuple(row['shape']);choice=row['choice'];gain=row['gain'];low=row['stock_lower_ms'];high=row['overlap_upper_ms']
            if len(shape)!=3 or any(not isinstance(x,int) or isinstance(x,bool) or x<=0 for x in shape) or shape in obj.exact:
                raise ValueError('invalid shape')
            if choice not in ('stock',route.method) or not all(math.isfinite(x) for x in (gain,low,high)) or min(low,high)<=0:
                raise ValueError('invalid guarded entry')
            expected=route.method if high<low*(1-obj.margin) else 'stock'
            if choice!=expected:raise ValueError('guarded choice mismatch')
            obj.exact[shape]={'choice':choice,'gain':gain,'stock_lower_ms':low,'overlap_upper_ms':high}
        if not obj.exact:raise ValueError('empty exact inventory')
        if data['fallback'] is not None:
            obj.fallback=RouteLocalResidual.from_dict(data['fallback'])
            if obj.fallback.route!=route:raise ValueError('fallback route mismatch')
        return obj
