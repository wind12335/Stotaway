"""Grouped conditional policy using optional measured GEMM/communication times."""
import math
import statistics

from .orthogonal_predictor import RouteRegressor


class ConditionalRoutePolicy:
    THRESHOLDS=(-.12,-.06,-.03,0.,.03,.06,.12)

    def __init__(self,route,*,penalty=2.,use_components=True):
        if not math.isfinite(penalty) or penalty<0:raise ValueError('invalid penalty')
        self.route=route;self.penalty=penalty;self.use_components=use_components;self.model=None

    def fit(self,rows):
        rows=tuple(rows)
        if any(r.sample.route!=self.route for r in rows):raise ValueError('mixed route')
        families=sorted({r.family for r in rows})
        if len(families)<3:raise ValueError('insufficient families')
        oof=[]
        for held in families:
            model=RouteRegressor(self.route,use_components=self.use_components).fit([r.sample for r in rows if r.family!=held])
            for r in rows:
                if r.family==held:
                    s=r.sample
                    oof.append((model.predict(self.route,s.shape,s.gemm_ms,s.comm_ms),math.log(s.stock_ms/s.candidate_ms)))
        choices=[('stock',None),('overlap',None)]+[('threshold',t) for t in self.THRESHOLDS]
        def score(choice):
            kind,t=choice
            values=[g if kind=='overlap' or (kind=='threshold' and p>t) else 0. for p,g in oof]
            return statistics.mean(v-self.penalty*max(0.,-v) for v in values)
        self.policy,self.threshold=max(choices,key=lambda c:(score(c),c[0]=='stock'))
        self.model=RouteRegressor(self.route,use_components=self.use_components).fit([r.sample for r in rows])
        self.audit={'samples':len(rows),'families':len(families),'policy':self.policy,'threshold':self.threshold,
                    'oof_utility':score((self.policy,self.threshold)),'component_inputs':self.use_components}
        return self

    def predict(self,route,sample):
        if route!=self.route or self.model is None:raise ValueError('route mismatch or not fitted')
        gain=self.model.predict(route,sample.shape,sample.gemm_ms,sample.comm_ms)
        enabled=self.policy=='overlap' or (self.policy=='threshold' and gain>self.threshold)
        return {'choice':route.method if enabled else 'stock','predicted_log_speedup':gain,'policy':self.policy}
