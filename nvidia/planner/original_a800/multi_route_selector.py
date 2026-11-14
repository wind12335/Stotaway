"""Route-specific grouped model selection with a shared candidate interface.

Candidates are policies over one method; candidates never mix method kernels.
Grouped outer evaluation is the caller's responsibility.
"""
import math
import statistics

from .orthogonal_predictor import RouteRegressor


class MultiRouteSelector:
    """Choose stock, overlap, or a ridge threshold using inner family folds."""
    THRESHOLDS=(-.12,-.06,-.03,0.,.03,.06,.12)

    def __init__(self,route,*,penalty=2.):
        if not math.isfinite(penalty) or penalty<0:raise ValueError('invalid penalty')
        self.route=route;self.penalty=penalty;self.models={};self.choice=None;self.audit=None

    def fit(self,rows):
        rows=tuple(rows)
        if any(r.sample.route!=self.route for r in rows):raise ValueError('mixed routes')
        families=sorted({r.family for r in rows})
        if len(families)<4:raise ValueError('four independent families required')
        data=[]
        for held in families:
            train=[r.sample for r in rows if r.family!=held]
            models={ridge:RouteRegressor(self.route,ridge=ridge).fit(train) for ridge in (2.,10.)}
            for row in rows:
                if row.family==held:
                    s=row.sample
                    data.append((held,{ridge:model.predict(self.route,s.shape) for ridge,model in models.items()},
                                 math.log(s.stock_ms/s.candidate_ms)))
        options=[('stock',None,None),('overlap',None,None)]
        options += [('threshold',ridge,t) for ridge in (2.,10.) for t in self.THRESHOLDS]
        def score(opt):
            kind,ridge,t=opt;groups=[]
            for family in families:
                gains=[actual if kind=='overlap' or (kind=='threshold' and preds[ridge]>t) else 0.
                       for fam,preds,actual in data if fam==family]
                groups.append(statistics.mean(g-self.penalty*max(0.,-g) for g in gains))
            return statistics.mean(groups)
        self.choice=max(options,key=lambda o:(score(o),o[0]=='stock'))
        self.models={ridge:RouteRegressor(self.route,ridge=ridge).fit([r.sample for r in rows]) for ridge in (2.,10.)}
        self.audit={'families':len(families),'samples':len(rows),'selection':self.choice,'oof_utility':score(self.choice),
                    'warning':'exploratory family-weighted model selection'}
        return self

    def predict(self,route,shape):
        if self.choice is None:raise RuntimeError('not fitted')
        if route!=self.route:raise ValueError('route mismatch')
        kind,ridge,t=self.choice
        gain=self.models[ridge or 2.].predict(route,shape)
        enabled=kind=='overlap' or (kind=='threshold' and gain>t)
        return {'choice':route.method if enabled else 'stock','predicted_log_speedup':gain,
                'policy':kind,'ridge':ridge,'threshold':t}
