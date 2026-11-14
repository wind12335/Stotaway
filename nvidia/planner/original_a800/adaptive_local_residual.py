"""Select residual shrinkage inside disjoint training families per operation."""
import math
import statistics
from .local_residual_policy import LocalResidualPolicy


class AdaptiveLocalResidual:
    CANDIDATES=(0.,.25,.5,.75,1.)

    def __init__(self,*,penalty=2.,scale_floor=.05):
        if not math.isfinite(penalty) or penalty<0:raise ValueError('invalid penalty')
        self.penalty=penalty;self.scale_floor=scale_floor;self.model=None

    def fit(self,rows,*,op):
        rows=tuple(rows);families=sorted({r['family'] for r in rows})
        if len(families)<5 or any(r['op']!=op for r in rows):raise ValueError('five same-operation families required')
        predictions=[]
        for family in families:
            model=LocalResidualPolicy(shrink=.5,scale_floor=self.scale_floor).fit([r for r in rows if r['family']!=family],op=op)
            for row in rows:
                if row['family']!=family:continue
                pred=model.predict(row);actual=math.log(row['stock_ms']/row['overlap_ms'])
                predictions.append((pred['base_log_speedup'],pred['correction']/.5,actual))
        def utility(alpha):
            gains=[actual if base+alpha*delta>0 else 0. for base,delta,actual in predictions]
            return statistics.mean(g-self.penalty*max(0.,-g) for g in gains)
        # Ties favor smaller corrections, preventing needless model complexity.
        scores={alpha:utility(alpha) for alpha in self.CANDIDATES}
        self.alpha=max(self.CANDIDATES,key=lambda a:(scores[a],-a))
        self.model=LocalResidualPolicy(shrink=.5,scale_floor=self.scale_floor).fit(rows,op=op)
        self.op=op;self.audit={'training_families':len(families),'scores':{str(k):v for k,v in scores.items()},'alpha':self.alpha}
        return self

    def predict(self,row):
        if self.model is None or row['op']!=self.op:raise ValueError('unfitted or route mismatch')
        result=self.model.predict(row);gain=result['base_log_speedup']+self.alpha*result['correction']/.5
        return {**result,'choice':'overlap' if gain>0 else 'stock','predicted_log_speedup':gain,
                'fixed_local_log_speedup':result['predicted_log_speedup'],'adaptive_alpha':self.alpha}

    def to_dict(self):
        if self.model is None:raise RuntimeError('unfitted model')
        return {'schema':'stowaway-adaptive-local-residual-v1','op':self.op,'penalty':self.penalty,
                'scale_floor':self.scale_floor,'alpha':self.alpha,'model':self.model.to_dict(),'audit':self.audit}

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='stowaway-adaptive-local-residual-v1':raise ValueError('invalid schema')
        result=cls(penalty=data['penalty'],scale_floor=data['scale_floor']);result.op=data['op'];result.alpha=data['alpha']
        if result.alpha not in cls.CANDIDATES:raise ValueError('invalid alpha')
        result.model=LocalResidualPolicy.from_dict(data['model']);result.audit=data['audit']
        if result.op!=result.model.op or result.model.shrink!=.5:raise ValueError('model mismatch')
        return result
