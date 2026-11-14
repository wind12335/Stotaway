"""Piecewise ridge model for component ratios and small/large M regimes."""
import math
import numpy as np


def features(row):
    vals=[row[k] for k in ('M','N','K','full_ms','split_ms','comm_ms','stock_ms')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in vals):
        raise ValueError('positive finite inputs required')
    m,n,k,g,split,c,s=map(float,vals)
    ratio=g/c;fragment=split/g
    # Fixed hinge locations encode distinct latency regimes, not tuned on any
    # target dataset. Ridge shrinks redundant and interaction coefficients.
    logratio=math.log(ratio);frag=math.log(fragment);small=float(m<=2048)
    return np.array([math.log(m),math.log(n),math.log(k),math.log(k/n),
                     logratio,frag,small,small*logratio,small*frag,
                     max(0.,math.log(2.)-logratio),max(0.,frag),
                     c/s,(split-g)/s,min(g/2,c)/s],dtype=float)


class PiecewiseGainPolicy:
    def __init__(self,*,ridge=12.,scale_floor=0.):
        if not math.isfinite(ridge) or ridge<=0:raise ValueError('invalid ridge')
        if not math.isfinite(scale_floor) or scale_floor<0:raise ValueError('invalid scale floor')
        self.ridge=ridge;self.scale_floor=scale_floor;self.params=None

    def fit(self,rows):
        rows=list(rows)
        if len(rows)<6:raise ValueError('insufficient training observations')
        x=np.stack([features(r) for r in rows]);y=np.array([math.log(r['stock_ms']/r['overlap_ms']) for r in rows])
        if not np.all(np.isfinite(y)):raise ValueError('invalid labels')
        mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-9,1.,np.maximum(sd,self.scale_floor));z=(x-mu)/sd
        beta=np.linalg.solve(z.T@z+self.ridge*np.eye(z.shape[1]),z.T@(y-y.mean()))
        self.params=(mu,sd,beta,float(y.mean()));return self

    def predict(self,row):
        if self.params is None:raise RuntimeError('not fitted')
        mu,sd,beta,mean=self.params;gain=float(mean+((features(row)-mu)/sd)@beta)
        return {'choice':'overlap' if gain>0 else 'stock','predicted_log_speedup':gain}

    def to_dict(self):
        if self.params is None:raise RuntimeError('not fitted')
        mu,sd,beta,mean=self.params
        return {'schema':'stowaway-piecewise-gain-v1','ridge':self.ridge,'scale_floor':self.scale_floor,'mu':mu.tolist(),'sd':sd.tolist(),
                'beta':beta.tolist(),'mean':mean}

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='stowaway-piecewise-gain-v1':raise ValueError('invalid schema')
        obj=cls(ridge=data['ridge'],scale_floor=data.get('scale_floor',0.));mu,sd,beta=(np.array(data[k],dtype=float) for k in ('mu','sd','beta'));mean=float(data['mean'])
        if any(x.shape!=(14,) or not np.all(np.isfinite(x)) for x in (mu,sd,beta)) or np.any(sd<=0) or not math.isfinite(mean):
            raise ValueError('invalid parameters')
        obj.params=(mu,sd,beta,mean);return obj
