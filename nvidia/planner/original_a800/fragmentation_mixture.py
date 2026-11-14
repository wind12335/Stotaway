"""Route-specific additive component model for unseen physical shape families.

Predicts log(stock/overlap) from the measured independent full/split GEMM and
communication times, plus the exact per-shape stock latency. Requires a
separate stock calibration. Fixed model class and thresholds; no target label
or overlap latency is used in feature construction.
"""
import math
import numpy as np


def features(row):
    values=[row[name] for name in ('M','N','K','full_ms','split_ms','comm_ms','stock_ms')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('positive finite measurements required')
    m,n,k,g,split,comm,stock=map(float,values)
    # How much arithmetic the decomposition adds relative to serial baseline,
    # the serial communication fraction, and how much communication could be
    # overlapped with one GEMM half. Signed split term retains cache effects.
    overhead=(split-g)/stock
    communication=comm/stock
    overlap_space=min(g/2,comm)/stock
    return np.array([math.log(m), math.log(n), math.log(k), math.log(k/n),
                     overhead, communication, overlap_space,
                     overhead*communication],dtype=float)


class FragmentationMixture:
    def __init__(self,*,ridge=10.):
        if not math.isfinite(ridge) or ridge<=0:raise ValueError('invalid ridge')
        self.ridge=ridge;self.params=None

    def fit(self,rows):
        rows=list(rows)
        if len(rows)<4:raise ValueError('insufficient observations')
        x=np.stack([features(row) for row in rows])
        y=np.array([math.log(row['stock_ms']/row['overlap_ms']) for row in rows])
        if not np.all(np.isfinite(y)):raise ValueError('invalid target timings')
        mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-9,1.,scale);z=(x-mean)/scale
        beta=np.linalg.solve(z.T@z+self.ridge*np.eye(z.shape[1]),z.T@(y-y.mean()))
        self.params=(mean,scale,beta,float(y.mean()))
        return self

    def predict(self,row):
        if self.params is None:raise RuntimeError('unfitted model')
        mu,sd,beta,intercept=self.params;gain=float(intercept+((features(row)-mu)/sd)@beta)
        return {'choice':'overlap' if gain>0 else 'stock','predicted_log_speedup':gain}

    def to_dict(self):
        if self.params is None:raise RuntimeError('unfitted model')
        mu,sd,beta,intercept=self.params
        return {'schema':'stowaway-fragmentation-mixture-v1','ridge':self.ridge,'mu':mu.tolist(),
                'sd':sd.tolist(),'beta':beta.tolist(),'intercept':intercept}

    @classmethod
    def from_dict(cls,payload):
        if payload.get('schema')!='stowaway-fragmentation-mixture-v1':raise ValueError('invalid schema')
        model=cls(ridge=payload['ridge']);mu,sd,beta=(np.array(payload[k],dtype=float) for k in ('mu','sd','beta'))
        intercept=float(payload['intercept'])
        if any(v.shape!=(8,) or not np.all(np.isfinite(v)) for v in (mu,sd,beta)) or np.any(sd<=0) or not math.isfinite(intercept):
            raise ValueError('invalid parameters')
        model.params=(mu,sd,beta,intercept)
        return model
