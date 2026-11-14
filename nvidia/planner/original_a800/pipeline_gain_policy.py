"""Grouped DX policy with explicit overlap/fragmentation physics features.

Requires plan-time GEMM and communication component times. The latter may be
measured directly or estimated by a caller; no test labels enter predict().
"""
import math
import statistics
import numpy as np


def features(row):
    m,n,k,g,c=(float(row[x]) for x in ('M','N','K','G','C'))
    if any(not math.isfinite(x) or x<=0 for x in (m,n,k,g,c)):raise ValueError('invalid features')
    # Smooth pipeline proxy, dimensionless compute/comm balance, plus shape
    # features representing fragmentation and local GEMM efficiency.
    sequential=g+c;ideal=max(g,c)
    hideable=(sequential-ideal)/sequential
    return np.array([math.log(m),math.log(n),math.log(k),math.log(g),math.log(c),
                     math.log(g/c),hideable,math.log((2*m*n*k)/g)],dtype=float)


class PipelineGainPolicy:
    THRESHOLDS=(-.12,-.06,-.03,0.,.03,.06,.12)

    def __init__(self,*,penalty=2.,ridge=10.):
        if not math.isfinite(penalty) or penalty<0 or not math.isfinite(ridge) or ridge<=0:
            raise ValueError('invalid settings')
        self.penalty=penalty;self.ridge=ridge;self.params=None

    def _fit(self,rows):
        x=np.stack([features(r) for r in rows]);y=np.array([math.log(r['stock_b2b']/r['dx_b2b']) for r in rows])
        mu=x.mean(axis=0);sd=x.std(axis=0);sd=np.where(sd<1e-9,1.,sd);z=(x-mu)/sd
        beta=np.linalg.solve(z.T@z+self.ridge*np.eye(z.shape[1]),z.T@(y-y.mean()))
        return mu,sd,beta,float(y.mean())

    @staticmethod
    def _predict(params,row):
        mu,sd,beta,mean=params
        return float(mean+((features(row)-mu)/sd)@beta)

    def fit(self,rows):
        rows=tuple(rows);families=sorted({r['model'] for r in rows})
        if len(families)<3:raise ValueError('insufficient independent families')
        predictions=[]
        for held in families:
            model=self._fit([r for r in rows if r['model']!=held])
            predictions.extend((self._predict(model,r),math.log(r['stock_b2b']/r['dx_b2b'])) for r in rows if r['model']==held)
        candidates=[('stock',None),('overlap',None)]+[('threshold',t) for t in self.THRESHOLDS]
        def score(c):
            kind,t=c;gains=[g if kind=='overlap' or (kind=='threshold' and p>t) else 0. for p,g in predictions]
            return statistics.mean(g-self.penalty*max(0.,-g) for g in gains)
        self.policy,self.threshold=max(candidates,key=lambda c:(score(c),c[0]=='stock'))
        self.params=self._fit(rows)
        self.audit={'families':len(families),'policy':self.policy,'threshold':self.threshold,
                    'oof_utility':score((self.policy,self.threshold))}
        return self

    def predict(self,row):
        if self.params is None:raise RuntimeError('model not fitted')
        gain=self._predict(self.params,row)
        enable=self.policy=='overlap' or (self.policy=='threshold' and gain>self.threshold)
        return {'choice':'overlap' if enable else 'stock','predicted_log_speedup':gain}
