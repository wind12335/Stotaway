"""Backend-isolated relative gain model with grouped residual calibration.

The physical features are optional measured stage estimates; stage measurements
are caller inputs and may cost time to collect. Grouped OOF residual penalties
are empirical heuristics, not bounds or statistical guarantees.
"""
from dataclasses import dataclass
import math

import numpy as np

from .orthogonal_predictor import Route


@dataclass(frozen=True)
class Observation:
    route: Route
    family: str
    shape: tuple
    stock_ms: float
    candidate_ms: float
    gemm_ms: float
    comm_ms: float


@dataclass(frozen=True)
class Query:
    route: Route
    shape: tuple
    gemm_ms: float
    comm_ms: float


def features(o):
    m,n,k=o.shape
    g,c=o.gemm_ms,o.comm_ms
    values=(m,n,k,g,c)
    if any(not math.isfinite(x) or x<=0 for x in values):raise ValueError('positive finite features required')
    # Ratios are dimensionless and shared between backends; fitted parameters
    # remain specific to the complete execution route.
    # This prototype's payload proxy is defined only for the tested 16-bit
    # AG/RS local-shape convention. Other dtypes and collectives require an
    # explicitly validated byte-count convention, not an implicit 2-byte guess.
    if o.route.dtype not in ('bfloat16', 'float16', 'bf16', 'fp16') or o.route.direction not in ('ag', 'rs'):
        raise ValueError('unsupported dtype or collective for this payload proxy')
    payload=m*(k if o.route.direction=='ag' else n)*2
    flops=2*m*n*k
    return np.array([1.0,math.log(m),math.log(n),math.log(k),math.log(g/c),
                     math.log(flops/payload), c/(g+c)],dtype=float)


class DependencyResidual:
    def __init__(self,route,ridge=10.0,quantile=.8,min_families=4):
        if ridge<=0 or not 0<=quantile<=1 or min_families<3:raise ValueError('invalid hyperparameters')
        self.route=route;self.ridge=ridge;self.quantile=quantile;self.min_families=min_families
        self.params=None;self.guard=None

    def _fit(self,rows):
        x=np.stack([features(r) for r in rows]); y=np.array([math.log(r.stock_ms/r.candidate_ms) for r in rows])
        mu=x[:,1:].mean(axis=0);sd=x[:,1:].std(axis=0);sd=np.where(sd<1e-8,1.,sd)
        z=(x[:,1:]-mu)/sd; beta=np.linalg.solve(z.T@z+self.ridge*np.eye(z.shape[1]),z.T@(y-y.mean()))
        return mu,sd,beta,float(y.mean())

    def _predict(self,params,row):
        mu,sd,beta,mean=params
        return float(mean+((features(row)[1:]-mu)/sd)@beta)

    def fit(self,observations):
        rows=list(observations)
        if any(r.route!=self.route or not r.family for r in rows):raise ValueError('route/family mismatch')
        if any(not math.isfinite(x) or x<=0 for r in rows for x in (r.stock_ms,r.candidate_ms)):
            raise ValueError('invalid measured times')
        families=sorted({r.family for r in rows})
        if len(families)<self.min_families:raise ValueError('insufficient independent shape families')
        errors=[]
        for family in families:
            train=[r for r in rows if r.family!=family]
            params=self._fit(train)
            # Aggregate each family first so repeats do not become new families.
            error=max(self._predict(params,r)-math.log(r.stock_ms/r.candidate_ms) for r in rows if r.family==family)
            errors.append(error)
        self.guard=max(0.,float(np.quantile(errors,self.quantile)))
        self.params=self._fit(rows)
        return self

    def predict(self,observation,margin_log=0.):
        if observation.route!=self.route:raise ValueError('route mismatch')
        if self.params is None:raise RuntimeError('uncalibrated model')
        if not math.isfinite(margin_log) or margin_log<0:raise ValueError('invalid margin')
        estimate=self._predict(self.params,observation)
        return {'predicted_log_speedup':estimate,'guard_log':self.guard,
                'choice':self.route.method if estimate-self.guard>margin_log else 'stock'}
