"""Hierarchical site-aware gain model with shared and site-specific effects."""
import math
import numpy as np


class SiteResidualPolicy:
    """Ridge model with regularized interaction terms; shape-only at inference.

    The caller must fit one exact execution route and preserve physical-family
    grouping in all evaluation folds. Sites must be known deployment semantics.
    """
    def __init__(self,*,ridge=8.):
        if not math.isfinite(ridge) or ridge<=0:raise ValueError('invalid ridge')
        self.ridge=ridge;self.sites=None;self.params=None

    @staticmethod
    def _base(r):
        v=[r['M'],r['N'],r['K']]
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in v):raise ValueError('invalid shape')
        m,n,k=map(math.log,v)
        return np.array([m,n,k,k-n,m*(k-n)],dtype=float)

    def _x(self,r):
        base=self._base(r);dummy=np.array([float(r['site']==s) for s in self.sites[1:]])
        return np.concatenate((base,dummy,*(base*flag for flag in dummy)))

    def fit(self,rows):
        rows=tuple(rows);self.sites=tuple(sorted({r['site'] for r in rows}))
        if len(rows)<4 or not self.sites:raise ValueError('insufficient observations')
        x=np.stack([self._x(r) for r in rows]);y=np.array([math.log(r['stock_b2b']/r['dx_b2b']) for r in rows])
        mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-9,1.,sd);z=(x-mu)/sd
        beta=np.linalg.solve(z.T@z+self.ridge*np.eye(z.shape[1]),z.T@(y-y.mean()))
        self.params=(mu,sd,beta,float(y.mean()))
        return self

    def predict(self,row):
        if self.params is None:raise RuntimeError('not fitted')
        if row['site'] not in self.sites:return {'choice':'stock','reason':'unknown site'}
        mu,sd,beta,mean=self.params;gain=float(mean+((self._x(row)-mu)/sd)@beta)
        return {'choice':'overlap' if gain>0 else 'stock','predicted_log_speedup':gain}

    def to_dict(self):
        if self.params is None:raise RuntimeError('not fitted')
        mu,sd,beta,mean=self.params
        return {'schema':'stowaway-site-residual-v1','ridge':self.ridge,'sites':list(self.sites),
                'mu':mu.tolist(),'sd':sd.tolist(),'beta':beta.tolist(),'mean':mean}

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or data.get('schema')!='stowaway-site-residual-v1':raise ValueError('invalid schema')
        obj=cls(ridge=data['ridge']);obj.sites=tuple(data['sites'])
        if not obj.sites or len(set(obj.sites))!=len(obj.sites) or tuple(sorted(obj.sites))!=obj.sites:
            raise ValueError('invalid sites')
        size=5+6*(len(obj.sites)-1)
        mu,sd,beta=(np.asarray(data[k],dtype=float) for k in ('mu','sd','beta'));mean=float(data['mean'])
        if any(x.shape!=(size,) or not np.all(np.isfinite(x)) for x in (mu,sd,beta)) or np.any(sd<=0) or not math.isfinite(mean):
            raise ValueError('invalid parameters')
        obj.params=(mu,sd,beta,mean)
        return obj


class PipelineCorrectedSitePolicy:
    """Site estimate corrected by measured split-GEMM cost, in log latency.

    Coefficient is fixed before validation and split/full measurements must use
    the same route, shape, dtype and timing protocol. This is a heuristic: it
    attributes split GEMM overhead to overlap and ignores interference.
    """
    def __init__(self,base,*,coefficient=1.):
        if not math.isfinite(coefficient) or coefficient<0:raise ValueError('invalid coefficient')
        self.base=base;self.coefficient=coefficient

    def predict(self,row,*,full_ms,split_ms):
        if any(not math.isfinite(x) or x<=0 for x in (full_ms,split_ms)):raise ValueError('invalid component timings')
        base=self.base.predict(row)
        if 'predicted_log_speedup' not in base:return base
        # Baseline site model already captures typical split penalties. The
        # correction is anchored to no split overhead, so it may be overly
        # conservative; evaluate with independent target labels.
        corrected=base['predicted_log_speedup']-self.coefficient*max(0.,math.log(split_ms/full_ms))
        return {'choice':'overlap' if corrected>0 else 'stock','predicted_log_speedup':corrected,
                'base_log_speedup':base['predicted_log_speedup'],'split_ratio':split_ms/full_ms}
