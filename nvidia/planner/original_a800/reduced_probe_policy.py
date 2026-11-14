"""Two-GEMM-input route model with fold-local communication imputation.

Training rows include stock/communication labels. Prediction may only use
shape and target full/split GEMM; communication and stock estimates come
from this model's training fold. No cross-route sharing.
"""
import math
import numpy as np

from .piecewise_gain_policy import PiecewiseGainPolicy


class ReducedProbePolicy:
    def __init__(self,*,ridge=12.,scale_floor=0.):
        if not math.isfinite(scale_floor) or scale_floor<0:raise ValueError('invalid scale floor')
        self.ridge=ridge;self.scale_floor=scale_floor;self.comm_coeff=None;self.stock_coeff=None;self.gain_model=None

    @staticmethod
    def _payload(row,op):
        if op not in ('ag_up','ag_gu','rs'):raise ValueError('unsupported operation')
        return float(row['M']*(row['K'] if op.startswith('ag') else row['N'])*2/1e6)

    @staticmethod
    def _nonnegative_line(x,y):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
        if len(x)<2 or np.any(x<=0) or np.any(y<=0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('invalid communication measurements')
        a=np.stack((np.ones(len(x)),x),axis=1);w=1/y
        # Enumerate nonnegative coefficient boundaries; no new dependencies.
        weighted=a*w[:,None];target=y*w
        unconstrained=np.linalg.lstsq(weighted,target,rcond=None)[0]
        choices=[np.array([max(0.,float(np.mean(y))),0.]),
                 np.array([0.,max(0.,float(np.dot(weighted[:,1],target)/np.dot(weighted[:,1],weighted[:,1])))] )]
        if np.all(unconstrained>=0):choices.append(unconstrained)
        return min(choices,key=lambda v:float(np.sum((weighted@v-target)**2)))

    def fit(self,rows,*,op):
        rows=tuple(rows)
        if len(rows)<6 or any(r['op']!=op for r in rows):raise ValueError('one operation and at least six rows required')
        self.op=op
        self.comm_coeff=self._nonnegative_line([self._payload(r,op) for r in rows],[r['comm_ms'] for r in rows])
        # Small two-feature ridge in normalized space; constrain fitted stock
        # to positive at prediction, do not use target stock directly.
        x=np.asarray([[r['full_ms'],r['comm_ms']] for r in rows],dtype=float)
        y=np.asarray([r['stock_ms'] for r in rows],dtype=float)
        mu=x.mean(0);sd=np.maximum(x.std(0),1e-9);z=(x-mu)/sd
        beta=np.linalg.solve(z.T@z+2*np.eye(2),z.T@(y-y.mean()))
        self.stock_coeff=(mu,sd,beta,float(y.mean()))
        # Generate training covariates out-of-family so each target family's
        # communication/stock labels do not leak into its own imputation.
        families=sorted({r['family'] for r in rows})
        if len(families)<3:raise ValueError('three independent families required')
        imputed=[]
        for family in families:
            sub=[r for r in rows if r['family']!=family]
            probe=ReducedProbePolicy(ridge=self.ridge)
            probe.op=op
            probe.comm_coeff=probe._nonnegative_line([self._payload(r,op) for r in sub],[r['comm_ms'] for r in sub])
            x0=np.asarray([[r['full_ms'],r['comm_ms']] for r in sub],dtype=float)
            y0=np.asarray([r['stock_ms'] for r in sub],dtype=float)
            mu0=x0.mean(0);sd0=np.maximum(x0.std(0),1e-9);z0=(x0-mu0)/sd0
            beta0=np.linalg.solve(z0.T@z0+2*np.eye(2),z0.T@(y0-y0.mean()))
            probe.stock_coeff=(mu0,sd0,beta0,float(y0.mean()))
            for r in rows:
                if r['family']==family:
                    q=probe._impute(r);imputed.append({**r,'comm_ms':q['comm_ms'],'stock_ms':q['stock_ms']})
        self.gain_model=PiecewiseGainPolicy(ridge=self.ridge,scale_floor=self.scale_floor).fit(imputed)
        return self

    def _impute(self,row):
        if self.comm_coeff is None or row['op']!=self.op:raise ValueError('route operation mismatch')
        c=max(float(self.comm_coeff[0]+self.comm_coeff[1]*self._payload(row,self.op)),1e-6)
        mu,sd,beta,mean=self.stock_coeff
        stock=max(float(mean+((np.array([row['full_ms'],c])-mu)/sd)@beta),1e-6)
        return {'comm_ms':c,'stock_ms':stock}

    def predict(self,row):
        if self.gain_model is None:raise RuntimeError('not fitted')
        estimated=self._impute(row);result=self.gain_model.predict({**row,**estimated})
        return {**result,'estimated_comm_ms':estimated['comm_ms'],'estimated_stock_ms':estimated['stock_ms']}

    def to_dict(self):
        if self.gain_model is None:raise RuntimeError('not fitted')
        mu,sd,beta,mean=self.stock_coeff
        return {'schema':'stowaway-reduced-probe-v1','op':self.op,'ridge':self.ridge,'scale_floor':self.scale_floor,
                'comm_coeff':self.comm_coeff.tolist(),'stock_mu':mu.tolist(),'stock_sd':sd.tolist(),
                'stock_beta':beta.tolist(),'stock_mean':mean,'gain_model':self.gain_model.to_dict()}

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='stowaway-reduced-probe-v1':raise ValueError('invalid schema')
        result=cls(ridge=data['ridge'],scale_floor=data.get('scale_floor',0.));result.op=data['op']
        if result.op not in ('ag_up','ag_gu','rs'):raise ValueError('invalid operation')
        result.comm_coeff=np.asarray(data['comm_coeff'],dtype=float)
        mu,sd,beta=(np.asarray(data[k],dtype=float) for k in ('stock_mu','stock_sd','stock_beta'));mean=float(data['stock_mean'])
        if (result.comm_coeff.shape!=(2,) or np.any(result.comm_coeff<0) or not np.all(np.isfinite(result.comm_coeff)) or
            any(v.shape!=(2,) or not np.all(np.isfinite(v)) for v in (mu,sd,beta)) or np.any(sd<=0) or not math.isfinite(mean)):
            raise ValueError('invalid parameters')
        result.stock_coeff=(mu,sd,beta,mean);result.gain_model=PiecewiseGainPolicy.from_dict(data['gain_model'])
        return result
