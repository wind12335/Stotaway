"""Route-local ridge prediction with group-held-out nearest residual correction.

Correction is fitted only from out-of-family predictions. Neighbors are never
shared across operations, platforms, or measurement protocols.
"""
import math
import numpy as np

from .reduced_probe_policy import ReducedProbePolicy


def feature(row):
    vals=[row[k] for k in ('M','N','K','full_ms','split_ms')]
    if any(not isinstance(x,(float,int)) or isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in vals):
        raise ValueError('positive finite inputs required')
    m,n,k,g,split=map(float,vals)
    return np.array([math.log(m),math.log(n),math.log(k),math.log(split/g)])


class LocalResidualPolicy:
    def __init__(self,*,neighbors=3,shrink=.5,scale_floor=.05):
        if not isinstance(neighbors,int) or isinstance(neighbors,bool) or neighbors<1 or not 0<=shrink<=1:
            raise ValueError('invalid local residual settings')
        self.neighbors=neighbors;self.shrink=float(shrink);self.scale_floor=scale_floor
        self.model=None;self.support=[]

    def fit(self,rows,*,op):
        rows=tuple(rows);families=sorted({r['family'] for r in rows})
        if len(families)<4 or any(r['op']!=op for r in rows):
            raise ValueError('at least four physical families of one operation required')
        self.op=op;self.model=ReducedProbePolicy(scale_floor=self.scale_floor).fit(rows,op=op)
        all_features=np.stack([feature(r) for r in rows]);self.center=all_features.mean(0)
        self.scale=np.maximum(all_features.std(0),[.25,.15,.15,.10])
        support=[]
        for family in families:
            sub=ReducedProbePolicy(scale_floor=self.scale_floor).fit([r for r in rows if r['family']!=family],op=op)
            for r in rows:
                if r['family']==family:
                    predicted=sub.predict(r)['predicted_log_speedup']
                    residual=math.log(r['stock_ms']/r['overlap_ms'])-predicted
                    support.append((family,feature(r),residual))
        self.support=support;return self

    def predict(self,row):
        if self.model is None or row['op']!=self.op:raise ValueError('route operation mismatch')
        original=self.model.predict(row)['predicted_log_speedup'];q=feature(row)
        ordered=sorted(((float(np.linalg.norm((q-x)/self.scale)),family,residual)
                        for family,x,residual in self.support),key=lambda v:(v[0],v[1]))
        # At most one nearest example from a physical family: repeated M or
        # site aliases cannot receive duplicate votes for the same family.
        seen=set();neighbors=[]
        for distance,family,residual in ordered:
            if family not in seen:
                neighbors.append((distance,family,residual));seen.add(family)
            if len(neighbors)==self.neighbors:break
        weights=np.array([1/(.25+d) for d,_,_ in neighbors]);residuals=np.array([r for _,_,r in neighbors])
        correction=self.shrink*float(weights@residuals/weights.sum())
        return {'choice':'overlap' if original+correction>0 else 'stock',
                'predicted_log_speedup':original+correction,'base_log_speedup':original,
                'correction':correction,'neighbors':[{'family':f,'distance':d,'residual':r} for d,f,r in neighbors]}

    def to_dict(self):
        if self.model is None:raise RuntimeError('unfitted local model')
        return {'schema':'stowaway-local-residual-v1','op':self.op,'neighbors':self.neighbors,
                'shrink':self.shrink,'scale_floor':self.scale_floor,'model':self.model.to_dict(),
                'center':self.center.tolist(),'scale':self.scale.tolist(),
                'support':[{'family':f,'features':x.tolist(),'residual':r} for f,x,r in self.support]}

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='stowaway-local-residual-v1':raise ValueError('invalid schema')
        instance=cls(neighbors=data['neighbors'],shrink=data['shrink'],scale_floor=data['scale_floor'])
        instance.op=data['op'];instance.model=ReducedProbePolicy.from_dict(data['model'])
        instance.center=np.asarray(data['center'],dtype=float);instance.scale=np.asarray(data['scale'],dtype=float)
        if (instance.op!=instance.model.op or instance.center.shape!=(4,) or instance.scale.shape!=(4,)
                or not np.all(np.isfinite(instance.center)) or not np.all(np.isfinite(instance.scale))
                or np.any(instance.scale<=0)):
            raise ValueError('invalid local model parameters')
        instance.support=[]
        for row in data['support']:
            f=row['family'];x=np.asarray(row['features'],dtype=float);r=row['residual']
            if not isinstance(f,str) or not f or x.shape!=(4,) or not np.all(np.isfinite(x)) or not math.isfinite(r):
                raise ValueError('invalid local residual support')
            instance.support.append((f,x,float(r)))
        if len({f for f,_,_ in instance.support})<4:raise ValueError('insufficient family support')
        return instance
