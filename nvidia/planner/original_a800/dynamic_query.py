"""Construct a new route query only from explicit shape capability evidence."""
from .distributed_inference import Query
from .orthogonal_predictor import ContextRequirement


def make_query(route,shape,*,support,evidence):
    """Build a query for a synchronized safe point; this doesn't discover shapes."""
    cap=ContextRequirement(route,shape,support,evidence)
    if shape[0]%route.tp:
        raise ValueError('shape M not divisible by tensor parallel size')
    return Query(route,shape,cap)


def update_cache(cache,site,query,result):
    """Insert a broadcast decision, rejecting conflicting preexisting entries."""
    from ..tp2_lookup import key
    choice=result.get('choice')
    if choice not in ('stock',query.route.method):raise ValueError('invalid route decision')
    if query.capability.support!='supported' and choice!='stock':raise ValueError('unsupported capability cannot enable overlap')
    expected='ag' if site in ('qkv','fc1') else 'rs' if site in ('proj','fc2') else None
    if expected is None or query.route.direction!=expected:raise ValueError('site/collective mismatch')
    m,n,k=query.shape;name=key(site,m,k,n)
    value='stock' if choice=='stock' else 'dx' if query.route.method=='dx' else choice
    if name in cache and cache[name]!=value:raise ValueError('conflicting dynamic decision')
    cache[name]=value
    return name
