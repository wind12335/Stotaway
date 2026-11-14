"""Collectively precompute a fixed batch of per-route operator decisions.

All group members must call once in the same order before execution or CUDA
graph capture. This is not a per-kernel collective and is not an E2E planner.
The leader's model decides after every rank validates identical requests.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .orthogonal_predictor import ContextRequirement, Route


@dataclass(frozen=True)
class Query:
    route: Route
    shape: tuple
    capability: ContextRequirement
    gemm_ms: float | None = None
    comm_ms: float | None = None
    margin_log: float = 0.


def _digest(queries):
    payload=json.dumps([asdict(q) for q in queries],sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_collectively(selector, queries, *, group=None):
    """Return ordered decisions or raise identically on all participating ranks.

    Requests must be identical, including route, shape, evidence and features.
    A different capability or local validation failure causes a collective
    error. ``selector`` is required only on the group's leader.
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError('distributed process group is not initialized')
    # No collective, including the error exchange, may be issued inside a
    # CUDA graph capture. All callers must enter this API at a safe point.
    import torch
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('distributed predictor must run outside CUDA graph capture')
    rank=dist.get_rank(group); size=dist.get_world_size(group)
    try:
        queries=tuple(queries)
        if not queries or any(not isinstance(q,Query) for q in queries):
            raise ValueError('nonempty Query list required')
        for q in queries:
            if q.capability.route!=q.route or q.capability.shape!=q.shape:
                raise ValueError('capability mismatch')
            if not math.isfinite(q.margin_log) or q.margin_log<0:
                raise ValueError('invalid margin')
        signature=_digest(queries)
        local={'digest':signature,'error':None}
    except Exception as exc:
        queries=()
        local={'digest':None,'error':f'{type(exc).__name__}: {exc}'}
    received=[None]*size
    dist.all_gather_object(received,local,group=group)
    failures=[v['error'] for v in received if v['error']]
    digests={v['digest'] for v in received}
    if failures or len(digests)!=1:
        raise RuntimeError(f'inconsistent predictor query batch: {received}')
    result=None
    if rank==0:
        try:
            if selector is None:raise ValueError('leader selector required')
            decisions=[selector.predict(q.route,q.shape,capability=q.capability,gemm_ms=q.gemm_ms,
                                        comm_ms=q.comm_ms,margin_log=q.margin_log) for q in queries]
            result={'digest':signature,'decisions':decisions,'error':None}
        except Exception as exc:
            result={'digest':signature,'decisions':None,'error':f'{type(exc).__name__}: {exc}'}
    item=[result]
    source=0 if group is None or group==dist.group.WORLD else dist.get_global_rank(group,0)
    dist.broadcast_object_list(item,src=source,group=group)
    result=item[0]
    if result['error']:
        raise RuntimeError(f'leader predictor error: {result["error"]}')
    if result['digest']!=signature or not isinstance(result['decisions'],list) or len(result['decisions'])!=len(queries):
        raise RuntimeError('predictor response mismatch')
    return result
