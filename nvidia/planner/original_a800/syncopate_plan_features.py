"""Read CPU schedule features from an externally supplied Syncopate plan.

No GPU transport path or performance is inferred; caller supplies dtype bytes
and matrix output width. Raw upstream schedules remain the source of truth.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleFeatures:
    ranks: int
    waves: int
    blocks: int
    signaled_blocks: int
    signaled_bytes: int
    gemm_flops: int
    dependency_waves: int


def summarize(schedules,*,buffer_name,k,n_local,dtype_bytes):
    if any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in (k,n_local,dtype_bytes)):
        raise ValueError('positive integer dimensions required')
    ranks=sorted(schedules)
    if ranks!=list(range(len(ranks))) or not ranks:raise ValueError('contiguous rank schedules required')
    per_rank=[]
    for rank in ranks:
        sched=schedules[rank][buffer_name]
        shapes=sched.gen_block_shape_lists();signals=sched.gen_signal_lists()
        if len(shapes)!=len(signals) or not shapes:raise ValueError('invalid schedule')
        if any(len(shape)!=2 or shape[0]<=0 or shape[1]!=k for shape in shapes):raise ValueError('unexpected gather shape')
        signaled=sum(sig>=0 for sig in signals)
        signaled_rows=sum(shape[0] for shape,sig in zip(shapes,signals) if sig>=0)
        rows=sum(shape[0] for shape in shapes)
        waves=sched.block_infos
        if len(waves)!=sched.num_waves or sum(map(len,waves))!=len(shapes):raise ValueError('wave/block mismatch')
        dependency_waves=sum(any(block.signal is not None for block in wave) for wave in waves)
        per_rank.append(ScheduleFeatures(len(ranks),sched.num_waves,len(shapes),signaled,
                                         signaled_rows*k*dtype_bytes,2*rows*n_local*k,dependency_waves))
    return per_rank
