"""Choose which already-calibrated complete execution plan to deploy.

This operates on full-request or full-step measurements, not operator gains.
The cost bound is empirical and has no coverage guarantee.
"""
import math
import statistics


def select_execution_mode(blocks, *, calibration_ms, horizon, uncertainty_multiplier=1.):
    """Select optimized only when conservative net expected savings is positive.

    Each independent block contains one stock and one optimized full workload
    latency. Within-process repetitions must first be summarized to one value.
    """
    blocks=tuple(blocks)
    if len(blocks)<2 or isinstance(horizon,bool) or not isinstance(horizon,int) or horizon<1:
        raise ValueError('at least two blocks and positive horizon required')
    for value in (calibration_ms,uncertainty_multiplier):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('nonnegative finite cost and uncertainty multiplier required')
    deltas=[];seen=set()
    for block in blocks:
        if not isinstance(block,dict) or not isinstance(block.get('id'),str) or not block['id'] or block['id'] in seen:
            raise ValueError('unique nonempty block id required')
        seen.add(block['id'])
        for name in ('stock_ms','optimized_ms'):
            v=block.get(name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError('positive finite full-workload timings required')
        deltas.append(block['stock_ms']-block['optimized_ms'])
    center=statistics.median(deltas);mad=statistics.median(abs(x-center) for x in deltas)
    conservative=center-uncertainty_multiplier*mad
    net=conservative*horizon-calibration_ms
    return {'choice':'optimized' if net>0 else 'stock','median_saving_ms':center,'mad_ms':mad,
            'conservative_saving_ms':conservative,'estimated_net_saving_ms':net,
            'horizon':horizon,'calibration_ms':calibration_ms,'blocks':len(blocks),
            'warning':'empirical heuristic; not a confidence interval or transferable workload guarantee'}
