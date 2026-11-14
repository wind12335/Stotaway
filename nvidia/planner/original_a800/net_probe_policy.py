"""Cost-aware decision from paired same-route calibration blocks.

This policy is only an operator-level estimate; production adapters must call it
outside graph capture and coordinate all ranks. No performance guarantee.
"""
import math
import statistics


def select_paired(samples, *, probe_ms, remaining_calls, mad_multiplier=1.):
    samples=tuple(samples)
    if len(samples)<2 or isinstance(remaining_calls,bool) or not isinstance(remaining_calls,int) or remaining_calls<1:
        raise ValueError('at least two blocks and a positive call horizon required')
    for value in (probe_ms,mad_multiplier):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
            raise ValueError('nonnegative finite probe cost and multiplier required')
    savings=[]
    for sample in samples:
        for arm in ('stock','dx'):
            value=sample.get(arm)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
                raise ValueError('positive finite paired times required')
        savings.append(sample['stock']-sample['dx'])
    median=statistics.median(savings);mad=statistics.median(abs(v-median) for v in savings)
    conservative=median-mad_multiplier*mad
    net=conservative*remaining_calls-probe_ms
    return {'choice':'dx' if net>0 else 'stock','median_saving_ms':median,'mad_ms':mad,
            'conservative_saving_ms':conservative,'predicted_net_ms':net,'probe_ms':probe_ms,
            'remaining_calls':remaining_calls,'blocks':len(samples),
            'warning':'paired-sample heuristic; cost must include all incremental setup; not a confidence bound or E2E estimate'}
