"""Compute observed net from paired full-workload timing blocks.

Use measured full-workload times when available. Operator-level arguments do
not become an E2E result merely because this helper sums them.
"""
import math
import statistics


def summarize(blocks, *, incremental_setup_ms=0., completed_units_per_block=1):
    blocks=tuple(blocks)
    if not blocks or isinstance(completed_units_per_block,bool) or not isinstance(completed_units_per_block,int) or completed_units_per_block<1:
        raise ValueError('blocks and a positive equal work count required')
    if isinstance(incremental_setup_ms,bool) or not isinstance(incremental_setup_ms,(int,float)) or not math.isfinite(incremental_setup_ms) or incremental_setup_ms<0:
        raise ValueError('incremental setup cost must be nonnegative')
    ratios=[];savings=[];seen=set()
    for x in blocks:
        ident=x.get('id')
        if not isinstance(ident,str) or not ident or ident in seen:raise ValueError('unique block ids required')
        seen.add(ident)
        for name in ('stock_ms','optimized_ms'):
            value=x.get(name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('positive finite wall times required')
        saving=x['stock_ms']-x['optimized_ms']-incremental_setup_ms
        savings.append(saving);ratios.append(x['stock_ms']/(x['optimized_ms']+incremental_setup_ms))
    return {'blocks':len(blocks),'units_per_block':completed_units_per_block,'incremental_setup_ms':incremental_setup_ms,
            'median_net_saving_ms':statistics.median(savings),
            'median_net_saving_ms_per_unit':statistics.median(savings)/completed_units_per_block,
            'geomean_net_speedup':math.exp(statistics.mean(map(math.log,ratios))),
            'slower_blocks':sum(r<1 for r in ratios),'ratios':ratios,
            'warning':'descriptive measured paired blocks; independent processes may drift; no future performance guarantee'}
