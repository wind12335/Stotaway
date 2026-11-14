"""Measured whole-step mask selection within one explicitly identified context.

Blocks must compare exactly the same set of masks, including stock. This is a
small-sample exploratory heuristic: the MAD penalty is not a confidence bound.
No extrapolation to a different workload, backend, mask or runtime is implied.
"""
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class StepBlock:
    context: str
    block_id: str
    milliseconds: dict

    def __post_init__(self):
        if not isinstance(self.context, str) or not self.context or not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError('context and block id required')
        if not isinstance(self.milliseconds, dict) or 'stock' not in self.milliseconds:
            raise ValueError('stock and measured masks required')
        for mask, value in self.milliseconds.items():
            if not isinstance(mask, str) or not mask or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid measured mask or time')


def select(blocks, *, context, allowed_masks, mad_penalty=1.0, min_blocks=2):
    """Choose a measured mask by median delta plus a dispersion penalty.

    ``allowed_masks`` is the caller's capability-checked inventory; it must
    come from the same implementation and context as the measurements. Stock
    wins ties. Deltas are paired within blocks to reduce shared drift.
    """
    blocks = tuple(blocks)
    allowed_masks = set(allowed_masks)
    if not isinstance(context, str) or not context or 'stock' not in allowed_masks:
        raise ValueError('explicit context and stock capability required')
    if not math.isfinite(mad_penalty) or mad_penalty < 0 or not isinstance(min_blocks, int) or isinstance(min_blocks, bool) or min_blocks < 2:
        raise ValueError('invalid policy settings')
    if any(b.context != context for b in blocks) or len({b.block_id for b in blocks}) != len(blocks):
        raise ValueError('context mismatch or repeated block id')
    if any(set(b.milliseconds) != allowed_masks for b in blocks):
        raise ValueError('each block must measure the same capability-checked masks')
    if len(blocks) < min_blocks:
        return {'choice': 'stock', 'status': 'insufficient blocks', 'masks': {}}
    masks = {}
    for mask in sorted(allowed_masks - {'stock'}):
        deltas = [b.milliseconds[mask] - b.milliseconds['stock'] for b in blocks]
        median = statistics.median(deltas)
        mad = statistics.median(abs(x - median) for x in deltas)
        masks[mask] = {'median_delta_ms': median, 'mad_ms': mad,
                       'score_ms': median + mad_penalty * mad, 'deltas_ms': deltas}
    choice = min(masks, key=lambda m: (masks[m]['score_ms'], m), default='stock')
    if choice != 'stock' and masks[choice]['score_ms'] >= 0:
        choice = 'stock'
    return {'choice': choice, 'status': 'experimental measured-mask selection', 'masks': masks,
            'block_ids': [b.block_id for b in blocks], 'context': context,
            'mad_penalty': mad_penalty, 'warning': 'dispersion penalty is not a statistical guarantee'}
