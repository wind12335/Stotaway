"""Exact-shape, opt-in diagnostic plan for one DX implementation.

This schema carries user-supplied decisions; it is not a validated predictor.
The caller must compare plan digests across ranks before enabling the adapter.
Missing shapes use stock. BRS is a separate switch and is not controlled here.
"""
import hashlib
import json
from pathlib import Path


class ShapePlan:
    def __init__(self, path, *, tp, dtype, gpu, torch_version, triton_version, cuda_build,
                 schedule, batch_ack, backward, brs, brs_min_bytes, brs_slots,
                 cuda_device_max_connections):
        raw = Path(path).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if data.get('schema') != 'stowaway-dx-shape-plan-v1' or data.get('classification') != 'diagnostic-manual':
            raise ValueError('unsupported diagnostic shape plan')
        expected = {'tp': tp, 'dtype': dtype, 'gpu': gpu, 'torch': torch_version,
                    'triton': triton_version, 'cuda_build': cuda_build, 'schedule': schedule,
                    'batch_ack': batch_ack, 'backward': backward, 'brs': brs,
                    'brs_min_bytes': brs_min_bytes, 'brs_slots': brs_slots,
                    'cuda_device_max_connections': cuda_device_max_connections}
        if data.get('runtime') != expected:
            raise ValueError('shape plan runtime does not match')
        if data.get('backend') != 'dx':
            raise ValueError('this adapter only supports DX')
        self.entries = {}
        for entry in data.get('entries', []):
            site = entry.get('site')
            dims = tuple(entry.get(k) for k in ('M', 'K', 'N'))
            choice = entry.get('choice')
            if site not in ('qkv', 'proj', 'fc1', 'fc2') or choice not in ('stock', 'dx'):
                raise ValueError('invalid site or choice')
            if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in dims) or dims[0] % tp:
                raise ValueError('invalid shape')
            key = (site, *dims)
            if key in self.entries:
                raise ValueError('duplicate plan entry')
            self.entries[key] = choice

    def choose(self, site, m, k, n, dtype):
        if dtype != 'bfloat16':
            return 'stock'
        return self.entries.get((site, m, k, n), 'stock')
