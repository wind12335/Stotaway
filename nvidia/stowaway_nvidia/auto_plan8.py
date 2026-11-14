"""Load a TP8 automatic route plan with exact shape matching."""
import hashlib
import json
from pathlib import Path


class AutoPlan8:
    def __init__(self, path, workload):
        raw = Path(path).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        backend = data.get('external_backend')
        dtype = data.get('dtype')
        if (data.get('schema') != 'stowaway-tp8-auto-plan-v1' or data.get('workload') != workload
                or data.get('tp') != 8 or dtype not in ('bfloat16', 'float16')
                or backend not in ('async_tp', 'triton_dist', 'flashoverlap', 'syncopate')
                or data.get('default') not in ('stock', backend)):
            raise ValueError('incompatible automatic plan')
        self.backend, self.dtype, self.default = backend, dtype, data['default']
        self.entries = {}
        for item in data.get('entries', []):
            site, op = item.get('site'), item.get('op')
            dims = tuple(item.get(k) for k in ('M', 'N', 'K'))
            choice = item.get('choice')
            if (site not in ('qkv', 'proj', 'fc1', 'fc2') or op not in ('ag', 'rs')
                    or choice not in ('stock', backend, 'dx')
                    or (site in ('qkv', 'fc1') and op != 'ag')
                    or (site in ('proj', 'fc2') and op != 'rs')
                    or (backend == 'flashoverlap' and choice == backend and op != 'rs')
                    or any(type(v) is not int or v <= 0 for v in dims) or dims[0] % 8):
                raise ValueError('invalid plan entry')
            key = (site, op, *dims)
            if key in self.entries:
                raise ValueError('duplicate plan entry')
            self.entries[key] = choice
        if not self.entries:
            raise ValueError('empty automatic plan')

    def choose(self, site, op, m, n, k, dtype):
        if not isinstance(site, str) or not site or op not in ('ag', 'rs'):
            raise ValueError('invalid runtime site or operation')
        if dtype != self.dtype:
            return 'stock'
        return self.entries.get((site, op, m, n, k), self.default)
