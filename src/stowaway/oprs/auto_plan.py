"""Load a BW automatic route plan (serial / DEO / one external backend) with exact shape matching.

Hygon port of nvidia/stowaway_nvidia/auto_plan8.py. What differs from the NVIDIA loader:
  * the no-overlap choice is ``serial`` (upstream: ``stock``); the own overlap method is ``deo``
    (upstream: ``dx``);
  * TP, dtype and platform are read from the plan instead of being fixed to 8 / A800 / BF16;
  * the plan's platform block is compared with the running platform and any mismatch is a hard
    error, so an A800 plan or a plan fitted on another software stack is never dispatched here;
  * every non-serial entry and every allocated context must lie inside the static shape support
    of its BW route (a path that is not integrated or not shape-capable cannot be a candidate);
  * an unknown shape takes the explicit default ``serial`` and the dispatcher counts the fallback
    with its reason.
"""
import hashlib
import json
from pathlib import Path

from .platform import BACKENDS, DEO, DTYPES, platform_mismatches, static_support
from .predictor import OPS, SERIAL
from .workloads import SITE_OP, SITES

SCHEMA = 'stowaway-auto-plan-v1'
ENTRY_KEYS = {'site', 'op', 'M', 'N', 'K', 'choice', 'predicted_log_speedup', 'scores'}
REASONS = ('plan', 'unknown_shape', 'dtype_mismatch')


class AutoPlan:
    def __init__(self, path, workload, *, tp, runtime_platform):
        raw = Path(path).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        backend, dtype = data.get('external_backend'), data.get('dtype')
        if (data.get('schema') != SCHEMA or data.get('workload') != workload or data.get('tp') != tp
                or dtype not in DTYPES or backend not in ('none',) + BACKENDS
                or data.get('default') != SERIAL):
            raise ValueError('incompatible automatic plan')
        mismatches = platform_mismatches(data.get('platform') or {}, runtime_platform)
        if mismatches:
            raise ValueError('plan was generated for another platform: ' + '; '.join(mismatches))
        self.backend, self.dtype, self.default = backend, dtype, SERIAL
        self.methods = (DEO,) if backend == 'none' else (DEO, backend)
        self.platform, self.provenance = data.get('platform'), data.get('provenance') or {}
        self.entries = {}
        for item in data.get('entries', []):
            site, op = item.get('site'), item.get('op')
            dims = tuple(item.get(k) for k in ('M', 'N', 'K'))
            choice = item.get('choice')
            if (set(item) != ENTRY_KEYS or site not in SITES or SITE_OP[site] != op
                    or choice not in (SERIAL,) + self.methods
                    or any(type(v) is not int or v <= 0 for v in dims) or dims[0] % tp):
                raise ValueError('invalid plan entry')
            key = (site, op, *dims)
            if key in self.entries:
                raise ValueError('duplicate plan entry')
            if choice != SERIAL:
                self._require_support(choice, op, dims, tp, f'plan entry {key}')
            self.entries[key] = choice
        if not self.entries:
            raise ValueError('empty automatic plan')
        self.allocate = []
        for item in data.get('allocate', []):
            if (not isinstance(item, (list, tuple)) or len(item) != 5 or item[0] not in self.methods
                    or item[1] not in OPS or any(type(v) is not int or v <= 0 for v in item[2:])):
                raise ValueError('invalid plan allocation')
            method, op, dims = item[0], item[1], tuple(item[2:])
            if dims[0] % tp:
                raise ValueError('invalid plan allocation')
            self._require_support(method, op, dims, tp, 'plan allocation')
            if (method, op, *dims) not in self.allocate:
                self.allocate.append((method, op, *dims))

    @staticmethod
    def _require_support(method, op, dims, tp, what):
        support, evidence = static_support(method, op, *dims, tp)
        if support != 'supported':
            raise ValueError(f'{what} needs {method} on a shape it does not support: {evidence}')

    def choose(self, site, op, m, n, k, dtype):
        """(choice, reason). `m` is the global row count of the full projection."""
        # a site is one of the four call sites with its own direction (qkv/fc1: ag, proj/fc2: rs): an
        # adapter that asks for anything else has a bug, and that is not a fallback
        if not isinstance(site, str) or site not in SITES or SITE_OP[site] != op:
            raise ValueError('invalid runtime site or operation')
        if dtype != self.dtype:
            return SERIAL, 'dtype_mismatch'
        entry = self.entries.get((site, op, m, n, k))
        return (entry, 'plan') if entry is not None else (self.default, 'unknown_shape')
