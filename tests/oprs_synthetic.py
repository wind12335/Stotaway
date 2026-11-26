"""Synthetic operator samples with a known law, for the CPU tests: no GPU, no measured data.

The law is linear in (log M, log N, log K) -- what the route model assumes -- so a fit on it is
right up to ridge shrinkage, and the correct choice at every point is known:
    log(serial / candidate) = gain(method) + 0.03 log(M / 8192) - 0.02 log(N K / (1536 x 4096))
"""
from dataclasses import asdict
import json
import math

from stowaway.oprs.data import NEW, SAMPLE_SCHEMA, OperatorRecord
from stowaway.oprs.platform import ASYNC_TP, DEO, TRITON_DIST, bw_route
from stowaway.oprs.predictor import Sample

VERSIONS = (('hip', '6.3.26093'), ('torch', '2.9.0'))     # sorted unique pairs; no device is probed
FAMILIES = tuple(f'fam-{i}' for i in range(11))
# qwen3-8b at TP=8: inside the static support of all three methods
SHAPES = {'rs': (4096, 1536), 'ag_up': (1536, 4096), 'ag_gu': (3072, 4096)}
MS = (8192, 16384, 32768)
GAIN = {DEO: 0.15, ASYNC_TP: -0.05, TRITON_DIST: 0.02}


def law(method, m, n, k, gain=None):
    """`gain` may be a number or a callable (method, m, n, k) -> gain."""
    base = GAIN[method] if gain is None else (gain(method, m, n, k) if callable(gain) else gain)
    return base + 0.03 * math.log(m / 8192) - 0.02 * math.log(n * k / (1536 * 4096))


def grows_with_m(method, m, n, k):
    """A backend that loses at M=8192 (-0.5), nearly ties at 16384 (-0.1) and wins at 32768 (+0.3)."""
    return -0.5 + 0.4 * math.log2(m / 8192)


def records(method, op, tp=8, families=FAMILIES, gain=None, ms=MS):
    """Operator-table records of ONE route (method, op)."""
    route = bw_route(method, op, tp=tp, topology=f'BW{tp}', versions=VERSIONS)
    out = []
    for family in families:
        for table_op in [t for t in SHAPES if (t == 'rs') == (op == 'rs')]:
            n, k = SHAPES[table_op]
            for m in ms:
                out.append(OperatorRecord(
                    family=family, sample=Sample(route, (m, n, k), 1.0, math.exp(-law(method, m, n, k, gain))),
                    op=op, method=method, model=f'model-{family}', kind='operator', protocol=NEW,
                    origin=f'synthetic/{method}/{op}/{family}/{table_op}/M{m}', table_op=table_op))
    return out


def every_record(tp=8, **kw):
    """All three methods, both directions."""
    return [r for method in (DEO, ASYNC_TP, TRITON_DIST) for op in ('ag', 'rs')
            for r in records(method, op, tp, **kw)]


def sample_line(r):
    """One record as the line bench/oprs_matrix.py writes (schema stowaway-oprs-operator-sample-v1)."""
    route = asdict(r.sample.route)
    route['versions'] = [list(v) for v in route['versions']]
    m, n, k = r.shape
    speedup = r.sample.stock_ms / r.sample.candidate_ms
    return {'kind': r.kind, 'point_id': f'{r.model}/{r.table_op}/M{m}', 'model': r.model, 'site': None,
            'origin': r.origin, 'table_op': r.table_op, 'op': r.op, 'anomaly': None, 'family': r.family,
            'M': m, 'N': n, 'K': k, 'tp': route['tp'], 'dtype': route['dtype'], 'schema': SAMPLE_SCHEMA,
            'protocol': NEW, 'data_class': NEW, 'batch': 16, 'repeats': 8, 'warmup': 4, 'interleave': True,
            'python_gc': True, 'platform': None, 'software_versions': [list(v) for v in VERSIONS],
            'commit': 'synthetic', 'src_dirty': None, 'gpu': None, 'source_files': [],
            'triton_dist_source': None, 'fp32_matmul_allow_tf32': False, 'env': {},
            'source_files_note': 'synthetic test data', 'software': {}, 'candidate': r.method,
            'status': 'ok', 'support': {'status': 'supported', 'evidence': 'synthetic'}, 'route': route,
            'run_order': [], 'batch_id': 'synthetic', 'point_in_process': 0, 'serial_samples_ms': [],
            'candidate_samples_ms': [], 'serial_reference_median_ms': r.sample.stock_ms,
            'candidate_median_ms': r.sample.candidate_ms,
            'median_ms': {'serial': r.sample.stock_ms, 'candidate': r.sample.candidate_ms,
                          'gemm_only': 0.6, 'comm_only': 0.4},
            'speedup': speedup, 'lower_observed': speedup * 0.9, 'upper_observed': speedup * 1.1,
            'serial_parts_ms': {}, 'serial_part_samples_ms': {}, 'per_rank': {},
            'correctness': {'checked': True, 'pass': True, 'tol': 0.02, 'rel_vs_serial': 1e-6},
            'timing_scope': 'synthetic', 'warning': '', 'gpu_state_at_point': None}


def write_samples(rows, path, extra=()):
    with open(path, 'w') as handle:
        for r in rows:
            handle.write(json.dumps(sample_line(r)) + '\n')
        for line in extra:
            handle.write(json.dumps(line) + '\n')
    return str(path)
