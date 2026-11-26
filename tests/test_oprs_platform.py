"""BW routes: vocabulary, static shape support, symmetric-heap bytes, platform block (platform.py)."""
import pytest

from oprs_synthetic import VERSIONS
from stowaway.oprs.platform import (ASYNC_TP, BACKENDS, DEO, DTYPES, METHODS, TRITON_DIST, allocations_heap_bytes,
                                    bw_route, heap_bytes, legacy_deo_route, platform_mismatches,
                                    runtime_platform, static_support)


def test_vocabulary():
    assert BACKENDS == (ASYNC_TP, TRITON_DIST) and METHODS == (DEO,) + BACKENDS and DTYPES
    assert (DEO, ASYNC_TP, TRITON_DIST) == ('deo', 'async_tp', 'triton_dist')     # the words of the handoff


def test_deo_support():
    assert static_support(DEO, 'ag', 32768, 896, 5120, 8) == ('supported', 'dx_ag_gemm_op: M % TP only')
    assert static_support(DEO, 'rs', 32768, 5120, 2176, 8)[0] == 'supported'        # inside the vehicle envelope
    assert static_support(DEO, 'rs', 32768, 8192, 28672, 8)[0] == 'supported'
    assert static_support(DEO, 'rs', 32768, 4100, 2176, 8) == (
        'unknown', 'outside the validated vehicle envelope (N % 128 == 0 and (M/TP) % 64 == 0): N=4100, M/TP=4096')
    assert static_support(DEO, 'ag', 32765, 896, 5120, 8)[0] == 'unsupported'       # M % TP


def test_async_tp_support():
    assert static_support(ASYNC_TP, 'ag', 16384, 1280, 8192, 8) == (
        'supported', 'torch.ops.symm_mem fused op: M % TP, 16-byte aligned shard')
    assert static_support(ASYNC_TP, 'rs', 16384, 8192, 1024, 8)[0] == 'supported'
    assert static_support(ASYNC_TP, 'rs', 1, 8, 7, 1)[0] == 'unsupported'          # 7 x 2 bytes is not 16-aligned
    assert static_support(ASYNC_TP, 'ag', 32767, 1280, 8192, 8)[0] == 'unsupported'


def test_triton_dist_support():
    ok = 'ag_gemm_intra_node: M % 128, N % 128, K % 16, (M/TP) % 1024'
    assert static_support(TRITON_DIST, 'ag', 16384, 1280, 8192, 8) == ('supported', ok)
    assert static_support(TRITON_DIST, 'ag', 32768, 1280, 8192, 8)[0] == 'supported'      # M/TP = 4096
    assert static_support(TRITON_DIST, 'ag', 16384, 1281, 8192, 8)[0] == 'unsupported'    # N % 128
    assert static_support(TRITON_DIST, 'ag', 16384, 1280, 8191, 8)[0] == 'unsupported'    # K % 16
    assert static_support(TRITON_DIST, 'ag', 28672, 1280, 8192, 8)[0] == 'unsupported'    # M/TP = 3584
    assert static_support(TRITON_DIST, 'rs', 32768, 8192, 3584, 8) == (
        'supported', 'gemm_rs_intra_node: M % 256, N % 256, K % 256')                     # llama3-70b fc2
    for shape in [(32768, 5120, 640), (32768, 5120, 2176)]:       # bw27b proj / fc2: K is no multiple of 256
        assert static_support(TRITON_DIST, 'rs', *shape, 8)[0] == 'unsupported'


def test_static_coverage_of_the_operator_table_at_tp8():
    """Which points of the 99-point table each method may run at all (shape rules only: whether a path
    is integrated is a separate fact, established by measurement)."""
    from stowaway.oprs.workloads import FIRST_BATCH, operator_inventory
    table, first = operator_inventory(8), operator_inventory(8, FIRST_BATCH, (32768,))
    for method in (DEO, ASYNC_TP):
        assert all(static_support(method, p['op'], p['M'], p['N'], p['K'], 8)[0] == 'supported' for p in table)
    blocked = [p['point_id'] for p in table if static_support(TRITON_DIST, p['op'], p['M'], p['N'], p['K'], 8)[0] != 'supported']
    assert len(blocked) == 12 and len(table) - len(blocked) == 87
    assert {b.split('/')[0] for b in blocked} == {'gemma3-27b', 'qwen25-72b'}
    assert {b.split('/')[1] for b in blocked if b.startswith('gemma3')} == {'rs'}       # K = 2688: no multiple of 256
    assert sum(static_support(TRITON_DIST, p['op'], p['M'], p['N'], p['K'], 8)[0] != 'supported'
               for p in first) == 3 and len(first) == 21                               # qwen25-72b is in the first batch


def test_unknown_method_or_operation():
    assert static_support('flux', 'ag', 8192, 1280, 8192, 8) == ('unsupported', 'unknown method/op flux/ag')
    assert static_support(DEO, 'a2a', 8192, 1280, 8192, 8)[0] == 'unsupported'
    assert static_support(DEO, 'ag', 8192, 1280, 8192, 0)[0] == 'unsupported'


def test_heap_bytes():
    # DEO: M x (K for ag, N for rs) x 2 bytes (+16 x tp flags); a row site with the fused backward holds two
    assert heap_bytes(DEO, 'ag', 32768, 896, 5120, 8) == 32768 * 5120 * 2 + 16 * 8
    assert heap_bytes(DEO, 'rs', 32768, 5120, 2176, 8) == 2 * (32768 * 5120 * 2) + 16 * 8
    assert heap_bytes(DEO, 'rs', 32768, 5120, 2176, 8, deo_bwd=False) == 32768 * 5120 * 2 + 16 * 8
    assert heap_bytes(ASYNC_TP, 'ag', 32768, 896, 5120, 8) == 0             # torch's own workspace, not the heap
    assert heap_bytes(TRITON_DIST, 'ag', 16384, 1280, 8192, 8) == 16384 * 8192 * 2 + 8 * 8 * (2048 // 1024) + 4096
    assert heap_bytes(TRITON_DIST, 'rs', 32768, 8192, 3584, 8) == 32768 * 8192 * 2 + 4096
    assert heap_bytes(DEO, 'ag', 32765, 896, 5120, 8) == 0 and heap_bytes(DEO, 'ag', 32768, 896, 5120, 0) == 0
    alloc = [(DEO, 'ag', 32768, 896, 5120), (DEO, 'rs', 32768, 5120, 2176)]
    assert allocations_heap_bytes(alloc, 8) == heap_bytes(DEO, 'ag', 32768, 896, 5120, 8) \
        + heap_bytes(DEO, 'rs', 32768, 5120, 2176, 8)
    assert allocations_heap_bytes(alloc, 8, deo_bwd=False) < allocations_heap_bytes(alloc, 8)


def test_runtime_platform_without_a_device():
    platform = runtime_platform(tp=8, topology='BW8', probe_device=False)
    assert (platform['vendor'], platform['device'], platform['comm'], platform['symmetric']) == (
        'hygon', 'BW1000', 'rccl', 'dushmem')
    assert platform['topology'] == 'BW8' and platform['device_name'] is None and platform['arch'] is None
    assert [v[0] for v in platform['versions']] == ['dtk', 'hip', 'torch', 'triton']
    given = runtime_platform(tp=8, topology='BW8', probe_device=False, device_name='BW', arch='gfx936')
    assert (given['device_name'], given['arch']) == ('BW', 'gfx936')
    assert runtime_platform(tp=4, probe_device=False)['topology'] == 'BW4'


def test_platform_mismatches():
    plan = runtime_platform(tp=8, topology='BW8', probe_device=False)
    assert platform_mismatches(plan, dict(plan)) == []
    for key, value in [('vendor', 'nvidia'), ('device', 'A800'), ('topology', 'BW4'), ('comm', 'nccl'),
                       ('symmetric', 'nvshmem')]:
        assert platform_mismatches({**plan, key: value}, plan) == [f'{key}: plan={value!r} runtime={plan[key]!r}']
    other = {**plan, 'versions': [list(v) for v in plan['versions']][:-1]}
    assert platform_mismatches(plan, other)[0].startswith('versions:')
    assert platform_mismatches(plan, {**plan, 'arch': 'gfx936'}) == []         # informational on either side
    assert platform_mismatches({}, plan)                                      # a plan without a platform is refused


def test_routes():
    for method in METHODS:
        for op in ('ag', 'rs'):
            route = bw_route(method, op, tp=8, topology='BW8', versions=VERSIONS)
            assert (route.method, route.direction, route.tp, route.gpu) == (method, op, 8, 'BW1000')
            assert route.timing_mode == 'b2b16' and route.warmup == 4
    assert bw_route(ASYNC_TP, 'ag', tp=8, topology='BW8', versions=VERSIONS).actual_path == 'torch_ops_symm_mem'
    assert bw_route(TRITON_DIST, 'rs', tp=8, topology='BW8', versions=VERSIONS).actual_path == 'triton_dist_dcu'
    old = legacy_deo_route('rs', tp=8, topology='BW8', versions=VERSIONS)
    assert old.timing_mode == 'b2b16_perop_events' and old.warmup is None and old.config_family == 'legacy_env'
    assert old != bw_route(DEO, 'rs', tp=8, topology='BW8', versions=VERSIONS)   # never one fit with the new table
    assert bw_route(DEO, 'ag', tp=8, topology='BW8', versions=VERSIONS, dtype='float16').dtype == 'float16'
    with pytest.raises(ValueError):
        bw_route(DEO, 'ar', tp=8, topology='BW8', versions=VERSIONS)
