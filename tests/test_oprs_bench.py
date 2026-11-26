"""Pure pieces of the operator bench (bench/oprs_matrix.py): protocol constants, point selection, stats."""
import argparse
import json

import pytest

from stowaway.oprs.platform import WARMUP
from stowaway.oprs.workloads import FIRST_BATCH, RECHECK, SITE_TABLE_OP


@pytest.fixture(scope='module')
def bench(scripts):
    return scripts('oprs_matrix', 'bench', 'oprs_matrix.py')


def ns(**kw):
    base = dict(points=None, first_batch=False, sites=None, mbs=None, seq=None, rows=None, config=None,
                models='', ms='')
    return argparse.Namespace(**{**base, **kw})


def test_protocol_constants(bench):
    """b2b16: a group is 16 back-to-back calls, 8 repeat groups per arm, WARMUP calls before timing."""
    assert (bench.REPEATS, bench.BATCH) == (8, 16) and WARMUP == 4
    assert 'not an in-framework result' in bench.WARNING and 'forward-only' in bench.WARNING
    assert bench.SITE_OF_TABLE_OP == {v: k for k, v in SITE_TABLE_OP.items()}


def test_rotate_gives_no_arm_a_fixed_place(bench):
    arms = ['serial', 'deo', 'async_tp']
    rounds = [bench.rotate(arms, r) for r in range(bench.REPEATS)]
    assert all(sorted(r) == sorted(arms) for r in rounds)
    assert rounds[0] == arms and rounds[1] == arms[1:] + arms[:1]
    for a in arms:                                   # over 8 rounds no arm is always first or always last
        assert any(r[0] == a for r in rounds) and any(r[-1] == a for r in rounds)
    assert bench.rotate(arms, 3) == arms and bench.rotate(arms, 8) == arms[2:] + arms[:2]
    assert bench.rotate(arms, -1) == arms[-1:] + arms[:-1]


def test_clean_names_the_non_finite(bench):
    assert bench.clean({'a': float('nan'), 'b': [1.0, float('inf')], 'c': (float('-inf'),), 'd': 'x', 'e': 2}) == {
        'a': 'nan', 'b': [1.0, 'inf'], 'c': ['-inf'], 'd': 'x', 'e': 2}


def test_summary_is_a_ratio_of_medians_never_a_mean_of_ratios(bench):
    slow = {'serial': [2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'candidate': [1.0] * 7 + [5.0]}
    assert bench.summary(slow) == {'serial_ms': 2.0, 'candidate_ms': 1.0, 'speedup': 2.0,
                                   'lower_observed': 0.4, 'upper_observed': 2.0}
    flat = bench.summary({'serial': [3.0] * 8, 'candidate': [1.5] * 8})
    assert flat == {'serial_ms': 3.0, 'candidate_ms': 1.5, 'speedup': 2.0, 'lower_observed': 2.0,
                    'upper_observed': 2.0}


def test_point_selection(bench):
    table = bench.select_points(ns(), 8)
    assert len(table) == 99 and {p['kind'] for p in table} == {'operator'}
    first = bench.select_points(ns(first_batch=True), 8)
    assert len(first) == 21 and {p['M'] for p in first} == {32768} and {p['model'] for p in first} == set(FIRST_BATCH)
    some = bench.select_points(ns(models='llama3-8b,qwen3-32b', ms='8192,16384'), 8)
    assert len(some) == 2 * 3 * 2 and {p['point_id'] for p in some} >= {'llama3-8b/rs/M8192', 'qwen3-32b/ag_gu/M16384'}
    picked = bench.select_points(ns(points='llama3-8b/rs/M8192,qwen3-32b/ag_gu/M16384'), 8)
    assert [p['point_id'] for p in picked] == ['llama3-8b/rs/M8192', 'qwen3-32b/ag_gu/M16384']
    with pytest.raises(ValueError, match='unknown points'):
        bench.select_points(ns(points='llama3-8b/rs/M8192,nope/rs/M8192'), 8)


def test_the_two_legacy_anomalies_keep_their_label(bench):
    labelled = [p for p in bench.select_points(ns(), 8) if p['anomaly']]
    assert sorted(p['point_id'] for p in labelled) == sorted(RECHECK)
    assert all(p['recheck_of'] == p['point_id'] and p['anomaly'] == f'legacy-anomaly:{p["point_id"]}'
               for p in labelled)
    assert all(p['anomaly'] is None and p['recheck_of'] is None
               for p in bench.select_points(ns(), 8) if p['point_id'] not in RECHECK)


def test_call_site_points_are_targets_not_training_material(bench):
    sites = bench.select_points(ns(sites='bw27b_train'), 8)
    assert [(p['kind'], p['site'], p['table_op'], p['op'], p['M']) for p in sites] == [
        ('site', 'qkv', 'ag_up', 'ag', 32768), ('site', 'proj', 'rs', 'rs', 32768),
        ('site', 'fc1', 'ag_gu', 'ag', 32768), ('site', 'fc2', 'rs', 'rs', 32768)]
    assert all(p['model'] == 'bw27b_train' and p['anomaly'] is None for p in sites)
    assert {p['point_id'] for p in sites} == {f'bw27b_train/{s}/M32768' for s in ('qkv', 'proj', 'fc1', 'fc2')}
    small = bench.select_points(ns(sites='custom_train', mbs=1, seq=4096, rows=None,
                                   config='{"hidden":1024,"ffn":2048,"heads":16,"kv_groups":4}'), 1)
    assert {p['M'] for p in small} == {4096} and bench.select_points(ns(sites='bw70b_train'), 8)[0]['N'] == 1280
    key = bench.point_key(sites[0], 'deo')
    assert key == ('site', 'bw27b_train', 'ag_up', 'qkv', 32768, 896, 5120, 'deo')


def test_resume_skips_what_is_finished(bench, tmp_path):
    path = tmp_path / 'samples.jsonl'
    rows = [{'kind': 'operator', 'model': 'm', 'table_op': 'rs', 'site': None, 'M': 8192, 'N': 4096, 'K': 1536,
             'candidate': c, 'status': s}
            for c, s in (('deo', 'ok'), ('async_tp', 'unsupported'), ('triton_dist', 'error'), ('deo', 'error'))]
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    assert bench.done_set(path) == {('operator', 'm', 'rs', None, 8192, 4096, 1536, 'deo'),
                                    ('operator', 'm', 'rs', None, 8192, 4096, 1536, 'async_tp')}
    assert bench.done_set(tmp_path / 'none.jsonl') == set()


def test_sha256_of_a_file(bench, tmp_path):
    path = tmp_path / 'x'
    path.write_bytes(b'hello')
    import hashlib
    assert bench.sha256(path) == hashlib.sha256(b'hello').hexdigest()
