"""Operator-sample records: the new-protocol JSONL of the bench and the legacy table (data.py)."""
import json
import math

import pytest

from oprs_synthetic import DEO, VERSIONS, law, records, sample_line, write_samples
from stowaway.oprs.data import (LEGACY, NEW, grouped, load_legacy_cells, load_samples, op_of,
                                record_from_dict, usable)
from stowaway.oprs.platform import legacy_deo_route

LEGACY_TABLE = 'planner/data/cells.json'


def test_sample_line_round_trip(tmp_path):
    rows = records(DEO, 'rs', families=tuple(f'fam-{i}' for i in range(3)))
    loaded, failures = load_samples([write_samples(rows, tmp_path / 'a.jsonl')])
    assert failures == [] and len(loaded) == len(rows)
    for got, want in zip(loaded, rows):
        assert (got.family, got.shape, got.method, got.op, got.table_op, got.protocol) == (
            want.family, want.shape, want.method, want.op, want.table_op, NEW)
        assert got.log_speedup == pytest.approx(law(DEO, *want.shape)) and got.sample.route == want.sample.route
        assert got.identity == (want.model, 'rs', *want.shape) and got.correctness_pass is True
    assert len(grouped(loaded, DEO, 'rs')) == len(rows) and grouped(loaded, DEO, 'ag') == []
    assert op_of('ag_gu') == 'ag' and op_of('rs') == 'rs'
    with pytest.raises(ValueError):
        op_of('a2a')


def test_failed_rows_are_kept_and_excluded(tmp_path):
    rows = records(DEO, 'rs', families=tuple(f'fam-{i}' for i in range(4)))
    lines = [sample_line(r) for r in rows]
    lines[0] = dict(lines[0], candidate='triton_dist', status='unsupported',      # the shape is not runnable
                    support={'status': 'unsupported', 'evidence': 'needs M % 256'})
    lines[1] = dict(lines[1], status='error', error='boom')                        # it crashed
    lines[2] = dict(lines[2], correctness={'checked': True, 'pass': False, 'tol': 0.02,
                                           'rel_vs_serial': 0.5})                  # it ran, wrongly
    path = write_samples([], tmp_path / 'a.jsonl', lines)
    loaded, failures = load_samples([path])
    assert [f['status'] for f in failures] == ['unsupported', 'error']      # never a 0, never dropped
    assert len(loaded) == len(rows) - 2 and len(usable(loaded)) == len(rows) - 3
    bad = next(r for r in loaded if r.shape == rows[2].shape and r.family == rows[2].family)
    assert bad.correctness_pass is False and bad.origin.endswith(':3')
    assert all(r.correctness_pass is not False for r in usable(loaded))     # a failed check is not training material


def test_malformed_input_is_refused(tmp_path):
    rows = records(DEO, 'rs', families=tuple(f'fam-{i}' for i in range(3)))
    with pytest.raises(ValueError, match='duplicate'):
        load_samples([write_samples(rows + rows[:1], tmp_path / 'dup.jsonl')])
    with pytest.raises(ValueError, match='unreadable'):
        path = write_samples(rows, tmp_path / 'cut.jsonl')
        open(path, 'a').write('{"kind": "oper')                           # a crashed writer's partial last line
        load_samples([path])
    with pytest.raises(ValueError, match='schema'):
        record_from_dict(dict(sample_line(rows[0]), schema='something-else'), 'x')
    with pytest.raises(ValueError, match='does not match'):
        record_from_dict(dict(sample_line(rows[0]), candidate='async_tp'), 'x')   # route says deo
    line = sample_line(rows[0])
    line['serial_median_ms'] = line.pop('serial_reference_median_ms')              # the bench's key is the long one
    with pytest.raises(KeyError):
        record_from_dict(line, 'x')


def test_anomaly_switch(tmp_path):
    rows = records(DEO, 'rs', families=tuple(f'fam-{i}' for i in range(4)))
    loaded, _ = load_samples([write_samples(rows, tmp_path / 'a.jsonl')])
    loaded[0] = type(loaded[0])(**{**loaded[0].__dict__, 'anomaly': 'legacy-anomaly:x'})
    assert len(usable(loaded)) == len(rows) and len(usable(loaded, exclude_anomaly=True)) == len(rows) - 1
    site = type(loaded[1])(**{**loaded[1].__dict__, 'kind': 'site', 'site': 'fc2'})
    assert len(usable(loaded + [site])) == len(rows)                            # a call site is never training material


def _legacy(tmp_path, cells):
    path = tmp_path / 'cells.json'
    path.write_text(json.dumps(cells))
    return load_legacy_cells(path, tp=8, topology='BW8', versions=VERSIONS)


def test_legacy_cells_become_deo_records_with_the_legacy_route(tmp_path):
    cells = [{'model': 'llama3-8b', 'op': 'ag_gu', 'M': 8192, 'N': 3584, 'K': 4096, 'stock_b2b': 2.0, 'dx_b2b': 1.8},
             {'model': 'llama3-8b', 'op': 'rs', 'M': 8192, 'N': 4096, 'K': 1792, 'stock_b2b': 2.0, 'dx_b2b': 2.4,
              'anomaly': True}]
    out = _legacy(tmp_path, cells)
    assert [(r.op, r.method, r.protocol, r.table_op, r.kind) for r in out] == [
        ('ag', DEO, LEGACY, 'ag_gu', 'operator'), ('rs', DEO, LEGACY, 'rs', 'operator')]
    assert out[0].sample.route == legacy_deo_route('ag', tp=8, topology='BW8', versions=VERSIONS)
    assert out[0].sample.route.warmup is None and out[0].sample.route.timing_mode == 'b2b16_perop_events'
    assert out[0].log_speedup == pytest.approx(math.log(2.0 / 1.8))
    assert out[1].anomaly == 'legacy-anomaly:llama3-8b/rs/M8192' and out[0].anomaly is None
    assert out[0].correctness_pass is None                                  # the old table recorded no check


def test_the_real_legacy_table_loads():
    try:
        open(LEGACY_TABLE)
    except OSError:                                                        # pragma: no cover
        pytest.skip('planner/data/cells.json not available')
    out = load_legacy_cells(LEGACY_TABLE, tp=8, topology='BW8', versions=VERSIONS)
    assert len(out) == 99 and len({r.family for r in out}) == 11
    assert sum(r.anomaly is not None for r in out) == 2 and len(usable(out, exclude_anomaly=True)) == 97
    assert {r.table_op for r in out} == {'rs', 'ag_gu', 'ag_up'} and {r.sample.shape[0] for r in out} == {
        8192, 16384, 32768}


def test_two_data_classes_are_never_mixed(tmp_path):
    from stowaway.oprs.evaluate import evaluate
    from stowaway.oprs.export_plan import fit_routes
    new = records(DEO, 'rs')
    with pytest.raises(ValueError, match='one data class'):
        fit_routes(new + _legacy(tmp_path, [{'model': 'm', 'op': 'rs', 'M': 8192, 'N': 4096, 'K': 1536,
                                             'stock_b2b': 2.0, 'dx_b2b': 1.5}]))
    old = _legacy(tmp_path, [{'model': 'm', 'op': 'rs', 'M': 8192, 'N': 4096, 'K': 1536,
                              'stock_b2b': 2.0, 'dx_b2b': 1.5}])
    with pytest.raises(ValueError, match='one data class'):
        evaluate(new + old, (DEO,))
