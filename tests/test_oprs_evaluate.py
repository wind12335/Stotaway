"""Family-holdout evaluation (evaluate.py): the operator-level number that says how good the predictor is."""
from collections import Counter
import json
import sys

import pytest

from oprs_synthetic import ASYNC_TP, DEO, FAMILIES, TRITON_DIST, every_record, grows_with_m, records
from stowaway.oprs import evaluate as E
from stowaway.oprs.data import LEGACY, NEW, load_legacy_cells
from stowaway.oprs.export_plan import LEGACY_LIMITATION

LEGACY_TABLE = 'planner/data/cells.json'
POINTS = 11 * 9                                   # 11 families x (ag_gu, ag_up, rs) x 3 values of M


def all_(summary, policy):
    return summary[policy]['all']


def test_holdout_is_exact_on_a_world_that_follows_its_law():
    result = E.evaluate(every_record(8), (DEO, TRITON_DIST))
    summary = result['summary']
    assert result['classification'] == 'family_holdout_b2b16' and result['data_class'] == NEW
    assert result['denominators']['points_with_all_candidates'] == POINTS
    assert result['denominators']['points_missing_a_candidate'] == []
    assert (all_(summary, 'oprs')['count'], all_(summary, 'oprs')['accuracy']) == (POINTS, 1.0)
    assert all_(summary, 'oprs')['selected_below_stock'] == 0 and all_(summary, 'oprs')['regret_gmean_ratio'] == 1.0
    assert all_(summary, 'always_deo')['accuracy'] == 1.0                  # deo wins everywhere in this world
    assert all_(summary, 'always_triton_dist')['correct'] == 0 and all_(summary, 'oracle')['accuracy'] == 1.0
    assert (summary['oprs']['ag']['count'], summary['oprs']['rs']['count']) == (66, 33)
    assert all(row['choices']['oprs'] == DEO == row['oracle_choice'] for row in result['rows'])
    assert all(row['below_stock']['oprs'] is False for row in result['rows'])


def test_the_choice_follows_the_shape_not_the_method():
    """A backend that loses at M=8192, nearly ties at 16384 and wins at 32768: the plan must take it only
    at 32768, on families it has never seen."""
    rows = records(DEO, 'ag') + records(DEO, 'rs') + records(TRITON_DIST, 'ag', gain=grows_with_m) \
        + records(TRITON_DIST, 'rs', gain=grows_with_m)
    result = E.evaluate(rows, (DEO, TRITON_DIST))
    picks = {m: {row['choices']['oprs'] for row in result['rows'] if row['M'] == m} for m in (8192, 16384, 32768)}
    assert picks == {8192: {DEO}, 16384: {DEO}, 32768: {TRITON_DIST}}
    assert all_(result['summary'], 'oprs')['accuracy'] == 1.0
    assert all_(result['summary'], 'always_deo')['accuracy'] == pytest.approx(66 / 99)
    assert all_(result['summary'], 'always_triton_dist')['accuracy'] == pytest.approx(33 / 99)
    assert all_(result['summary'], 'always_triton_dist')['selected_below_stock'] == 66   # a loss at two of three M


def test_the_held_out_family_never_trains_its_own_fit(monkeypatch):
    """The holdout unit is the model family, the key the training rows are filtered by. (A model name is
    not a family: here family 'fam-i' is measured as model 'model-fam-i'.) Every fit must miss exactly
    one family, and each family must be the missing one of every fit made while it was held out."""
    seen = []

    class Spy(E.RouteLocalResidual):
        def fit(self, rows):
            seen.append(frozenset(r.family for r in rows))
            return super().fit(rows)

    monkeypatch.setattr(E, 'RouteLocalResidual', Spy)
    rows = every_record(8)
    assert {r.model for r in rows if r.family == FAMILIES[0]} == {f'model-{FAMILIES[0]}'} != {FAMILIES[0]}
    E.evaluate(rows, (DEO, TRITON_DIST))
    assert len(seen) == 2 * 2 * 11                              # 2 directions x 2 methods x 11 held-out families
    assert all(len(s) == 10 for s in seen)
    assert Counter(next(iter(set(FAMILIES) - s)) for s in seen) == Counter({f: 4 for f in FAMILIES})


def test_a_backend_that_always_loses_is_never_picked_and_its_arm_is_marked():
    summary = E.evaluate(every_record(8), (DEO, ASYNC_TP))['summary']
    arm = all_(summary, f'always_{ASYNC_TP}')
    assert arm['accuracy'] == 0.0 and arm['selected_below_stock'] == POINTS and arm['selected_gmean_speedup'] < 1.0
    assert all_(summary, 'oprs')['accuracy'] == 1.0 and all_(summary, 'base_ridge')['accuracy'] == 1.0


def test_what_cannot_be_evaluated_is_refused_or_counted():
    # five families at least: four to train on while one is held out (family is not the model name here)
    with pytest.raises(ValueError, match='four training families required to hold out'):
        E.evaluate(every_record(8, families=tuple(f'fam-{i}' for i in range(4))), (DEO, TRITON_DIST))
    partial = records(DEO, 'ag') + records(TRITON_DIST, 'ag', ms=(32768,)) + records(DEO, 'rs') \
        + records(TRITON_DIST, 'rs')
    result = E.evaluate(partial, (DEO, TRITON_DIST))
    assert result['denominators']['points_with_all_candidates'] == 22 + 33      # only where both were measured
    assert len(result['denominators']['points_missing_a_candidate']) == 44
    assert all(row['M'] == 32768 for row in result['rows'] if row['op'] == 'ag')


def test_policies_and_anomalies():
    rows = E.usable(every_record(8))
    flagged = [type(r)(**{**r.__dict__, 'anomaly': 'legacy-anomaly:x'}) if i == 0 else r for i, r in enumerate(rows)]
    kept, dropped = E.evaluate(flagged, (DEO, ASYNC_TP)), E.evaluate(flagged, (DEO, ASYNC_TP), exclude_anomaly=True)
    assert kept['denominators']['anomalies_included'] == ['legacy-anomaly:x']
    assert kept['denominators']['points_with_all_candidates'] == POINTS
    assert dropped['denominators']['points_with_all_candidates'] == POINTS - 1
    assert dropped['denominators']['anomalies_excluded'] == ['legacy-anomaly:x']
    plain = E.evaluate(every_record(8), (DEO, ASYNC_TP), with_route_policy=False)
    assert 'route_policy' not in plain['summary'] and set(E.LEARNED) == {'base_ridge', 'route_policy', 'oprs'}
    assert set(plain['summary']) == {f'always_{DEO}', f'always_{ASYNC_TP}', 'oracle', 'base_ridge', 'oprs'}
    assert set(plain['policies']) == set(E.DESCRIPTION)


def test_choice_rows_and_the_exclusive_csv(tmp_path):
    result = E.evaluate(every_record(8), (DEO, TRITON_DIST), sources=[{'path': 'a', 'sha256': 'b'}],
                        limitation='unit test')
    assert result['sources'] == [{'path': 'a', 'sha256': 'b'}] and result['limitation'] == 'unit test'
    flat = E.choice_rows(result)
    policies = {r['policy'] for r in flat}
    assert len(flat) == POINTS * len(policies) and 'oracle' in policies and 'oprs' in policies
    row = next(r for r in flat if r['policy'] == 'oprs')
    assert (row['data_class'], row['classification'], row['choice']) == (NEW, 'family_holdout_b2b16', DEO)
    assert f'predicted_log_speedup_{DEO}' in row and f'actual_speedup_{TRITON_DIST}' in row
    for r in flat:                                   # the fixed candidates and the oracle have no prediction
        if r['policy'].startswith('always') or r['policy'] == 'oracle':
            assert r[f'predicted_log_speedup_{DEO}'] is None and r[f'predicted_log_speedup_{TRITON_DIST}'] is None
    path = E.write_csv_exclusive(tmp_path / 'choices.csv', flat)
    assert sum(1 for _ in open(path)) == len(flat) + 1
    with pytest.raises(FileExistsError):
        E.write_csv_exclusive(path, flat)


def test_legacy_replay_keeps_its_label():
    """The historical table under the ported predictor: `historical_replay_legacy`, never averaged with
    a b2b16 number. 81 of 97 with the two anomalies excluded is the figure of the NVIDIA side's
    evaluation of the same table (A800 software stack assumed for the old rows)."""
    try:
        open(LEGACY_TABLE)
    except OSError:                                                    # pragma: no cover
        pytest.skip('planner/data/cells.json not available')
    from oprs_synthetic import VERSIONS
    cells = load_legacy_cells(LEGACY_TABLE, tp=8, topology='BW8', versions=VERSIONS)
    kept = E.evaluate(cells, (DEO,), sources=[], limitation=LEGACY_LIMITATION)
    assert kept['classification'] == 'historical_replay_legacy' and kept['data_class'] == LEGACY
    assert (all_(kept['summary'], 'oprs')['count'], all_(kept['summary'], 'oprs')['correct']) == (99, 84)
    dropped = E.evaluate(cells, (DEO,), exclude_anomaly=True)
    oprs, oracle = all_(dropped['summary'], 'oprs'), all_(dropped['summary'], 'oracle')
    assert (oprs['count'], oprs['correct'], oprs['accuracy']) == (97, 81, 81 / 97)
    assert oracle['accuracy'] == 1.0 and oprs['selected_gmean_speedup'] > 1.0
    assert dropped['denominators']['anomalies_excluded'] == [
        'legacy-anomaly:llama4-mav-d/ag_gu/M32768', 'legacy-anomaly:qwen3-8b/rs/M32768']
    assert all_(dropped['summary'], f'always_{DEO}')['selected_below_stock'] > oprs['selected_below_stock']


def test_cli(tmp_path, capsys, monkeypatch):
    from oprs_synthetic import write_samples
    data = write_samples(every_record(8), tmp_path / 'samples.jsonl')
    out, csv = tmp_path / 'holdout.json', tmp_path / 'choices.csv'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--data', data, '--tp', '8', '--topology', 'BW8', '--output',
                                       str(out), '--csv', str(csv)])
    E.main()
    assert json.loads(out.read_text())['classification'] == 'family_holdout_b2b16'
    assert 'acc=1.000' in capsys.readouterr().out
    with pytest.raises(FileExistsError):                                  # history is never overwritten
        E.main()
