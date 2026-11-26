"""Plan export (export_plan.py) and the plan-set generator of one deployment (tools/oprs_plans.py)."""
import hashlib
import json
import math
import os

import pytest

from oprs_synthetic import ASYNC_TP, DEO, TRITON_DIST, FAMILIES, every_record, records, write_samples
from stowaway.oprs import export_plan as X
from stowaway.oprs.auto_plan import ENTRY_KEYS, AutoPlan
from stowaway.oprs.data import LEGACY, NEW
from stowaway.oprs.export_plan import LEGACY_LIMITATION, NEW_LIMITATION, build_plan, fit_routes
from stowaway.oprs.platform import runtime_platform
from stowaway.oprs.predictor import SERIAL
from stowaway.oprs.workloads import build_workload

LEGACY_TABLE = 'planner/data/cells.json'
COMMIT = {'commit': 'test', 'src_dirty': False}                # the tests do not depend on the git state


def make(rows=None, backend=TRITON_DIST, workload='bw27b_train', tp=8, config=None, fitted=None, **kw):
    rows = every_record(tp) if rows is None else rows
    platform = runtime_platform(tp=tp, topology=f'BW{tp}', probe_device=False)
    return build_plan(build_workload(workload, tp, config=config), tp, backend=backend, platform=platform,
                      dtype='bfloat16', records=rows, fitted=fit_routes(rows) if fitted is None else fitted,
                      sources=[], data_class=NEW, limitation='synthetic test', commit=COMMIT, **kw)


def choices(plan):
    return {e['site']: e['choice'] for e in plan['entries']}


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    """Synthetic samples of the three methods as one bench JSONL file."""
    from oprs_synthetic import write_samples
    return write_samples(every_record(8), tmp_path_factory.mktemp('data') / 'samples_tp8_bfloat16.jsonl')


def test_fit_routes_covers_every_route_and_needs_four_families():
    fitted = fit_routes(every_record(8))
    assert sorted(fitted) == sorted((m, op) for m in (DEO, ASYNC_TP, TRITON_DIST) for op in ('ag', 'rs'))
    assert fitted[(DEO, 'rs')].route.tp == 8 and fitted[(ASYNC_TP, 'ag')].route.method == ASYNC_TP
    with pytest.raises(ValueError, match='four model families'):
        fit_routes(every_record(8, families=FAMILIES[:3]))


def test_predicted_plan_takes_the_best_measured_method_per_site():
    rows = records(DEO, 'rs', gain=0.3) + records(DEO, 'ag', gain=-0.1) + records(TRITON_DIST, 'ag') \
        + records(TRITON_DIST, 'rs')
    plan = make(rows)
    assert choices(plan) == {'qkv': TRITON_DIST, 'proj': DEO, 'fc1': TRITON_DIST, 'fc2': DEO}
    by = {e['site']: e for e in plan['entries']}
    # the rs route saw only one (N, K): its model knows the M trend, shrunk by the ridge, and nothing else
    assert by['proj']['predicted_log_speedup'] == pytest.approx(0.3 + 0.03 * math.log(4) * 33 / 35, abs=2e-3)
    assert by['qkv']['scores'][SERIAL] == 0.0 and set(by['qkv']['scores']) == {SERIAL, DEO, TRITON_DIST}
    assert all(set(e) == ENTRY_KEYS for e in plan['entries'])
    assert plan['external_backend'] == TRITON_DIST and plan['candidates'] == [SERIAL, DEO, TRITON_DIST]
    assert plan['allocate'] == sorted([[TRITON_DIST, 'ag', 32768, 896, 5120], [DEO, 'rs', 32768, 5120, 640],
                                       [TRITON_DIST, 'ag', 32768, 4352, 5120], [DEO, 'rs', 32768, 5120, 2176]])
    assert plan['provenance']['arm'] == 'oprs' and plan['provenance']['n_samples'] == 11 * 9 * 2
    assert [o['origin'] for o in plan['provenance']['origins']] == ['qkv', 'proj', 'fc1', 'fc2']


def test_a_tie_is_serial():
    class Flat:
        route = every_record(8)[0].sample.route

        def predict(self, route, shape):
            return {'predicted_log_speedup': 0.0}

        def to_dict(self):
            return {}

    flat = {(m, op): Flat() for m in (DEO, TRITON_DIST) for op in ('ag', 'rs')}
    plan = make(fitted=flat)
    assert set(choices(plan).values()) == {SERIAL} and plan['allocate'] == []


def test_constant_arms_and_the_two_way_plan():
    always = make(force=DEO)
    assert set(choices(always).values()) == {DEO} and always['provenance']['arm'] == 'always_deo'
    assert {tuple(x) for x in always['allocate']} == {('deo', 'ag', 32768, 896, 5120),
                                                     ('deo', 'rs', 32768, 5120, 640),
                                                     ('deo', 'ag', 32768, 4352, 5120),
                                                     ('deo', 'rs', 32768, 5120, 2176)}
    forced = make(force=TRITON_DIST)
    assert [e['choice'] for e in forced['entries']] == [TRITON_DIST, SERIAL, TRITON_DIST, SERIAL]
    assert forced['provenance']['arm'] == f'always_{TRITON_DIST}' and len(forced['allocate']) == 2
    two_way = make(backend='none')
    assert two_way['candidates'] == [SERIAL, DEO] and set(choices(two_way).values()) == {DEO}
    assert {x[0] for x in two_way['allocate']} == {DEO}


def test_an_unmeasured_method_is_never_a_candidate():
    """A path that was not measured cannot be a candidate, however good it looks."""
    plan = make(records(DEO, 'ag', gain=-0.2) + records(DEO, 'rs', gain=-0.2), backend=ASYNC_TP)
    assert set(choices(plan).values()) == {SERIAL} and plan['allocate'] == []
    assert plan['provenance']['unmeasured'] == [f'{ASYNC_TP}/ag', f'{ASYNC_TP}/rs']
    assert all(set(e['scores']) == {SERIAL, DEO} for e in plan['entries'])          # async_tp has no score
    assert ASYNC_TP not in {a[0] for a in plan['allocate']}


def test_null_arm_borrows_the_contexts_of_its_plan(tmp_path):
    predicted = make(records(DEO, 'ag', gain=-0.2) + records(DEO, 'rs', gain=-0.2)
                     + records(TRITON_DIST, 'ag') + records(TRITON_DIST, 'rs'))
    assert {x[0] for x in predicted['allocate']} == {TRITON_DIST}
    source = X.write_json_exclusive(tmp_path / 'predicted.json', predicted)
    null = make(records(DEO, 'ag'), force=SERIAL, allocate_from=source)
    assert null['external_backend'] == TRITON_DIST and set(choices(null).values()) == {SERIAL}
    assert null['allocate'] == predicted['allocate']
    assert null['provenance']['arm'] == 'null' and null['provenance']['data_class'] == 'derived'
    assert null['provenance']['allocate_from']['sha256'] == hashlib.sha256(open(source, 'rb').read()).hexdigest()
    assert 'Null arm' in null['provenance']['limitation']
    assert AutoPlan(str(tmp_path / 'predicted.json'), 'bw27b_train', tp=8,
                    runtime_platform=runtime_platform(tp=8, topology='BW8', probe_device=False)).backend


def test_out_of_range_sites_are_recorded():
    """qkv / proj / fc2 of the vehicle are outside the span of the synthetic samples: their scores are
    extrapolations of a log-linear model, and the plan says which dimension is out of range."""
    plan = make()
    out = {(o['site'], o['method']): ''.join(o['outside']) for o in plan['provenance']['out_of_range']}
    # the synthetic table has N in [1536, 3072] (ag) / 4096 (rs) and K = 4096 (ag) / 1536 (rs)
    # (triton_dist is not a candidate at proj / fc2 of this vehicle: demoted before it is scored)
    assert out == {('qkv', DEO): 'NK', ('qkv', TRITON_DIST): 'NK', ('proj', DEO): 'NK',
                   ('fc1', DEO): 'NK', ('fc1', TRITON_DIST): 'NK', ('fc2', DEO): 'NK'}
    assert all(set(o) == {'site', 'origin', 'op', 'M', 'N', 'K', 'method', 'outside', 'predicted_log_speedup'}
               for o in plan['provenance']['out_of_range'])
    inside = make(every_record(2), workload='custom_train', tp=2, backend=ASYNC_TP,
                  config='{"hidden":4096,"ffn":3072,"heads":48,"kv_groups":4,"head_dim":64}')
    assert inside['provenance']['out_of_range'] == [] and set(choices(inside).values()) == {DEO}


def test_write_is_exclusive_and_the_file_ends_in_the_loader(tmp_path):
    plan = make()
    path = X.write_json_exclusive(tmp_path / 'p.json', plan)
    text = path.read_text()
    assert 'NaN' not in text and 'Infinity' not in text and text.endswith('\n')
    with pytest.raises(FileExistsError):
        X.write_json_exclusive(path, plan)
    with pytest.raises(ValueError):
        X.write_json_exclusive(tmp_path / 'nan.json', {**plan, 'x': float('nan')})
    loaded = AutoPlan(str(path), 'bw27b_train', tp=8,
                      runtime_platform=runtime_platform(tp=8, topology='BW8', probe_device=False))
    assert [loaded.entries[(e['site'], e['op'], e['M'], e['N'], e['K'])] for e in plan['entries']] == [
        e['choice'] for e in plan['entries']]


def test_git_commit_is_recorded():
    state = X.git_commit()
    assert set(state) == {'commit', 'src_dirty'}
    assert state['commit'] == 'unknown' or (len(state['commit']) == 40 and int(state['commit'], 16) >= 0)
    assert state['src_dirty'] in (True, False, None)


def test_cli_predicted_constant_and_null_plans(tmp_path, data, capsys):
    common = ['--workload', 'bw27b_train', '--tp', '8']
    out = tmp_path / 'oprs_triton_dist.json'
    X.main([*common, '--backend', 'triton_dist', '--data', data, '--output', str(out), '--note', 'n'])
    plan = json.loads(out.read_text())
    assert plan['provenance']['note'] == 'n' and plan['provenance']['data_class'] == NEW
    assert plan['provenance']['limitation'] == NEW_LIMITATION
    assert plan['provenance']['sources'] == [
        {'path': os.path.realpath(data), 'sha256': hashlib.sha256(open(data, 'rb').read()).hexdigest()}]
    assert set(plan['provenance']['overheads']) == {'fit_s', 'export_s'}
    assert plan['provenance']['overheads']['fit_s'] >= 0 and plan['provenance']['overheads']['export_s'] >= 0
    assert plan['platform']['topology'] == 'BW8' and 'commit' in plan['provenance']
    assert 'src_dirty' in plan['provenance'] and 'fit_s=' in capsys.readouterr().out
    with pytest.raises(FileExistsError):                              # never overwritten
        X.main([*common, '--backend', 'triton_dist', '--data', data, '--output', str(out)])
    always = tmp_path / 'always_deo.json'
    X.main([*common, '--backend', 'triton_dist', '--force', 'deo', '--data', data, '--output', str(always)])
    assert json.loads(always.read_text())['provenance']['arm'] == 'always_deo'
    null = tmp_path / 'null.json'
    X.main([*common, '--force', 'serial', '--allocate-from', str(out), '--output', str(null)])
    assert 'external backend triton_dist taken from' in capsys.readouterr().out      # no --backend given
    borrowed = json.loads(null.read_text())
    assert borrowed['external_backend'] == TRITON_DIST and borrowed['allocate'] == plan['allocate']


def test_cli_refuses_what_cannot_be_a_plan(tmp_path, data):
    common = ['--workload', 'bw27b_train', '--tp', '8']
    deo_only = tmp_path / 'deo_only.jsonl'
    write_samples(records(DEO, 'ag') + records(DEO, 'rs'), deo_only)
    with pytest.raises(SystemExit):                                   # no measured async_tp sample in the data
        X.main([*common, '--backend', 'async_tp', '--data', str(deo_only), '--output', str(tmp_path / 'a.json')])
    with pytest.raises(SystemExit):                                   # a predicted plan needs data
        X.main([*common, '--backend', 'triton_dist', '--output', str(tmp_path / 'b.json')])
    with pytest.raises(SystemExit):                                   # --allocate-from is the Null arm only
        X.main([*common, '--allocate-from', str(data), '--output', str(tmp_path / 'c.json')])
    with pytest.raises(SystemExit):
        X.main(['--workload', 'nope', '--tp', '8', '--output', str(tmp_path / 'd.json'), '--force', 'serial'])
    X.main([*common, '--backend', 'async_tp', '--force', 'serial', '--output', str(tmp_path / 'e.json')])   # Null
    assert json.loads((tmp_path / 'e.json').read_text())['allocate'] == []


def test_cli_legacy_plan_is_labelled_legacy(tmp_path):
    cells = [{'model': f'old-{i}', 'op': op, 'M': m, 'N': 4096 if op == 'rs' else 3072,
              'K': 1536 if op == 'rs' else 4096, 'stock_b2b': 2.0, 'dx_b2b': 2.0 / math.exp(0.1)}
             for i in range(5) for op in ('rs', 'ag_gu') for m in (8192, 16384, 32768)]
    legacy = tmp_path / 'cells.json'
    legacy.write_text(json.dumps(cells))
    out = tmp_path / 'legacy.json'
    X.main(['--legacy', str(legacy), '--workload', 'bw27b_train', '--tp', '4', '--topology', 'BW4',
            '--output', str(out)])
    plan = json.loads(out.read_text())
    assert plan['provenance']['data_class'] == LEGACY and plan['provenance']['limitation'] == LEGACY_LIMITATION
    assert set(choices(plan).values()) == {DEO} and plan['platform']['topology'] == 'BW4'
    assert plan['provenance']['routes'][0]['timing_mode'] == 'b2b16_perop_events'       # never a b2b16 route


def test_the_real_legacy_table_exports_an_interim_plan(tmp_path):
    try:
        open(LEGACY_TABLE)
    except OSError:                                                    # pragma: no cover
        pytest.skip('planner/data/cells.json not available')
    out = tmp_path / 'legacy.json'
    X.main(['--legacy', LEGACY_TABLE, '--workload', 'bw27b_train', '--tp', '8', '--output', str(out)])
    plan = json.loads(out.read_text())
    assert plan['provenance']['data_class'] == LEGACY and plan['provenance']['n_samples'] == 99
    assert plan['workload'] == 'bw27b_train'
    assert 're-measure under the b2b16 protocol' in plan['provenance']['limitation']


def test_plan_set_of_one_deployment(tmp_path, scripts, data, capsys):
    plans = scripts('oprs_plans', 'tools', 'oprs_plans.py')
    out = tmp_path / 'bw27b_async_tp'
    plans.main([str(out), '--tp', '8', '--backend', 'async_tp', '--workload', 'bw27b_train', '--data', data,
                '--note', 'plumbing'])
    assert sorted(os.listdir(out)) == sorted([
        'always_async_tp.json', 'always_deo.json', 'export.log', 'manifest.json', 'null.json',
        'oprs_async_tp.json', 'oprs_deo_only.json'])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['schema'] == 'stowaway-oprs-plan-set-v1' and manifest['backend'] == ASYNC_TP
    assert manifest['data_class'] == NEW and manifest['workload_args'] == {
        'seq': 4096, 'mbs': 8, 'rows': None, 'config': None}
    platform = runtime_platform(tp=8, topology='BW8', probe_device=False)
    for arm, item in manifest['plans'].items():
        path = out / item['path']
        assert item['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        plan = json.loads(path.read_text())
        assert item['arm'] == plan['provenance']['arm'] and item['choices'] == choices(plan)
        assert AutoPlan(str(path), 'bw27b_train', tp=8, runtime_platform=platform).backend == (
            'none' if arm == 'oprs_deo_only' else ASYNC_TP)
    assert set(manifest['plans']) == {'oprs', 'oprs_deo_only', 'always_deo', f'always_{ASYNC_TP}', 'null'}
    assert all(set(item['overheads']) == {'fit_s', 'export_s'} for item in manifest['plans'].values())
    printed = capsys.readouterr().out
    assert f'backend={ASYNC_TP} workload=bw27b_train data_class={NEW}' in printed and 'null.json' in printed
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        plans.main([str(out), '--tp', '8', '--backend', 'async_tp', '--workload', 'bw27b_train', '--data', data])
    for bad in [[], ['--backend', 'flux'], ['--workload', 'nope']]:           # a source is required; choices
        with pytest.raises(SystemExit):
            plans.main([str(tmp_path / 'x'), '--tp', '8', '--workload', 'bw27b_train', '--data', data, *bad])
    custom = tmp_path / 'set'
    plans.main([str(custom), '--tp', '1', '--backend', TRITON_DIST, '--workload', 'custom_train', '--seq', '4096',
                '--mbs', '1', '--config', '{"hidden":2048,"ffn":2048,"heads":16,"kv_groups":4}',
                '--data', write_samples(every_record(1), tmp_path / 'tp1.jsonl')])
    assert json.loads((custom / 'manifest.json').read_text())['workload_args']['mbs'] == 1
    assert {e['M'] for e in json.loads((custom / 'oprs_triton_dist.json').read_text())['entries']} == {4096}
