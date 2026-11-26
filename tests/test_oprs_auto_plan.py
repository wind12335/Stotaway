"""Runtime plan loader (auto_plan.AutoPlan): what the dispatcher trusts before it allocates anything."""
import copy
import hashlib
import json

import pytest

from oprs_synthetic import ASYNC_TP, DEO, TRITON_DIST, every_record, records
from stowaway.oprs.auto_plan import ENTRY_KEYS, REASONS, SCHEMA, AutoPlan
from stowaway.oprs.export_plan import build_plan, fit_routes
from stowaway.oprs.platform import runtime_platform
from stowaway.oprs.predictor import SERIAL
from stowaway.oprs.workloads import build_workload

TP = 8
PLATFORM = runtime_platform(tp=TP, topology=f'BW{TP}', probe_device=False)


def write(data, tmp_path, name='plan.json'):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def plan_dict(backend=TRITON_DIST, **kw):
    """A plan as export_plan builds it from the synthetic samples (b2b16, bw27b_train, TP=8)."""
    records = kw.pop('records', None) or every_record(TP)
    return build_plan(build_workload('bw27b_train', TP), TP, backend=backend, platform=PLATFORM,
                      dtype='bfloat16', records=records, fitted=fit_routes(records), sources=[],
                      data_class='b2b16', limitation='synthetic test plan', **kw)


def load(data, tmp_path, workload='bw27b_train', tp=TP, name='plan.json'):
    return AutoPlan(write(data, tmp_path, name), workload, tp=tp, runtime_platform=PLATFORM)


def test_exported_plan_loads_and_dispatches_as_written(tmp_path):
    data = plan_dict()
    assert data['schema'] == SCHEMA and data['default'] == 'serial' and data['candidates'] == [
        'serial', DEO, TRITON_DIST]
    assert all(set(e) == ENTRY_KEYS for e in data['entries'])          # exactly the loader's schema
    plan = load(data, tmp_path)
    assert plan.digest == hashlib.sha256(open(write(data, tmp_path, 'again.json'), 'rb').read()).hexdigest()
    assert plan.backend == TRITON_DIST and plan.methods == (DEO, TRITON_DIST) and plan.dtype == 'bfloat16'
    for e in data['entries']:
        assert plan.choose(e['site'], e['op'], e['M'], e['N'], e['K'], 'bfloat16') == (e['choice'], 'plan')
    assert sorted(plan.allocate) == sorted(tuple(x) for x in data['allocate'])


def test_fallbacks_are_explicit_and_counted_by_reason(tmp_path):
    plan = load(plan_dict(), tmp_path)
    assert plan.choose('fc2', 'rs', 999, 1, 1, 'bfloat16') == ('serial', 'unknown_shape')
    assert plan.choose('fc2', 'rs', 32768, 5120, 2176, 'float16') == ('serial', 'dtype_mismatch')
    assert set(REASONS) == {'plan', 'unknown_shape', 'dtype_mismatch'} and plan.default == 'serial'
    with pytest.raises(ValueError):
        plan.choose('nope', 'rs', 1, 1, 1, 'bfloat16')
    with pytest.raises(ValueError):
        plan.choose('fc2', 'a2a', 1, 1, 1, 'bfloat16')


def test_incompatible_plans_are_refused(tmp_path):
    good = plan_dict()
    for change in [{'default': 'stock'},                     # the NVIDIA vocabulary never loads here
                   {'external_backend': 'flux'},             # not an integrated backend
                   {'tp': 4}, {'dtype': 'float32'}, {'schema': 'x'}]:
        with pytest.raises(ValueError, match='incompatible'):
            load({**copy.deepcopy(good), **change}, tmp_path)
    with pytest.raises(ValueError, match='incompatible'):
        load(copy.deepcopy(good), tmp_path, workload='bw70b_train')       # another workload


def test_an_nvidia_plan_never_loads(tmp_path):
    """The rule of the handoff: NVIDIA numbers and choices are never dispatched on BW1000."""
    nvidia = copy.deepcopy(plan_dict())
    nvidia['default'] = 'stock'                                           # upstream vocabulary: stock / dx
    for e in nvidia['entries']:
        e['choice'] = 'dx' if e['choice'] == DEO else 'stock'
    nvidia['platform'].update(vendor='nvidia', device='A800', comm='nccl', symmetric='nvshmem')
    with pytest.raises(ValueError):
        load(nvidia, tmp_path)
    same_words = copy.deepcopy(nvidia)                                    # our words, their platform: still refused
    same_words['default'] = 'serial'
    for e in same_words['entries']:
        e['choice'] = DEO if e['choice'] == 'dx' else 'serial'
    same_words['external_backend'] = 'none'
    with pytest.raises(ValueError, match='another platform'):
        load(same_words, tmp_path)
    with pytest.raises(TypeError):
        AutoPlan(write(plan_dict(), tmp_path), 'bw27b_train', tp=TP)       # the runtime platform must be given


def test_platform_mismatch_is_a_hard_error(tmp_path):
    good = plan_dict()
    for key, value in [('vendor', 'nvidia'), ('device', 'A800'), ('comm', 'nccl'), ('symmetric', 'nvshmem'),
                       ('topology', 'BW4')]:
        data = copy.deepcopy(good)
        data['platform'][key] = value
        with pytest.raises(ValueError, match='another platform'):
            load(data, tmp_path)
    data = copy.deepcopy(good)
    data['platform']['versions'] = [['torch', '2.0.0']]                   # another software stack
    with pytest.raises(ValueError, match='versions'):
        load(data, tmp_path)
    moved = copy.deepcopy(good)
    moved['platform'].pop('device_name')                                  # informational fields are free
    assert load(moved, tmp_path).backend == TRITON_DIST


def with_entry(change, **kw):
    data = plan_dict(**kw)
    data['entries'][0].update(change)
    return data


def test_invalid_entries_are_refused(tmp_path):
    for change in [{'origin': 'qkv'},                     # a key outside the schema
                   {'site': 'mlp'}, {'op': 'rs'},         # qkv is a column-parallel (ag) site
                   {'choice': ASYNC_TP},                  # the plan does not offer that method
                   {'M': 0}, {'M': 32765}]:               # rows not divisible by TP
        with pytest.raises(ValueError, match='invalid plan entry'):
            load(with_entry(change), tmp_path)
    data = plan_dict()
    data['entries'].append(dict(data['entries'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        load(data, tmp_path)
    with pytest.raises(ValueError, match='empty'):
        load({**data, 'entries': []}, tmp_path)
    with pytest.raises(ValueError, match='does not support'):      # a non-serial entry on an unsupported shape
        load(with_entry({'choice': TRITON_DIST, 'N': 897}), tmp_path)


def test_a_method_outside_its_support_is_demoted_at_export(tmp_path):
    """triton_dist rs needs M, N and K multiples of 256. On the bw27b vehicle proj (K=640) and fc2
    (K=2176) cannot take it: `always triton_dist` leaves both serial and says so, and the loader
    agrees. That is the coverage figure of the handoff: 2 of the 4 sites of bw27b."""
    forced = plan_dict(force=TRITON_DIST)
    proj = next(e for e in forced['entries'] if e['site'] == 'proj')
    assert proj['choice'] == 'serial' and set(proj['scores']) == {SERIAL, DEO}  # deo was a candidate only
    assert [(d['site'], d['method'], d['support']) for d in forced['provenance']['demoted']] == [
        ('proj', TRITON_DIST, 'unsupported'), ('fc2', TRITON_DIST, 'unsupported')]
    plan = load(forced, tmp_path)
    assert [plan.choose(e['site'], e['op'], e['M'], e['N'], e['K'], 'bfloat16')[0]
            for e in forced['entries']] == ['triton_dist', 'serial', 'triton_dist', 'serial']
    data = copy.deepcopy(forced)
    data['entries'][1]['choice'] = TRITON_DIST                             # hand-edited: refused
    with pytest.raises(ValueError, match='does not support'):
        load(data, tmp_path)


def test_invalid_allocations_are_refused(tmp_path):
    for item in [['flux', 'ag', 32768, 896, 5120],                # not a method
                 ['async_tp', 'ag', 32768, 896, 5120],             # a method this plan does not offer
                 ['deo', 'a2a', 32768, 896, 5120], ['deo', 'ag', 32768, 896], ['deo', 'ag', 0, 896, 5120],
                 ['triton_dist', 'rs', 32767, 5120, 2176], 'deo']:
        data = plan_dict()
        data['allocate'] = [item]
        with pytest.raises(ValueError, match='invalid plan allocation'):
            load(data, tmp_path)
    data = plan_dict()
    data['allocate'] = [['triton_dist', 'rs', 32768, 5120, 640]]            # an unsupported shape
    with pytest.raises(ValueError, match='does not support'):
        load(data, tmp_path)


def test_null_plan_offers_its_backends_contexts(tmp_path):
    """The Null arm allocates the contexts of the predicted plan. A hand export without --backend used to
    offer deo only, and the loader refused the triton_dist contexts of that plan."""
    weak_deo = records(DEO, 'ag', gain=-0.2) + records(DEO, 'rs', gain=-0.2)      # the backend wins
    predicted = plan_dict(records=weak_deo + records(TRITON_DIST, 'ag') + records(TRITON_DIST, 'rs'))
    assert {x[0] for x in predicted['allocate']} == {TRITON_DIST}
    null = plan_dict(records=weak_deo, force='serial', allocate_from=write(predicted, tmp_path, 'predicted.json'))
    assert null['provenance']['arm'] == 'null' and null['provenance']['data_class'] == 'derived'
    plan = load(null, tmp_path)
    assert {e['choice'] for e in null['entries']} == {'serial'} and plan.backend == TRITON_DIST
    assert sorted(plan.allocate) == sorted(tuple(x) for x in predicted['allocate'])
    assert all(plan.choose(e['site'], e['op'], e['M'], e['N'], e['K'], 'bfloat16') == ('serial', 'plan')
               for e in null['entries'])
    data = copy.deepcopy(null)
    data['external_backend'] = 'none'                                   # what the old export wrote: not loadable
    with pytest.raises(ValueError, match='invalid plan allocation'):
        load(data, tmp_path)


def test_two_way_plan_offers_deo_only(tmp_path):
    two_way = plan_dict(backend='none')
    assert two_way['candidates'] == ['serial', DEO]
    plan = load(two_way, tmp_path)
    assert plan.backend == 'none' and plan.methods == (DEO,)
    assert {e['choice'] for e in two_way['entries']} == {DEO}             # DEO wins the synthetic world
    assert plan.digest != load(plan_dict(), tmp_path).digest
