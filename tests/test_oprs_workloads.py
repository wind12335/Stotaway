"""Call-site manifests and the operator inventory (stowaway.oprs.workloads)."""
import importlib.util

import pytest

from stowaway.oprs import workloads as W
from stowaway.oprs.workloads import (FIRST_BATCH, MODEL_FAMILIES, RECHECK, TABLE_MS, build_workload,
                                     dense_sites, operator_inventory, qwen36_sites)

OPS = ('rs', 'ag_up', 'ag_gu')


def test_inventory_is_the_99_point_table():
    points = operator_inventory(8)
    assert len(points) == len(MODEL_FAMILIES) * len(OPS) * len(TABLE_MS) == 99
    by = {(p['model'], p['table_op'], p['M']): p for p in points}
    assert len(by) == 99                                  # no duplicate point
    hidden, ffn = MODEL_FAMILIES['llama3-70b']
    assert (by['llama3-70b', 'rs', 32768]['N'], by['llama3-70b', 'rs', 32768]['K']) == (hidden, ffn // 8)
    assert (by['llama3-70b', 'ag_up', 8192]['N'], by['llama3-70b', 'ag_up', 8192]['K']) == (ffn // 8, hidden)
    assert by['llama3-70b', 'ag_gu', 16384]['N'] == 2 * ffn // 8
    assert all(p['kind'] == 'operator' and p['site'] is None for p in points)
    assert {p['point_id'] for p in points} == {f'{m}/{o}/M{mm}' for m in MODEL_FAMILIES for o in OPS for mm in TABLE_MS}
    assert all(p['op'] == ('rs' if p['table_op'] == 'rs' else 'ag') for p in points)


def test_first_batch_is_21_shapes_at_32768():
    points = operator_inventory(8, FIRST_BATCH, (32768,))
    assert len(points) == 7 * 3 and {p['M'] for p in points} == {32768}
    assert len(FIRST_BATCH) == 7 and set(FIRST_BATCH) <= set(MODEL_FAMILIES)
    assert {p['model'] for p in points} == set(FIRST_BATCH)


def test_recheck_names_the_two_legacy_anomalies():
    assert set(RECHECK) == {'llama4-mav-d/ag_gu/M32768', 'qwen3-8b/rs/M32768'}
    assert all(r.split('/')[0] in MODEL_FAMILIES for r in RECHECK)


@pytest.mark.parametrize('name,widths', [('bw27b', (896, 5120, 4352, 5120)), ('bw70b', (1280, 8192, 7168, 8192))])
def test_dense_vehicles_sites_at_tp8(name, widths):
    sites = {s['site']: s for s in build_workload(f'{name}_train', 8)['sites']}
    assert set(sites) == set(W.SITES)
    assert {s['M'] for s in sites.values()} == {4096 * 8}                       # seq 4096 x mbs 8
    assert tuple(sites[s]['N'] for s in W.SITES) == widths
    assert [sites[s]['op'] for s in W.SITES] == ['ag', 'rs', 'ag', 'rs']
    assert [sites[s]['table_op'] for s in W.SITES] == ['ag_up', 'rs', 'ag_gu', 'rs']
    assert all(s['origin'] == s['site'] for s in sites.values())
    assert 'NOT a real Qwen3.6 model' in build_workload(f'{name}_train', 8)['note']


def test_custom_vehicle():
    with pytest.raises(ValueError):
        build_workload('custom_train', 8)                              # no config
    with pytest.raises(ValueError):
        build_workload('custom_train', 8, config='{"hidden": 1024, "ffn": 2048}')     # heads / kv_groups missing
    sites = build_workload('custom_train', 8, config='{"hidden":1024,"ffn":2048,"heads":16,"kv_groups":4}')['sites']
    assert {s['site']: (s['N'], s['K']) for s in sites} == {
        'qkv': (192, 1024), 'proj': (1024, 128), 'fc1': (512, 1024), 'fc2': (1024, 256)}
    with pytest.raises(ValueError):                                    # 2 x 2050 is not divisible by 8
        build_workload('custom_train', 8, config='{"hidden":1024,"ffn":2050,"heads":16,"kv_groups":4}')


def test_rejected_workloads():
    with pytest.raises(ValueError):
        build_workload('no_such_workload', 8)
    with pytest.raises(ValueError):                                    # rows must be divisible by TP
        build_workload('custom_train', 8, rows=3, config='{"hidden":1024,"ffn":2048,"heads":16,"kv_groups":4}')
    with pytest.raises(ValueError):
        dense_sites(1024, 2048, 16, 4, 4096, 15)                       # qkv width 1536 not divisible by 15


def test_llama70_infer_sites(tmp_path):
    config = W.Path(W.__file__).resolve().parents[3] / '3rdparty' / 'models' / 'LLAMA3-70B' / 'config.json'
    if not config.exists():                                            # pragma: no cover - weights not checked out
        pytest.skip('3rdparty/models/LLAMA3-70B not available')
    table = {(p['table_op'], p['N'], p['K']) for p in operator_inventory(8)}
    sites = {s['site']: s for s in build_workload('llama70_infer', 8)['sites']}
    assert {s['M'] for s in sites.values()} == {16384}
    # fc1 and fc2 ARE operator-table points of the same model; qkv and proj are off-table targets
    assert ('ag_gu', sites['fc1']['N'], sites['fc1']['K']) in table
    assert ('rs', sites['fc2']['N'], sites['fc2']['K']) in table
    assert ('ag_up', sites['qkv']['N'], sites['qkv']['K']) not in table
    assert ('rs', sites['proj']['N'], sites['proj']['K']) not in table
    several = build_workload('llama70_infer', 8, rows='8192,16384')
    assert sorted({s['M'] for s in several['sites']}) == [8192, 16384] and len(several['sites']) == 8


def qwen_config():
    """The synthetic Qwen3.6 text config of nvidia/tests/test_qwen36_shapes.py."""
    return {'text_config': {'hidden_size': 5120, 'intermediate_size': 17408,
            'num_attention_heads': 24, 'num_key_value_heads': 4, 'head_dim': 256,
            'linear_num_key_heads': 16, 'linear_key_head_dim': 128,
            'linear_num_value_heads': 48, 'linear_value_head_dim': 128,
            'attn_output_gate': True, 'num_hidden_layers': 64,
            'layer_types': ['linear_attention'] * 48 + ['full_attention'] * 16}}


def test_qwen36_sites_match_the_nvidia_manifest():
    """Shape parity with nvidia/planner/original_a800/qwen36_shapes.build_manifest (the real Qwen3.6
    config is not in this repository: the same synthetic config as the NVIDIA test)."""
    path = W.Path(W.__file__).resolve().parents[3] / 'nvidia' / 'planner' / 'original_a800' / 'qwen36_shapes.py'
    if not path.exists():                                              # pragma: no cover
        pytest.skip('nvidia/planner not available')
    spec = importlib.util.spec_from_file_location('nvidia_qwen36_shapes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ours = qwen36_sites(qwen_config(), 8192, 8)
    theirs = module.build_manifest(qwen_config(), 8192)['sites']
    # ours: hook call site + layer-specific origin + route direction; upstream: layer key + adapter name + op
    assert [(s['site'], s['origin'], s['table_op'], s['M'], s['K'], s['N'], s['layers']) for s in ours] == [
        (t['adapter_name'], t['site'], t['op'], t['M'], t['K'], t['N'], t['layers']) for t in theirs]
    assert all(s['op'] == 'ag' for s in ours if s['table_op'].startswith('ag'))
    by = {(s['origin']): s for s in ours}
    assert (by['full_qkv']['K'], by['full_qkv']['N'], by['full_qkv']['layers']) == (5120, 1792, 16)
    assert (by['gdn_in']['K'], by['gdn_in']['N'], by['gdn_in']['layers']) == (5120, 2060, 48)
    assert (by['mlp_fc2']['K'], by['mlp_fc2']['N']) == (2176, 5120)
    assert sum(1 for s in ours if s['site'] == 'fc1') == 2            # gdn_in and mlp_fc1 share the hook site


def test_qwen36_sites_reject_bad_context():
    with pytest.raises(ValueError):
        qwen36_sites(qwen_config(), 8193, 8)
    bad = qwen_config()
    bad['text_config']['num_hidden_layers'] = 63
    with pytest.raises(ValueError):
        qwen36_sites(bad, 8192, 8)
    with pytest.raises(ValueError):
        qwen36_sites(qwen_config(), 8192, 7)
