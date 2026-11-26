"""tools/oprs_heap.py (one symmetric-heap size for all arms) and tools/oprs_tiny_model.py."""
import hashlib
import json
from pathlib import Path

import pytest

from oprs_synthetic import ASYNC_TP, DEO, every_record, write_samples
from stowaway.oprs.auto_plan import AutoPlan
from stowaway.oprs.platform import allocations_heap_bytes, heap_bytes, runtime_platform
from stowaway.oprs.workloads import build_workload

ROOT = Path(__file__).resolve().parents[1]
SITES = {s['site']: (s['op'], s['M'], s['N'], s['K']) for s in build_workload('bw27b_train', 8)['sites']}


@pytest.fixture(scope='module')
def plan_set(tmp_path_factory, scripts):
    """A plan set of the synthetic samples (bw27b_train, TP=8, external backend async_tp)."""
    plans = scripts('oprs_plans', 'tools', 'oprs_plans.py')
    out = tmp_path_factory.mktemp('plans') / 'bw27b_async_tp'
    data = write_samples(every_record(8), tmp_path_factory.mktemp('data') / 'samples_tp8_bfloat16.jsonl')
    plans.main([str(out), '--tp', '8', '--backend', 'async_tp', '--workload', 'bw27b_train', '--data', data])
    return out, data


def test_heap_tool_sizes_every_arm_by_the_largest_plan(tmp_path, capsys, scripts, plan_set):
    out, _ = plan_set
    heap = scripts('oprs_heap', 'tools', 'oprs_heap.py')
    files = heap.plan_files(out)
    assert sorted(Path(f).name for f in files) == sorted(
        p.name for p in out.glob('*.json') if p.name != 'manifest.json') and len(files) == 5
    assert heap.plan_files(out / 'manifest.json') == files
    assert [Path(f).name for f in heap.plan_files(out / 'always_deo.json')] == ['always_deo.json']
    # the constant DEO arm needs the most: two AG planes M x K and two row sites that hold two M x N planes
    # each (GEMM+RS forward, AG+dgrad backward)
    need = heap_bytes(DEO, 'ag', 32768, 896, 5120, 8) + heap_bytes(DEO, 'ag', 32768, 4352, 5120, 8) \
        + 2 * heap_bytes(DEO, 'rs', 32768, 5120, 2176, 8)
    assert need == max(
        allocations_heap_bytes([tuple(x) for x in json.loads(Path(f).read_text())['allocate']], 8) for f in files)
    for target in (out, out / 'manifest.json', out / 'always_deo.json'):
        heap.main([str(target)])
        captured = capsys.readouterr()
        assert captured.out.strip() == str(int(need * 1.25 + 2**30)) and 'largest need' in captured.err
    heap.main([str(out), '--scale', '1', '--extra-gib', '0', '--deo-bwd', 'stock'])
    assert capsys.readouterr().out.strip() == str(allocations_heap_bytes(
        [tuple(x) for x in json.loads((out / 'always_deo.json').read_text())['allocate']], 8, deo_bwd=False))
    with pytest.raises(SystemExit, match='not found'):
        heap.main([str(tmp_path / 'nope')])
    assert all(heap_bytes(ASYNC_TP, *SITES[s], 8) == 0 for s in SITES)          # async_tp: torch's own workspace


def test_every_plan_of_the_set_loads(plan_set):
    out, _ = plan_set
    platform = runtime_platform(tp=8, topology='BW8', probe_device=False)
    for path in sorted(out.glob('*.json')):
        if path.name == 'manifest.json':
            continue
        plan = json.loads(path.read_text())
        loaded = AutoPlan(str(path), 'bw27b_train', tp=8, runtime_platform=platform)
        assert loaded.digest == hashlib.sha256(path.read_bytes()).hexdigest()
        assert {e['choice'] for e in plan['entries']} <= {'serial', DEO, ASYNC_TP}


def test_tiny_model_directory(tmp_path, scripts):
    source = ROOT / '3rdparty' / 'models' / 'LLAMA3-70B'
    if not (source / 'tokenizer.json').is_file():                      # pragma: no cover
        pytest.skip('3rdparty/models/LLAMA3-70B not available')
    tiny = scripts('oprs_tiny_model', 'tools', 'oprs_tiny_model.py')
    out = tmp_path / 'tiny'
    tiny.main([str(out), '--from-model', str(source)])
    config = json.loads((out / 'config.json').read_text())
    assert (config['hidden_size'], config['intermediate_size'], config['num_hidden_layers'],
            config['vocab_size'], config['num_attention_heads'], config['num_key_value_heads']) == (
        1024, 3584, 4, 128256, 8, 8)
    assert not list(out.glob('*.safetensors')) and not list(out.glob('*.bin'))     # no weights: dummy load
    assert all((out / name).is_file() for name in tiny.COPIES)
    assert 'no weights' in (out / 'NOTE.txt').read_text()
    sites = {s['site']: (s['N'], s['K']) for s in build_workload('hf_infer', 1, rows=2048,
                                                                  config=str(out / 'config.json'))['sites']}
    assert sites == {'qkv': (1536, 1024), 'proj': (1024, 512), 'fc1': (7168, 1024), 'fc2': (1024, 3584)}
    assert all(n != k for n, k in sites.values())                      # no square weight: its orientation is unambiguous
    with pytest.raises(SystemExit, match='refusing'):
        tiny.main([str(out), '--from-model', str(source)])
    with pytest.raises(SystemExit, match='lacks'):
        tiny.main([str(tmp_path / 'x'), '--from-model', str(tmp_path)])
