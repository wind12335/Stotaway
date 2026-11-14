import json
from types import SimpleNamespace

import pytest
import torch

from stowaway_nvidia.auto_plan8 import AutoPlan8
from stowaway_nvidia.vllm_async_tp import AsyncTPLlamaSequenceParallel


def test_auto_plan_selection_and_fallback(tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps({'schema': 'stowaway-tp8-auto-plan-v1', 'workload': 'llama70_infer',
                                'tp': 8, 'dtype': 'bfloat16', 'external_backend': 'async_tp',
                                'default': 'async_tp', 'entries': [
                                    {'site': 'fc2', 'op': 'rs', 'M': 16384, 'N': 8192,
                                     'K': 3584, 'choice': 'dx'}]}))
    plan = AutoPlan8(path, 'llama70_infer')
    assert plan.choose('fc2', 'rs', 16384, 8192, 3584, 'bfloat16') == 'dx'
    assert plan.choose('fc2', 'rs', 8192, 8192, 3584, 'bfloat16') == 'async_tp'
    assert plan.choose('fc2', 'rs', 16384, 8192, 3584, 'float16') == 'stock'
    with pytest.raises(ValueError):
        AutoPlan8(path, 'qwen36_train')


def test_auto_plan_rejects_duplicate_and_invalid(tmp_path):
    path = tmp_path / 'plan.json'
    entry = {'site': 'fc2', 'op': 'rs', 'M': 8192, 'N': 4096, 'K': 1792, 'choice': 'dx'}
    plan = {'schema': 'stowaway-tp8-auto-plan-v1', 'workload': 'qwen36_train', 'tp': 8,
            'dtype': 'bfloat16', 'external_backend': 'async_tp', 'default': 'async_tp',
            'entries': [entry, entry]}
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        AutoPlan8(path, 'qwen36_train')


def test_vllm_dispatch_matches_shape_and_counts(tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps({'schema': 'stowaway-tp8-auto-plan-v1', 'workload': 'llama70_infer',
                                'tp': 8, 'dtype': 'bfloat16', 'external_backend': 'async_tp',
                                'default': 'async_tp', 'entries': [
                                    {'site': 'fc2', 'op': 'rs', 'M': 16384, 'N': 8192,
                                     'K': 3584, 'choice': 'dx'}]}))
    adapter = object.__new__(AsyncTPLlamaSequenceParallel)
    adapter.auto_plan = AutoPlan8(path, 'llama70_infer')
    adapter.auto_hits = {}
    weight = SimpleNamespace(shape=(8192, 3584))
    assert adapter._choice('fc2', 'rs', 16384, weight, torch.bfloat16) == 'dx'
    assert adapter._choice('fc2', 'rs', 8192, weight, torch.bfloat16) == 'async_tp'
    assert sorted(adapter.auto_hits.values()) == [1, 1]
