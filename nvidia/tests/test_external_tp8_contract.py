import json

import pytest
import torch

from stowaway_nvidia.auto_plan8 import AutoPlan8


@pytest.mark.parametrize('backend,dtype,op', [
    ('triton_dist', 'bfloat16', 'ag'),
    ('syncopate', 'float16', 'rs'),
    ('flashoverlap', 'float16', 'rs'),
])
def test_fixed_external_backend_and_stock_default(tmp_path, backend, dtype, op):
    site = 'fc1' if op == 'ag' else 'fc2'
    payload = {'schema': 'stowaway-tp8-auto-plan-v1', 'workload': 'operator_smoke', 'tp': 8,
               'dtype': dtype, 'external_backend': backend, 'default': 'stock',
               'entries': [{'site': site, 'op': op, 'M': 8192, 'N': 4096, 'K': 1792, 'choice': backend}]}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(payload))
    plan = AutoPlan8(path, 'operator_smoke')
    assert plan.choose(site, op, 8192, 4096, 1792, dtype) == backend
    assert plan.choose(site, op, 16384, 4096, 1792, dtype) == 'stock'
    assert plan.choose(site, op, 8192, 4096, 1792, 'float32') == 'stock'


def test_plan_does_not_mix_external_backends(tmp_path):
    payload = {'schema': 'stowaway-tp8-auto-plan-v1', 'workload': 'operator_smoke', 'tp': 8,
               'dtype': 'bfloat16', 'external_backend': 'triton_dist', 'default': 'stock',
               'entries': [{'site': 'fc2', 'op': 'rs', 'M': 8192, 'N': 4096, 'K': 1792,
                            'choice': 'syncopate'}]}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        AutoPlan8(path, 'operator_smoke')


def test_flash_layout_restores_contiguous_rank_shards():
    # Emulate the tile permutation and all-to-all without a GPU: after the
    # upstream reorder, rank r owns global rows r, r+8, r+16, ... .
    world, rows, n, bm, bn = 8, 128, 8, 2, 4
    full = torch.arange(rows * n).reshape(rows, n)
    local_rows = rows // world
    permutation = torch.randperm((local_rows // bm) * (n // bn), generator=torch.Generator().manual_seed(17))
    inverse = torch.argsort(permutation)
    cyclics = []
    for rank in range(world):
        cyclic = full[rank::world]
        logical_tiles = cyclic.reshape(local_rows // bm, bm, n // bn, bn).permute(0, 2, 1, 3).reshape(-1, bm, bn)
        raw = logical_tiles[inverse]
        restored = raw[permutation].reshape(local_rows // bm, n // bn, bm, bn).permute(0, 2, 1, 3).contiguous().reshape(local_rows, n)
        cyclics.append(restored)
    for target in range(world):
        received = torch.stack([cyclic[target * (local_rows // world):(target + 1) * (local_rows // world)] for cyclic in cyclics])
        contiguous = received.transpose(0, 1).contiguous().reshape(local_rows, n)
        assert torch.equal(contiguous, full[target * local_rows:(target + 1) * local_rows])
