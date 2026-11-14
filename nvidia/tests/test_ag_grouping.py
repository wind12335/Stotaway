import pytest
from planner.original_a800.ag_grouping import contiguous_groups

@pytest.mark.parametrize('world', [2, 8])
@pytest.mark.parametrize('group_size', [1, 2, 4, 8])
@pytest.mark.parametrize('order', ['fixed', 'shift'])
def test_groups_preserve_exact_peer_order_and_contiguous_ranges(world, group_size, order):
    for rank in range(world):
        groups = contiguous_groups(rank, world, group_size, order)
        peers = [p for g in groups for p in g]
        expected = ([p for p in range(world) if p != rank] if order == 'fixed'
                    else [(rank + d) % world for d in range(1, world)])
        assert peers == expected
        assert all(1 <= len(g) <= group_size and list(g) == list(range(g[0], g[-1] + 1)) for g in groups)

@pytest.mark.parametrize('args', [(-1, 8, 2), (8, 8, 2), (0, 1, 2), (0, 8, 3), (0, 8, 0)])
def test_invalid_grouping(args):
    with pytest.raises(ValueError):
        contiguous_groups(*args)
