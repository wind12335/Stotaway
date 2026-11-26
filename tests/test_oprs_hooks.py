"""CPU-visible pieces of the framework adapters and the arm table (what can be checked without a GPU)."""
import pytest
import torch

from stowaway.oprs import arms
from stowaway.oprs import megatron_hook as M
from stowaway.oprs import vllm_hook as V


def test_call_sites_are_named_by_module_path():
    assert M.SITES == ('qkv', 'proj', 'fc1', 'fc2') == V.SITES
    assert M.site_of('decoder.layers.3.self_attention.linear_qkv') == 'qkv'
    assert M.site_of('module.decoder.layers.0.self_attention.linear_proj') == 'proj'
    assert M.site_of('decoder.layers.0.mlp.linear_fc1') == 'fc1'
    assert M.site_of('decoder.layers.12.mlp.linear_fc2') == 'fc2'
    assert M.site_of('output_layer') is None                       # the vocabulary head is not a call site
    assert M.site_of('decoder.layers.0.mlp.linear_fc1.weight') is None     # a parameter, not the layer
    assert M.site_of('decoder.layers.0.mlp.experts.0.linear_fc1') is None  # an expert of a MoE layer
    assert set(M.BY_LEAF.values()) == set(M.SITES) and M.BY_LEAF['linear_proj'] == 'proj'


def test_an_unarmed_process_exits_normally():
    assert M._ST['disp'] is None and M._snapshot() is None
    assert M._exit() is None                                   # nothing was started: no os._exit
    M.report()                                                 # prints one line, harmless twice
    M.report()


def test_arm_kinds_map_to_the_plan_files():
    assert arms.plan_file('sp_rccl', 'async_tp') == 'null.json'
    assert arms.plan_file('null', 'async_tp') == 'null.json'
    assert arms.plan_file('all_deo', 'triton_dist') == arms.plan_file('always_deo', 'triton_dist') == 'always_deo.json'
    assert arms.plan_file('oprs', 'triton_dist') == 'oprs_triton_dist.json'
    assert arms.plan_file('oprs', 'async_tp') == 'oprs_async_tp.json'
    assert arms.plan_file('always_async_tp', 'async_tp') == 'always_async_tp.json'
    assert arms.plan_file('oprs_deo_only', 'triton_dist') == 'oprs_deo_only.json'
    for kind in ('stock', 'serial', 'dx', 'plan', 'always_flux'):
        with pytest.raises(ValueError):
            arms.plan_file(kind, 'async_tp')


def test_vllm_arm_specs_resolve():
    resolve = arms.resolve_arm
    assert resolve('s1:stock', '/p', 'async_tp') == ('s1', 'stock', None)
    assert resolve('stock', '/p', 'async_tp') == ('stock', 'stock', None)
    assert resolve('sp_rccl', '/p', 'async_tp') == ('sp_rccl', 'sp_rccl', '/p/null.json')
    assert resolve('o1:oprs', '/p/', 'triton_dist') == ('o1', 'oprs', '/p/oprs_triton_dist.json')
    assert resolve('b1:always_triton_dist', '/p', 'triton_dist') == ('b1', 'always_triton_dist',
                                                                     '/p/always_triton_dist.json')
    assert resolve('n:/tmp/x.json', '/p', 'triton_dist') == ('n', 'plan', '/tmp/x.json')
    with pytest.raises(ValueError):
        resolve('x:dx', '/p', 'async_tp')


class Layer:
    def __init__(self, weight, k_in, n_out):
        self.weight, self.input_size_per_partition, self.output_size_per_partition = weight, k_in, n_out


def test_weight_orientation():
    """The das fork may store [K_in, N_out]; every method is given the [N_out, K_in] view."""
    kn = torch.randn(64, 32)
    view = V._w_nk(Layer(kn, 64, 32))
    assert view.shape == (32, 64) and view.data_ptr() == kn.data_ptr() and view.stride(1) != 1
    nk = torch.randn(32, 64)
    assert V._w_nk(Layer(nk, 64, 32)) is nk
    with pytest.raises(AssertionError):
        V._w_nk(Layer(torch.randn(16, 16), 16, 16))             # square: the orientation is ambiguous
    with pytest.raises(AssertionError):
        V._w_nk(Layer(torch.randn(8, 8), 64, 32))


def test_k_contiguous_copy_is_cached_and_refreshed():
    """Triton-distributed's AG kernel needs a K-contiguous weight: one cached copy per weight, re-made
    when the parameter is rewritten."""
    layer = Layer(torch.randn(64, 32), 64, 32)
    V._ST['kc'].clear()
    view = V._w_nk(layer)
    first = V._k_contiguous(layer, view)
    again = V._k_contiguous(layer, view)
    assert first.is_contiguous() and first.stride(1) == 1 and again is first and len(V._ST['kc']) == 1
    assert torch.equal(first, view.contiguous())
    with torch.no_grad():
        layer.weight.mul_(2)                                    # a rewritten parameter (its _version moves)
    fresh = V._k_contiguous(layer, V._w_nk(layer))
    assert fresh is not first and torch.equal(fresh, V._w_nk(layer).contiguous())
    V._ST['kc'].clear()


def test_vllm_stats_without_a_dispatcher():
    V._ST.update(disp=None, n_sp=0, n_stock=0)
    stats = V.stats()
    assert stats['disp'] is None and stats['runs'] == {} and 'dispatcher not built' in V.stats_line()
