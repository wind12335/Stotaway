import torch
from stowaway_nvidia.async_tp_exchange import AsyncTPExchange


def test_async_exchange_routes_operator_arguments(monkeypatch):
    calls=[]
    def ag(x, weights, dim, group, *, return_A):
        calls.append(('ag', x, weights[0], dim, group, return_A))
        return None, [x]
    def rs(x, weight, reduction, dim, group):
        calls.append(('rs', x, weight, reduction, dim, group))
        return x
    monkeypatch.setattr(torch.ops.symm_mem, 'fused_all_gather_matmul', ag, raising=False)
    monkeypatch.setattr(torch.ops.symm_mem, 'fused_matmul_reduce_scatter', rs, raising=False)
    exchange=AsyncTPExchange.__new__(AsyncTPExchange)
    exchange.group=type('Group', (), {'group_name':'tp8'})()
    exchange.counts={'ag':0,'rs':0}
    x=torch.ones((2,3)); weight=torch.ones((4,3))
    assert exchange.all_gather_gemm(x,weight) is x
    assert exchange.gemm_reduce_scatter(x,weight) is x
    assert calls[0][0]=='ag' and calls[0][3:]==(0,'tp8',False)
    assert calls[1][0]=='rs' and calls[1][3:]==('sum',0,'tp8')
    assert calls[0][2].shape == calls[1][2].shape == (3,4)
    assert exchange.counts=={'ag':1,'rs':1}
