"""Two-rank vLLM projection adapter scratch integration checks (not E2E)."""
import datetime
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from stowaway_nvidia.vllm_sp import LlamaSequenceParallel


class Quant:
    def apply(self, layer, value, bias):return value @ layer.weight.T


class Layer:
    def __init__(self,w):self.weight=w;self.quant_method=Quant()


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
    os.environ['VS_ROW_EXACT']='0';os.environ['VS_DX_COMPONENTS']='row'
    os.environ['STOWAWAY_REUSE_RS_SCRATCH']='1' if rank==0 else '0'
    different=LlamaSequenceParallel(None,rank,'dx',[128])
    try:
        different._verify_scratch_config(dist.group.WORLD)
        raise AssertionError('inconsistent ranks accepted')
    except RuntimeError as exc:
        if 'inconsistent RS scratch' not in str(exc):raise
    os.environ['STOWAWAY_REUSE_RS_SCRATCH']='1'
    adapter=LlamaSequenceParallel(None,rank,'dx',[128]);adapter._weight=lambda layer:layer.weight
    try:
        adapter._context(128,96,64,torch.bfloat16,'rs')
        raise AssertionError('unchecked scratch accepted')
    except RuntimeError as exc:
        if 'collectively verified' not in str(exc):raise
    adapter._verify_scratch_config(dist.group.WORLD)
    generator=torch.Generator(device='cuda').manual_seed(400+rank)
    layer=Layer(torch.randn((64,96),device='cuda',dtype=torch.bfloat16,generator=generator)*.1)
    saved=[]
    for _ in range(12):
        x=torch.randn((128,96),device='cuda',dtype=torch.bfloat16,generator=generator)*.1
        ref=torch.empty((64,64),device='cuda',dtype=torch.bfloat16)
        dist.reduce_scatter_tensor(ref,(x@layer.weight.T).contiguous())
        got=adapter._row(layer,x,128,dist.group.WORLD,'proj');saved.append((got,ref))
    torch.cuda.synchronize()
    for got,ref in saved:torch.testing.assert_close(got,ref,rtol=.02,atol=.02)
    ctx=adapter._context(128,96,64,torch.bfloat16,'rs')
    if not ctx.reuse_rs_scratch or ctx.rs_scratch is None:raise AssertionError('scratch not used')
    try:
        incompatible=LlamaSequenceParallel(None,rank,'dx',[128],forward_plan='invalid',
                                           plan_workload={'seq_length':128,'micro_batch_size':1})
        incompatible._verify_scratch_config(dist.group.WORLD)
        raise AssertionError('original plan accepted with scratch')
    except ValueError as exc:
        if 'no existing forward plan' not in str(exc):raise
    adapter.close()
    import nvshmem.core
    nvshmem.core.finalize();dist.destroy_process_group()
    if rank==0:
        payload={'classification':'synthetic vLLM projection adapter scratch integration',
                 'ranks':2,'outputs':12,'dtype':'bfloat16','mismatched_rank_config_rejected':True,
                 'unchecked_execution_rejected':True,'original_plan_rejected':True,
                 'warning':'not a full vLLM model or E2E performance result'}
        with Path(os.environ['SCRATCH_TEST_OUT']).open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
        print(json.dumps(payload),flush=True)


if __name__=='__main__':main()
