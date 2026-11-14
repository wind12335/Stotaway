"""Synthetic TP8 model exercises actual vLLM async/DX projection dispatch."""
import datetime, os, types, importlib.util
from pathlib import Path
import torch
import torch.distributed as dist
spec=importlib.util.spec_from_file_location('vllm_synthetic', Path(__file__).resolve().parents[1] / 'tests/test_vllm_forward_simulated.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
install_stub, Layer, Norm, reference = module.install_stub, module.Layer, module.Norm, module.reference
from stowaway_nvidia.vllm_async_tp import AsyncTPLlamaSequenceParallel

class Adapter(AsyncTPLlamaSequenceParallel):
    @staticmethod
    def _weight(layer): return layer.weight

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
install_stub()
gen=torch.Generator(device='cuda').manual_seed(501+rank)
model=types.SimpleNamespace(layers=[Layer(64,torch.bfloat16,gen) for _ in range(2)],start_layer=0,end_layer=2,aux_hidden_state_layers=(),norm=Norm())
os.environ['VS_ASYNC_DX_SITES']=os.environ.get('VS_ASYNC_DX_SITES','')
a=Adapter(None,rank,'dx',[1024],world=8)
for turn in range(2):
    source=torch.Generator(device='cuda').manual_seed(77+turn)
    x=torch.randn(1024,64,dtype=torch.bfloat16,device='cuda',generator=source)
    expected=reference(model,x)
    actual=a.forward(model,None,torch.arange(1024,device='cuda'),inputs_embeds=x)
    torch.testing.assert_close(actual,expected,rtol=.05,atol=.05)
print(f'rank={rank} hits={a.backend_hits} PASS',flush=True)
a.close()
if a.cuda is not None:
    import nvshmem.core
    nvshmem.core.finalize()
dist.destroy_process_group()
