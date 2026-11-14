"""Validate FlashOverlap RS raw-output normalization against ordinary RS.

Uses upstream ReorderRMSNorm solely as a reversible layout probe: rank-local
input reference is the even/odd rows used by upstream correctness_rs.py.
"""
import argparse,json,os,sys
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0,os.getcwd())
from RMSNorm import RMSNorm,ReorderRMSNorm
from RowParallelLinear import OverlapRowParallelLayer


def worker(rank,world,uid,m,n,k,config,shared):
    torch.cuda.set_device(rank);os.environ.update(RANK=str(rank),WORLD_SIZE=str(world),MASTER_ADDR='127.0.0.1',MASTER_PORT='29661')
    dist.init_process_group('nccl');gen=torch.Generator(device='cuda').manual_seed(5500+rank)
    x=torch.randn((m,k),device='cuda',dtype=torch.float16,generator=gen)*.05
    w=torch.randn((n,k),device='cuda',dtype=torch.float16,generator=gen)*.05
    reference=torch.empty((m//world,n),device='cuda',dtype=torch.float16)
    dist.reduce_scatter_tensor(reference,(x@w.T).contiguous())
    # Upstream interprets rank's result as even/odd global rows, whereas DX
    # returns contiguous shards. Reassemble logical rank shards explicitly.
    parts=[torch.empty_like(reference) for _ in range(world)];dist.all_gather(parts,reference)
    contiguous=torch.cat(parts,dim=0);expected=contiguous[rank::world].contiguous()
    layer=OverlapRowParallelLayer(rank,world,k,n,m,config,'reduce_scatter',uid);layer.weight=torch.nn.Parameter(w)
    normal=RMSNorm(n).cuda();reordered=ReorderRMSNorm(n,m//world,config['BM']//world,config['BN'],config['hint']).cuda()
    weight=torch.ones((n,),device='cuda',dtype=torch.float16);normal.weight=torch.nn.Parameter(weight);reordered.weight=torch.nn.Parameter(weight)
    actual=layer(x);torch.cuda.synchronize()
    norm_actual=reordered(actual);norm_expected=normal(expected)
    torch.testing.assert_close(norm_actual,norm_expected,rtol=.05,atol=.05)
    delta=(norm_actual.float()-norm_expected.float()).abs()
    shared[rank]={'rank':rank,'normalized_allclose':True,'max_abs_normalized':delta.max().item()}
    dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--m',type=int,required=True);ap.add_argument('--n',type=int,required=True);ap.add_argument('--k',type=int,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if torch.cuda.device_count()!=8:raise ValueError('TP8 required')
    raw=a.config.read_bytes();config=json.loads(raw);uid=torch.ops.flashoverlap_op.generate_nccl_id()
    manager=mp.Manager();shared=manager.dict();mp.spawn(worker,args=(8,uid,a.m,a.n,a.k,config,shared),nprocs=8)
    result={'classification':'FlashOverlap TP8 output layout compatibility probe','shape':[a.m,a.n,a.k],
            'rank_results':[shared[i] for i in range(8)],'normalization':'upstream ReorderRMSNorm compared to even/odd logical rows',
            'warning':'demonstrates required output permutation; does not measure standalone conversion cost or full model E2E'}
    # A successful earlier check can be safely replaced on a measurement retry.
    with a.output.open('w') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
