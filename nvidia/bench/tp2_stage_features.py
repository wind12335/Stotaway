"""Target-shape timing of full and split GEMMs; overlap labels not used here."""
import datetime
import json
import os
import random
import statistics
import time
from pathlib import Path
import torch
import torch.distributed as dist
import triton

from bench.tp2_cells import configured_sites,measure


def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
    m=int(os.environ.get('CELL_M','1024'));batch=16;repeats=int(os.environ.get('STAGE_REPEATS','8'));seed=int(os.environ.get('STAGE_SEED','1'))
    if repeats < 1:raise ValueError('STAGE_REPEATS must be positive')
    result={'metadata':{'tp':2,'M':m,'batch':batch,'repeats':repeats,'warmup':3,'seed':seed,
                        'p2p_bidirectional':torch.cuda.can_device_access_peer(0,1) and torch.cuda.can_device_access_peer(1,0),
                        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                        'cuda_build':torch.version.cuda,'triton':triton.__version__,
                        'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset'),
                        'classification':'standalone GEMM and optional stock/comm component measurements',
                        'rs_actual_path':'direct_exchange_rs_scratch' if os.environ.get('CELL_RS_SCRATCH')=='1' else 'direct_exchange'},'results':[]}
    for model,site,op,rows,k,n in configured_sites(m):
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*.05
        w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)*.05
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        def full():torch.matmul(x,w.T,out=out)
        def split():
            torch.matmul(x[:m//2],w.T,out=out[:m//2]);torch.matmul(x[m//2:],w.T,out=out[m//2:])
        tasks=[('full',full),('split',split)];random.Random(f'{seed}:{model}:{site}').shuffle(tasks)
        if os.environ.get('STAGE_INCLUDE_BASELINE')=='1':
            shard=x[:m//2].contiguous();reduced=torch.empty((m//2,n),device='cuda',dtype=torch.bfloat16)
            if op=='ag':
                def comm():dist.all_gather_into_tensor(x,shard)
                def stock():comm();full()
            else:
                def comm():dist.reduce_scatter_tensor(reduced,out)
                def stock():full();comm()
            tasks.extend([('comm',comm),('stock',stock)])
            random.Random(f'{seed}:{model}:{site}:baseline').shuffle(tasks)
        start=time.monotonic();times={name:measure(fn,repeats,batch) for name,fn in tasks}
        result['results'].append({'model':model,'site':site,'op':op,'M':m,'N':n,'K':k,
                                  'measurements':times,'elapsed_wall_ms':(time.monotonic()-start)*1000})
        if rank==0:print(json.dumps({'model':model,'site':site,'M':m,'N':n,'K':k,
                                    'full_ms':times['full']['median_ms'],'split_ms':times['split']['median_ms']}),flush=True)
        del x,w,out
    if rank==0:
        with Path(os.environ['STAGE_OUT']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
