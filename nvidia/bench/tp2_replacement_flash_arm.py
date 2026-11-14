"""Single FlashOverlap RS arm with explicit manifest and native output layout."""
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0,os.getcwd())
from test import reorder_indices,generate_row_remap_array


def worker(rank,world,uid,m,n,k,config,batch,repeats,warmup,shared):
    torch.cuda.set_device(rank);os.environ.update(RANK=str(rank),WORLD_SIZE=str(world),MASTER_ADDR='127.0.0.1',MASTER_PORT='29659')
    dist.init_process_group('nccl');gen=torch.Generator(device='cuda').manual_seed(5500+rank)
    a=torch.randn((m,k),device='cuda',dtype=torch.float16,generator=gen)*.05
    b=torch.randn((n,k),device='cuda',dtype=torch.float16,generator=gen)*.05
    c=torch.empty((m,n),device='cuda',dtype=torch.float16);d=torch.empty((m//world,n),device='cuda',dtype=torch.float16)
    bm,bn,algo,seg,hint=(config[x] for x in ('BM','BN','Algo','cSeg','hint'))
    if len(seg)<=1:raise ValueError('overlap configuration required')
    counter=torch.zeros((n//bn,),dtype=torch.int32,device='cuda')
    order=reorder_indices((m//bm)*(n//bn),hint).reshape((m//bm,n//bn))
    row=generate_row_remap_array(m,n,bm,bn,seg,world)
    seg_cpu=torch.tensor(seg,dtype=torch.int32);seg_gpu=seg_cpu.cuda(rank)
    impl=torch.classes.flashoverlap_class.OverlapImpl();impl.nccl_init(rank,world,uid);impl.cutlass_init();impl.overlap_init()
    def run():impl.gemm_reducescatter_overlap(a,b,c,d,counter,order,row,1,seg_cpu,seg_gpu,algo,False)
    for _ in range(warmup):run()
    torch.cuda.synchronize();dist.barrier();t0=time.perf_counter();samples=[]
    for _ in range(repeats):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
        for _ in range(batch):run()
        end.record();end.synchronize();v=torch.tensor([begin.elapsed_time(end)/batch],device='cuda');dist.all_reduce(v,op=dist.ReduceOp.MAX);samples.append(v.item())
    if rank==0:shared['value']={'samples_ms':samples,'median_ms':statistics.median(samples),'measurement_wall_ms':(time.perf_counter()-t0)*1000}
    dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    manifest_raw=a.manifest.read_bytes();manifest=json.loads(manifest_raw);target=manifest['targets'][a.index];m,n,k=(target[x] for x in ('M','N','K'))
    contract=json.loads(a.contract.read_bytes())
    if contract.get('shape')!=[m,n,k] or len(contract.get('rank_results',[]))!=2 or not all(x['normalized_allclose'] for x in contract['rank_results']):
        raise ValueError('matching two-rank layout correctness check required')
    cfg_raw=a.config.read_bytes();config=json.loads(cfg_raw);uid=torch.ops.flashoverlap_op.generate_nccl_id()
    manager=mp.Manager();shared=manager.dict();mp.spawn(worker,args=(2,uid,m,n,k,config,manifest['batch'],manifest['repeats'],manifest['warmup'],shared),nprocs=2)
    result={'classification':'single arm TP2 RS replacement measurement','arm':'flashoverlap','target':target,
            'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'dtype':'float16','tp':2,'batch':manifest['batch'],
            'warmup':manifest['warmup'],'repeats':manifest['repeats'],'gpu':torch.cuda.get_device_name(0),
            'torch':str(torch.__version__),'triton':__import__('triton').__version__,'cuda_build':torch.version.cuda,
            'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset'),
            'config_sha256':hashlib.sha256(cfg_raw).hexdigest(),'contract_sha256':hashlib.sha256(a.contract.read_bytes()).hexdigest(),
            'checks':2,'output_semantics':'FlashOverlap native reordered tiles, validated through upstream reorder RMSNorm',
            'warning':'raw operator timing excludes output layout conversion; independent process; no E2E',**dict(shared)['value']}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
