"""Local emulation of DX AG grouped compute; no cross-GPU synchronization.

Peer inputs are pre-ready and local; full peer slots are laid out in rank
order, as in DX's local symmetric input buffer. It measures grouped direct
reads and the original per-peer temporary copy, not distributed overlap.
"""
import argparse, datetime, json, random, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from planner.original_a800.ag_grouping import contiguous_groups

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--rank',type=int,default=3);args=ap.parse_args()
    torch.cuda.set_device(0);torch.manual_seed(13314)
    inventory=json.loads(Path('planner/original_a800/data/cells.json').read_text())
    rows=[x for x in inventory if x['model'] in ('llama3-70b','qwen3-32b') and x['op'] in ('ag_gu','ag_up') and x['M'] in (8192,16384,32768)]
    modes=('copy_1','direct_1','direct_2','direct_4','direct_8');result=[]
    for x in rows:
        m,n,k=x['M'],x['N'],x['K'];block=m//8;rank=args.rank
        slots=torch.randn((8,block,k),device='cuda',dtype=torch.bfloat16).mul_(.05);w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16).mul_(.05)
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);tmp=torch.empty((block,k),device='cuda',dtype=torch.bfloat16)
        local=slots[rank]
        def execute(mode):
            torch.matmul(local,w.T,out=out[rank*block:(rank+1)*block])
            g=1 if mode=='copy_1' else int(mode.split('_')[1])
            for group in contiguous_groups(rank,8,g):
                lo,hi=group[0],group[-1]+1
                if mode=='copy_1':
                    tmp.copy_(slots[lo]);torch.matmul(tmp,w.T,out=out[lo*block:hi*block])
                else:
                    torch.matmul(slots[lo:hi].view(-1,k),w.T,out=out[lo*block:hi*block])
        expected=slots.view(m,k)@w.T
        for mode in modes:
            execute(mode);torch.testing.assert_close(out,expected,atol=.02,rtol=.03)
        for _ in range(4):
            for mode in modes:execute(mode)
        torch.cuda.synchronize();values={mode:[] for mode in modes};rng=random.Random(f"{x['model']}/{x['op']}/{m}")
        for _ in range(8):
            order=list(modes);rng.shuffle(order)
            for mode in order:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                for _ in range(16):execute(mode)
                end.record();end.synchronize();values[mode].append(begin.elapsed_time(end)/16)
        med={k:statistics.median(v) for k,v in values.items()}
        z={'model':x['model'],'op':x['op'],'M':m,'N':n,'K':k,'rank':rank,'groups':{str(g):contiguous_groups(rank,8,g) for g in (1,2,4,8)},'median_ms':med,'speedup_vs_copy_1':{k:med['copy_1']/v for k,v in med.items()},'samples_ms':values}
        result.append(z);print(json.dumps({k:v for k,v in z.items() if k!='samples_ms'}),flush=True)
        del slots,w,out,tmp,expected
    payload={'metadata':{'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'dtype':'bfloat16','warmup':4,'repeats':8,'batch':16,'scope':'single GPU pre-ready inputs; grouped GEMM and optional local copy; no communication or ACK; not DX end-to-end'},'results':result}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
if __name__=='__main__':main()
