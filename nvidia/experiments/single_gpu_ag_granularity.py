"""Single GPU: full, 2-, 4-, 8-slice AG-shaped GEMM; no communication."""
import argparse, datetime, json, random, statistics
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cells',type=Path,default=Path('planner/original_a800/data/cells.json'));a=p.parse_args()
 torch.cuda.set_device(0); torch.manual_seed(20261114)
 cells=json.loads(a.cells.read_text()); models=('llama3-70b','qwen3-32b'); shapes=[x for x in cells if x['model'] in models and x['op'] in ('ag_gu','ag_up') and x['M'] in (8192,16384,32768)]
 results=[]
 for x in shapes:
  m,n,k=x['M'],x['N'],x['K'];v=torch.randn((m,k),device='cuda',dtype=torch.bfloat16).mul_(.05);w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16).mul_(.05);o=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
  def run(parts):
   for i in range(parts):
    sl=slice(i*m//parts,(i+1)*m//parts);torch.matmul(v[sl],w.T,out=o[sl])
  reference=v@w.T
  for parts in (1,2,4,8):
   run(parts);torch.testing.assert_close(o,reference,rtol=.03,atol=.02)
  for _ in range(4):
   for parts in (1,2,4,8):run(parts)
  torch.cuda.synchronize();samples={p:[] for p in (1,2,4,8)};rng=random.Random(f"{x['model']}/{x['op']}/{m}")
  for _ in range(8):
   order=[1,2,4,8];rng.shuffle(order)
   for parts in order:
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
    for _ in range(16):run(parts)
    end.record();end.synchronize();samples[parts].append(start.elapsed_time(end)/16)
  med={str(parts):statistics.median(vals) for parts,vals in samples.items()}
  item={'model':x['model'],'op':x['op'],'M':m,'N':n,'K':k,'median_ms':med,'relative_to_full':{s:t/med['1'] for s,t in med.items()},'samples_ms':samples}
  print(json.dumps(item,ensure_ascii=False),flush=True);results.append(item)
  del v,w,o,reference
 report={'metadata':{'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'dtype':'bfloat16','warmup':4,'repeats':8,'batch':16,'scope':'one GPU GEMM only; no transfers or overlap','order':'shuffled per repeat'},'results':results}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
if __name__=='__main__':main()
