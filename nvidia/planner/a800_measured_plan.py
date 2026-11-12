"""Use already measured A800 TP8 operator points for four-site choices.

This is an exact-shape measured lookup, not a replacement for the original
cost model or its leave-one-model validation. Missing measurements choose stock.
"""
import argparse
import json
from pathlib import Path

DATA = Path('/root/data/stowaway-nvidia/results/tp8_20261111')
MODELS = {'70b':(8192,28672,64,8), '27b':(5120,17408,40,8)}

def sites(model, m):
    h,f,heads,kv = MODELS[model]
    d = h//heads
    return [('qkv','ag',m,h,(heads+2*kv)*d//8),
            ('proj','rs',m,h//8,h), ('fc1','ag',m,h,2*f//8), ('fc2','rs',m,f//8,h)]

def choose(model,m,margin=.02):
    result=[]
    for site,op,rows,k,n in sites(model,m):
        path=DATA/f'bwmatch_probe_llama3-70b_m{rows}_k{k}_n{n}_b16.json'
        # Shape, kernel schedule, batch size and correctness checks all must match.
        if model != '70b' or not path.exists():
            result.append(dict(site=site,choice='stock',reason='缺少匹配的 A800 标定'))
            continue
        d=json.loads(path.read_text())
        if (d['world'],d['M'],d['K'],d['N'],d['batch'],d['schedule'],d['batch_ack']) != (8,rows,k,n,16,'shift',True) or d['checks']<4:
            raise ValueError(f'incompatible probe: {path}')
        x=d['result'][op]
        speed=x['stock_median_ms']/x['dx_median_ms']
        result.append(dict(site=site,choice='dx' if speed>=1/(1-margin) else 'stock',
                           speedup=speed,stock_ms=x['stock_median_ms'],dx_ms=x['dx_median_ms'],source=str(path)))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=MODELS,required=True)
    ap.add_argument('--M',type=int,required=True)
    ap.add_argument('--margin',type=float,default=.02)
    args=ap.parse_args()
    print(json.dumps({'classification':'exact-shape measured forward lookup; not original cost model',
                      'model':args.model,'M':args.M,'sites':choose(args.model,args.M,args.margin)},indent=2,ensure_ascii=False))
