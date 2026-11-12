"""Export the original cost-model's A800 stage-1 choices for vLLM TP8."""
import argparse
import hashlib
import json
from pathlib import Path
import planner

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=planner.BUILTIN,default='70b')
    ap.add_argument('--M',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    h,f,heads,kv,layers=planner.BUILTIN[a.model]
    res=planner.plan(h,f,heads,kv,layers,1,model=a.model,stage2=False,M=a.M,quiet=True)
    mapping={'qkv':'qkv','o_proj':'proj','fc1_gate_up':'fc1','fc2':'fc2'}
    data=Path(planner.DATA)/'cells.json'
    profile=data.read_bytes()
    report={'platform':'A800-TP8','model':a.model,'M':a.M,
            'classification':'original costmodel stage1; forward only',
            'calibration_sha256':hashlib.sha256(profile).hexdigest(),
            'calibration_cells':len(json.loads(profile)),
            'choices':{mapping[r['key']]:r['deploy'] for r in res['rows']},
            'predictions':{mapping[r['key']]:r['pred'] for r in res['rows']}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
