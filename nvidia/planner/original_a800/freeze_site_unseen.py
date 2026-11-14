"""Freeze a site-aware candidate and strictly novel target families."""
import argparse
import hashlib
import json
from pathlib import Path

from .site_residual_policy import SiteResidualPolicy
from .validate_tp2_new import load
from bench.tp2_cells import shapes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-report',type=Path,required=True)
    ap.add_argument('--shapes',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    report=json.loads(a.train_report.read_bytes());rows,sig,sources=load([Path(s['path']) for s in report['train_sources']])
    inventory={(r['op'],r['K'],r['N']) for r in rows};shape_raw=a.shapes.read_bytes();vehicles=json.loads(shape_raw)
    models={op:SiteResidualPolicy().fit([r for r in rows if r['op']==op]) for op in sorted({r['op'] for r in rows})}
    predictions=[]
    for model,dims in vehicles.items():
        for site,direction,_,k,n in shapes(*dims,1024):
            op='rs' if direction=='rs' else 'ag_up' if site=='qkv' else 'ag_gu'
            if (op,k,n) in inventory:raise ValueError(f'target family in training: {(op,k,n)}')
            for m in (1024,4096):
                row={'site':site,'M':m,'N':n,'K':k};pred=models[op].predict(row)
                predictions.append({'model':model,**row,'op':op,'prediction':pred})
    output={'schema':'stowaway-site-unseen-freeze-v1','signature':sig,'sources':sources,
            'shape_file_sha256':hashlib.sha256(shape_raw).hexdigest(),
            'models':{op:model.to_dict() for op,model in models.items()},'predictions':predictions,
            'warning':'frozen before target collection, model designed after earlier target set was examined'}
    raw=(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(raw).hexdigest(),'predictions':len(predictions)},indent=2))


if __name__=='__main__':main()
