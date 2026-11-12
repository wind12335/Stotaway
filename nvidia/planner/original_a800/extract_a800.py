"""Build the original cost-model cell schema from checked A800 measurements.

Only actual A800 timing is used; BW1000 cells supply model names and shapes.
The output is incremental and marks coverage so partial profiles remain visible.
"""
import json
import math
import os
from pathlib import Path

SOURCE = Path('/root/Stotaway/planner/data/cells.json')
RAW = Path('/root/data/stowaway-nvidia/results/tp8_20261111')
DEST = Path(__file__).resolve().parent / 'data' / 'cells.json'

def measured(path, m, k, n, op, batch):
    if not path.exists(): return None
    d=json.loads(path.read_text())
    if (d.get('world'),d.get('M'),d.get('K'),d.get('N'),d.get('batch'),d.get('schedule'),d.get('batch_ack')) != (8,m,k,n,batch,'shift',True) or d.get('checks',0)<4:
        raise ValueError(f'incompatible measurement {path}')
    x=d['result'][op]
    return x['stock_median_ms'],x['dx_median_ms'],d['repeats']

def main():
    rows=json.loads(SOURCE.read_text())
    cells=[]
    for r in rows:
        model,m,k,n,op=(r[z] for z in ('model','M','K','N','op'))
        kind='rs' if op=='rs' else 'ag'
        prefix=f'{model}_m{m}_k{k}_n{n}'
        component=RAW/f'component_{prefix}_{kind}.json'
        if not component.exists(): continue
        d=json.loads(component.read_text())
        if (d.get('tp'),d.get('M'),d.get('K'),d.get('N'),d.get('op')) != (8,m,k,n,kind):
            raise ValueError(f'incompatible component {component}')
        one=measured(RAW/f'bwmatch_probe_{prefix}_b1.json',m,k,n,kind,1)
        b2b=measured(RAW/f'bwmatch_probe_{prefix}_b16.json',m,k,n,kind,16)
        if one is None or b2b is None: continue
        comp=d['components']
        cell=dict(model=model,op=op,M=m,K=k,N=n,B=(m*n*2 if kind=='rs' else m*k*2)/1e6,
                  G=comp['G']['median_ms'], C=comp['C']['median_ms'],
                  stock_single=one[0],dx_single=one[1],stock_b2b=b2b[0],dx_b2b=b2b[1],
                  anomaly=False, source_component=str(component), repeats_single=one[2],repeats_b2b=b2b[2])
        if kind=='rs':cell['Cdx']=comp['Cdx']['median_ms']
        if any(not math.isfinite(float(cell[x])) or cell[x]<=0 for x in ('B','G','C','stock_single','dx_single','stock_b2b','dx_b2b')):
            raise ValueError(f'nonpositive or invalid cell {prefix}/{op}')
        cells.append(cell)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    temp=DEST.with_suffix('.tmp')
    temp.write_text(json.dumps(cells,indent=2)+'\n')
    os.replace(temp,DEST)
    print(f'A800 cells {len(cells)}/{len(rows)} models={len(set(c["model"] for c in cells))}')

if __name__=='__main__':main()
