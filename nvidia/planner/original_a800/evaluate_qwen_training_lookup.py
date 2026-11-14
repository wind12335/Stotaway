"""Evaluate a frozen TP2 training lookup against independently collected samples."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from planner.tp2_lookup import Lookup, QWEN_SITES


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--test',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--interpolate',action='store_true')
    a=ap.parse_args();profile=json.loads(a.profile.read_text());lookup=Lookup(a.profile,profile['runtime'],interpolate=a.interpolate);rows=[];sources=[]
    for p in a.test:
        raw=p.read_bytes();d=json.loads(raw);m=d['metadata']
        signature={k:m[k] for k in ('tp','dtype','gpu','torch','triton','cuda_build','cuda_device_max_connections')}
        if signature!=profile['runtime'] or m['batch']!=16 or m['classification']!='diagnostic unfused site forward+backward':
            raise ValueError('test context mismatch')
        sources.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
        for r in d['results']:
            choice=lookup.choose(QWEN_SITES[r['site']],r['M'],r['K'],r['N'])
            v=r['measurements'];stock=v['stock']['median_ms'];dx=v['dx']['median_ms']
            rows.append({'site':r['site'],'M':r['M'],'K':r['K'],'N':r['N'], 'choice':choice,
                         'oracle':'dx' if dx<stock else 'stock','stock':stock,'dx':dx,
                         'speedup':stock/dx if choice=='dx' else 1.0})
    gm=lambda v:math.exp(statistics.mean(math.log(x) for x in v))
    summary={'count':len(rows),'correct':sum(r['choice']==r['oracle'] for r in rows),
             'speedup_gmean':gm([r['speedup'] for r in rows]),'slower':sum(r['speedup']<1 for r in rows),
             'always_dx_correct':sum(r['oracle']=='dx' for r in rows),
             'always_dx_gmean':gm([r['stock']/r['dx'] for r in rows]),
             'always_stock_correct':sum(r['oracle']=='stock' for r in rows),
             'oracle_gmean':gm([max(1,r['stock']/r['dx']) for r in rows])}
    report={'classification':'same Qwen static shapes, independent forward+backward repeat; diagnostic lookup',
            'warning':'measured same-shape lookup, not generalization or Qwen training E2E',
            'profile_sha256':lookup.digest,'interpolate':a.interpolate,'test_sources':sources,'summary':summary,'rows':rows}
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
