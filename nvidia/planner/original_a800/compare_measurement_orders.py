"""Describe within-process early versus late paired samples, retaining all data."""
import argparse
import json
import math
import statistics
from pathlib import Path
from .replay_interleaved_calibration import SIGNATURE


def compare(paths, *, early_blocks=2):
    if isinstance(early_blocks,bool) or not isinstance(early_blocks,int) or early_blocks<1:
        raise ValueError('positive early block count required')
    output=[];signature=None;seeds=set()
    for path in paths:
        d=json.loads(Path(path).read_bytes());sig={k:d[k] for k in SIGNATURE}
        if signature is not None and sig!=signature:raise ValueError('incompatible measurement contexts')
        signature=sig
        if d['seed'] in seeds:raise ValueError('duplicate process seed')
        seeds.add(d['seed']);blocks=d['block_results']
        if len(blocks)!=d['blocks'] or len(blocks)<=early_blocks:raise ValueError('not enough blocks')
        speed=[]
        for block in blocks:
            a,b=block['stock_ms'],block['dx_ms']
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in (a,b)):
                raise ValueError('invalid timing')
            speed.append(a/b)
        early=statistics.median(speed[:early_blocks]);late=statistics.median(speed[early_blocks:])
        output.append({'path':str(path),'seed':d['seed'],'early_median':early,'late_median':late,
                       'early_samples':early_blocks,'late_samples':len(speed)-early_blocks,
                       'direction_flip':(early>1)!=(late>1),'all_speedups':speed})
    return {'classification':'within-process early/late observation', 'signature':signature,'processes':len(output),
            'direction_flips':sum(x['direction_flip'] for x in output),'rows':output,
            'warning':'retrospective diagnostic, small samples, not a change-point test or prediction guarantee'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('reports',nargs='+',type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--early-blocks',type=int,default=2);a=ap.parse_args()
    out=compare(a.reports,early_blocks=a.early_blocks)
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'processes':out['processes'],'direction_flips':out['direction_flips']}))


if __name__=='__main__':main()
