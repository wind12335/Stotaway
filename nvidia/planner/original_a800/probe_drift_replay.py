"""Replay within-process windows from existing measured blocks, without executing a policy.

A policy may use only completed earlier blocks. Alternating block order is
inherited from the collector; every block ran both arms regardless of choice.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .replay_interleaved_calibration import SIGNATURE


def replay(paths, *, burn_in=2, window=3, margin_ms=0.):
    if burn_in<1 or window<2 or margin_ms<0:raise ValueError('invalid protocol')
    reports=[];sig=None;seeds=set()
    for path in paths:
        raw=Path(path).read_bytes();d=json.loads(raw);current={k:d[k] for k in SIGNATURE}
        if sig is not None and current!=sig:raise ValueError('signature mismatch')
        sig=current
        if d['seed'] in seeds:raise ValueError('duplicate seed')
        seeds.add(d['seed']);blocks=d['block_results']
        if len(blocks)!=d['blocks'] or len(blocks)<=burn_in+window:raise ValueError('not enough blocks')
        rows=[]
        for j in range(burn_in+window,len(blocks)):
            past=blocks[j-window:j];diff=[x['stock_ms']-x['dx_ms'] for x in past]
            median=statistics.median(diff);mad=statistics.median(abs(x-median) for x in diff)
            choice='dx' if median-mad>margin_ms else 'stock'
            target=blocks[j];ratio=target['stock_ms']/target['dx_ms']
            rows.append({'block':j,'past_blocks':list(range(j-window,j)),'choice':choice,'ratio_dx':ratio,
                         'selected_ratio':ratio if choice=='dx' else 1.,'median_saving_ms':median,'mad_ms':mad})
        reports.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'seed':d['seed'],'rows':rows,
                        'geomean_ratio':math.exp(statistics.mean(math.log(r['selected_ratio']) for r in rows)),
                        'slower_blocks':sum(r['selected_ratio']<1 for r in rows)})
    return {'classification':'retrospective chronological within-process replay, no net cost',
            'signature':sig,'burn_in':burn_in,'window':window,'reports':reports,
            'warning':'all blocks measured both arms; choices did not control execution and probe costs are omitted; '
                      'within-process blocks are not independent process trials'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('reports',nargs='+',type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--burn-in',type=int,default=2);ap.add_argument('--window',type=int,default=3)
    a=ap.parse_args();v=replay(a.reports,burn_in=a.burn_in,window=a.window)
    with a.output.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps([{'seed':r['seed'],'geomean':r['geomean_ratio'],'slower':r['slower_blocks']} for r in v['reports']],indent=2))

if __name__=='__main__':main()
