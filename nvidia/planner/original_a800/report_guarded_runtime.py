"""Compare monitored execution with stock in fixed independent-process blocks."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .report_net_runtime import FIELDS


def report(root, blocks=4):
    root=Path(root);result=[];signature=None
    for block in range(1,blocks+1):
        runs={}
        for arm in ('stock','guarded'):
            path=root/f'tp2_guarded_h4608_{arm}_b{block}_20261113.json';raw=path.read_bytes();d=json.loads(raw)
            sig={k:d[k] for k in FIELDS}
            if signature is not None and sig!=signature:raise ValueError('context mismatch')
            signature=sig
            if d['arm']!=arm or d['rank_choices']!=[d['choice']]*2 or d['checks']<1:
                raise ValueError('invalid distributed result')
            val=d['total_max_rank_wall_ms']
            if not math.isfinite(val) or val<=0:raise ValueError('invalid timing')
            runs[arm]={'total_ms':val,'choice':d['choice'],'initial_choice':d['initial_choice'],
                       'probe_ms':d['probe_max_rank_wall_ms'],'transitions':d['transitions'],
                       'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
        result.append({'block':block,'order':'stock-guarded' if block%2 else 'guarded-stock',**runs,
                       'net_speedup':runs['stock']['total_ms']/runs['guarded']['total_ms']})
    return {'classification':'executed guarded vs stock operator loop', 'context':signature,'blocks':result,
            'geomean_net_speedup':math.exp(statistics.mean(math.log(b['net_speedup']) for b in result)),
            'slower_blocks':sum(b['net_speedup']<1 for b in result),
            'warning':'same known shape, independently launched processes; common context/warmup excluded; not model E2E'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();d=report(a.results)
    with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
    print(json.dumps({'geomean':d['geomean_net_speedup'],'slower':d['slower_blocks'],
                      'switches':sum(bool(x['guarded']['transitions']) for x in d['blocks'])}))

if __name__=='__main__':main()
