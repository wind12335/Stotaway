"""Report fixed-protocol chunk synchronization vs stock without E2E claims."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from .report_net_runtime import FIELDS


def report(root):
    root=Path(root);signature=None;blocks=[]
    for b in range(1,5):
        values={}
        for arm in ('stock','adaptive128'):
            path=root/f'tp2_sync_h4608_{arm}_b{b}_20261113.json';raw=path.read_bytes();d=json.loads(raw)
            context={k:d[k] for k in FIELDS}
            if signature is not None and context!=signature:raise ValueError('different contexts')
            signature=context
            if d['arm']!=arm or d['rank_choices']!=[d['choice']]*2 or d['checks']<1:raise ValueError('invalid run')
            t=d['total_max_rank_wall_ms']
            if not math.isfinite(t) or t<=0:raise ValueError('invalid elapsed time')
            values[arm]={'wall_ms':t,'choice':d['choice'],'probe_ms':d['probe_max_rank_wall_ms'],
                         'sha256':hashlib.sha256(raw).hexdigest(),'path':str(path)}
        blocks.append({'block':b,'order':'stock-adaptive128' if b%2 else 'adaptive128-stock',**values,
                       'net_speedup':values['stock']['wall_ms']/values['adaptive128']['wall_ms']})
    return {'classification':'executed bounded chunk synchronization comparison','context':signature,'blocks':blocks,
            'geomean_net_speedup':math.exp(statistics.mean(math.log(x['net_speedup']) for x in blocks)),
            'slower_blocks':sum(x['net_speedup']<1 for x in blocks),
            'warning':'independent process arms, common initialization and warmup excluded; synchronization changes execution, not model E2E'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();d=report(a.results)
    with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
    print(json.dumps({'geomean_net_speedup':d['geomean_net_speedup'],'slower_blocks':d['slower_blocks']}))

if __name__=='__main__':main()
