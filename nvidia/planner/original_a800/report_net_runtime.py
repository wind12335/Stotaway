"""Compare independently executed stock/adaptive operator runs per block."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

FIELDS=('model','site','op','M','N','K','calls','batch','warmup','probe_blocks','gpu','torch','triton','cuda_build','cuda_device_max_connections')


def report(design_path, results_dir):
    design=json.loads(Path(design_path).read_bytes());root=Path(results_dir);groups=[]
    for shape in design['shapes']:
        name=shape['model'].replace('shape_vehicle_','')
        # Historical run filenames omit the kv-head suffix; payloads below are
        # still checked against the complete model identity.
        ident=f"{name.split('_kv',1)[0]}_{shape['site']}_{shape['M']}"
        context=None;blocks=[]
        for b in range(1,design['blocks_per_shape']+1):
            entries={}
            for arm in design['arms']:
                path=root/f'tp2_net_{ident}_{arm}_b{b}_20261113.json';raw=path.read_bytes();row=json.loads(raw)
                sig={k:row[k] for k in FIELDS}
                if context is not None and sig!=context:raise ValueError('mixed measurement context')
                context=sig
                if row['arm']!=arm or row['checks']<1 or row['calls']!=design['calls'] or row['rank_choices']!=[row['choice']]*2:
                    raise ValueError('invalid execution report')
                if (row['model'],row['site'],row['M'])!=(shape['model'],shape['site'],shape['M']):
                    raise ValueError('shape mismatch')
                t=row['total_max_rank_wall_ms']
                if not math.isfinite(t) or t<=0:raise ValueError('invalid wall time')
                entries[arm]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                              'wall_ms':t,'choice':row['choice'],'probe_ms':row['probe_max_rank_wall_ms'],
                              'run_wall_ms':row['chosen_run_max_rank_wall_ms']}
            s=entries['stock']['wall_ms'];a=entries['adaptive']['wall_ms']
            blocks.append({'block':b,'order':design['order'][b-1],**entries,'net_speedup':s/a,'saving_ms':s-a})
        groups.append({'shape':shape,'context':context,'blocks':blocks,
                       'geomean_net_speedup':math.exp(statistics.mean(math.log(x['net_speedup']) for x in blocks)),
                       'median_saving_ms':statistics.median(x['saving_ms'] for x in blocks),
                       'slower_blocks':sum(x['net_speedup']<1 for x in blocks),
                       'dx_chosen':sum(x['adaptive']['choice']=='dx' for x in blocks)})
    return {'classification':'executed paired-process net operator benchmark','design_sha256':hashlib.sha256(Path(design_path).read_bytes()).hexdigest(),
            'groups':groups,'warning':'independent processes, common setup/warmup excluded; paired by order but no simultaneous control; not model E2E'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--design',required=True,type=Path);ap.add_argument('--results',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();value=report(a.design,a.results)
    with a.output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps([{'shape':x['shape'],'net':x['geomean_net_speedup'],'slower':x['slower_blocks'],
                       'dx_chosen':x['dx_chosen']} for x in value['groups']],indent=2))

if __name__=='__main__':main()
