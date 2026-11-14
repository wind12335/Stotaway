"""Summarize fixed three-arm TP2 replacement experiment without choosing policy."""
import argparse,hashlib,json,math,statistics
from pathlib import Path


def report(manifest_path,directory):
    raw=Path(manifest_path).read_bytes();design=json.loads(raw);sha=hashlib.sha256(raw).hexdigest();groups=[]
    for i,target in enumerate(design['targets']):
        blocks=[];signature=None
        for b,order in enumerate(design['orders'],1):
            arms={}
            for arm in order:
                path=Path(directory)/f'tp2_replace_ag_t{i}_{arm}_b{b}_20261113.json';data=path.read_bytes();d=json.loads(data)
                if d['manifest_sha256']!=sha or d['target']!=target or d['arm']!=arm or d['checks']<1:
                    raise ValueError('invalid provenance')
                sig=tuple(d[k] for k in ('tp','dtype','batch','warmup','repeats','gpu','torch','triton','cuda_build','cuda_device_max_connections'))
                if signature is not None and sig!=signature:raise ValueError('mismatched measurement protocol')
                signature=sig;t=d['median_ms']
                if not isinstance(t,(float,int)) or not math.isfinite(t) or t<=0:raise ValueError('invalid time')
                arms[arm]={'median_ms':t,'sha256':hashlib.sha256(data).hexdigest(),'path':str(path),
                           'actual_triton_dist_path':d.get('triton_dist_path')}
            blocks.append({'block':b,'order':order,'arms':arms,'dx_vs_external':arms['triton_dist']['median_ms']/arms['dx']['median_ms'],
                           'dx_vs_stock':arms['stock']['median_ms']/arms['dx']['median_ms'],
                           'external_vs_stock':arms['stock']['median_ms']/arms['triton_dist']['median_ms']})
        geomean=lambda field: math.exp(statistics.mean(math.log(x[field]) for x in blocks))
        groups.append({'target':target,'blocks':blocks,'dx_vs_external_gmean':geomean('dx_vs_external'),
                       'dx_vs_stock_gmean':geomean('dx_vs_stock'),'external_vs_stock_gmean':geomean('external_vs_stock'),
                       'dx_faster_blocks':sum(x['dx_vs_external']>1 for x in blocks)})
    return {'classification':'same-shape same-software three-arm TP2 AG operator comparison',
            'manifest_sha256':sha,'groups':groups,
            'warning':'targets selected after prior measurements; arms run in independent processes; '
                      'not a dynamic planner, no overhead or E2E; no confidence guarantee'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();v=report(a.manifest,a.directory)
    with a.output.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps([{'target':g['target'],'dx_vs_external_gmean':g['dx_vs_external_gmean'],
                       'dx_vs_stock_gmean':g['dx_vs_stock_gmean'],'dx_faster_blocks':g['dx_faster_blocks']} for g in v['groups']],indent=2))

if __name__=='__main__':main()
