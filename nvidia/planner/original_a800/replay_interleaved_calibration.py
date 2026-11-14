"""Sequentially replay independent paired probes of one exact route and shape.

A decision for a run uses only previous independent process-level medians.
This is an offline diagnostic, not an online controller or a confidence bound.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .runtime_efficiency import select_execution_mode


SIGNATURE=('model','site','op','M','N','K','batch','warmup','gpu','torch','triton',
           'cuda_build','cuda_device_max_connections')


def replay(paths, *, horizon=10000, multiplier=1.):
    reports=[];digests=set();seeds=set();signature=None
    for path in paths:
        raw=Path(path).read_bytes();digest=hashlib.sha256(raw).hexdigest();report=json.loads(raw)
        if digest in digests or report['seed'] in seeds:raise ValueError('duplicate report or seed')
        digests.add(digest);seeds.add(report['seed'])
        sig={k:report[k] for k in SIGNATURE}
        if signature is not None and sig!=signature:raise ValueError('measurement signature mismatch')
        signature=sig
        if report.get('classification')!='interleaved TP2 direct-exchange operator probe' or report['checks']<1:
            raise ValueError('checked interleaved probe required')
        samples=report['block_results']
        if len(samples)<2 or len(samples)!=report['blocks']:raise ValueError('incomplete probe')
        deltas=[];ratios=[]
        for item in samples:
            s=item['stock_ms'];d=item['dx_ms']
            if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in (s,d)):
                raise ValueError('invalid sample')
            deltas.append(s-d);ratios.append(s/d)
        reports.append({'path':str(path),'sha256':digest,'seed':report['seed'],
                        'stock_ms':statistics.median(x['stock_ms'] for x in samples),
                        'dx_ms':statistics.median(x['dx_ms'] for x in samples),
                        'saving_ms':statistics.median(deltas),'speedup':statistics.median(ratios),
                        'point_wall_ms':report['point_wall_ms'],'collector_wall_ms':report['collector_wall_ms']})
    if len(reports)<3:raise ValueError('at least three independent process reports required')
    history=[];out=[]
    for i,row in enumerate(reports):
        if len(history)<2:
            decision={'choice':'stock','status':'fewer than two prior process blocks'}
        else:
            # Preserve the paired median difference; synthetic values here encode
            # that difference only, they are not additional measurements.
            blocks=[{'id':str(h['seed']),'stock_ms':h['stock_ms'],
                     'optimized_ms':h['stock_ms']-h['saving_ms']} for h in history]
            decision=select_execution_mode(blocks,calibration_ms=sum(h['collector_wall_ms'] for h in history),
                                           horizon=horizon,uncertainty_multiplier=multiplier)
        if i>=2:
            speedup=row['speedup'] if decision['choice']=='optimized' else 1.
            out.append({'target_seed':row['seed'],'prior_seeds':[h['seed'] for h in history],
                        'choice':decision['choice'],'target_speedup_dx':row['speedup'],
                        'selected_speedup':speedup,'decision':decision})
        history.append(row)
    values=[r['selected_speedup'] for r in out]
    return {'classification':'post-hoc sequential replay of independent process probes',
            'signature':signature,'inputs':reports,'decisions':out,'horizon':horizon,
            'multiplier':multiplier,'summary':{'evaluated_runs':len(out),
                 'geomean_selected_speedup':math.exp(statistics.mean(math.log(x) for x in values)),
                 'slower_runs':sum(x<1 for x in values),'dx_selected':sum(r['choice']=='optimized' for r in out)},
            'warning':'one previously investigated shape; sequential simulation, not prospective validation; '
                      'horizon assumes equal calls, cost excludes external process startup; no statistical guarantee'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('inputs',nargs='+',type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--horizon',type=int,default=10000)
    ap.add_argument('--multiplier',type=float,default=1.);args=ap.parse_args()
    payload=replay(args.inputs,horizon=args.horizon,multiplier=args.multiplier)
    with args.output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(payload['summary'],indent=2))


if __name__=='__main__':main()
