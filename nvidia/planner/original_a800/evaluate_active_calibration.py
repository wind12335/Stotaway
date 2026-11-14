"""Replay budgeted exact-shape calibration using development held-out data.

This estimates calibration benefit from labels already recorded; it does not
measure a new calibration or establish an online confidence bound.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-oof',type=Path,required=True)
    ap.add_argument('--decisions',type=Path,required=True);ap.add_argument('--calibration-eval',type=Path,required=True)
    ap.add_argument('--holdout-eval',type=Path,required=True);ap.add_argument('--cost-ms',type=float,required=True)
    ap.add_argument('--horizon',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.horizon<1 or not math.isfinite(a.cost_ms) or a.cost_ms<0:raise ValueError('invalid budget')
    train=json.loads(a.training_oof.read_text());decision=json.loads(a.decisions.read_text())
    cal=json.loads(a.calibration_eval.read_text());held=json.loads(a.holdout_eval.read_text())
    errors={}
    for r in train['rows']:
        errors.setdefault(r['op'],[]).append(abs(r['prediction']['predicted_log_speedup']-math.log(r['actual_speedup'])))
    scales={k:statistics.median(v) for k,v in errors.items()}
    queries={tuple(r['identity']):r for r in decision['predictions']}
    calibration={tuple(r['identity']):r for r in cal['rows']};test={tuple(r['identity']):r for r in held['rows']}
    if set(queries)!=set(calibration) or set(queries)!=set(test):raise ValueError('inventory mismatch')
    selected=[];rows=[]
    for ident,q in queries.items():
        op=q['op'];gain=q['prediction']['predicted_log_speedup'];scale=scales[op]
        # Approximate per-call information value in relative units. Only use
        # caller-known horizon; objective is symmetric 0/1 log-benefit.
        sigma=max(scale,1e-9);z=gain/sigma;phi=math.exp(-z*z/2)/math.sqrt(2*math.pi);cdf=.5*math.erfc(-z/math.sqrt(2))
        e_positive=sigma*phi+gain*cdf;info=max(0.,e_positive-max(0.,gain))
        full=q['query']['full_ms'];approx_value_ms=info*full*a.horizon
        if approx_value_ms>a.cost_ms:selected.append(ident)
        truth=calibration[ident]['oracle'] if ident in selected else ('overlap' if gain>0 else 'stock')
        actual=test[ident]['actual_speedup'];rows.append({'identity':ident,'op':op,'predicted_gain':gain,'error_scale':scale,
            'estimated_optimistic_info_ms':approx_value_ms,'calibrated':ident in selected,'choice':truth,'actual_speedup':actual})
    def summary(use_calibration):
        choices=[r['choice'] if use_calibration else ('overlap' if r['predicted_gain']>0 else 'stock') for r in rows]
        speeds=[r['actual_speedup'] if c=='overlap' else 1. for r,c in zip(rows,choices)]
        return {'correct':sum(c==('overlap' if r['actual_speedup']>1 else 'stock') for r,c in zip(rows,choices)),
                'slower':sum(s<1 for s in speeds),'speedup_gmean':math.exp(statistics.mean(map(math.log,speeds)))}
    out={'classification':'post-hoc simulated active same-shape calibration','warning':'normal residual assumption, '
         'full GEMM as stock proxy, observed target timing reused as calibration; cost per point is hypothetical; not online measurement',
         'horizon_per_shape':a.horizon,'assumed_cost_ms_per_point':a.cost_ms,'scales':scales,'selected':len(selected),
         'summary':{'active':summary(True),'model':summary(False)},'rows':rows,
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.training_oof,a.decisions,a.calibration_eval,a.holdout_eval)}}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:out[k] for k in ('horizon_per_shape','assumed_cost_ms_per_point','selected','summary')},indent=2))


if __name__=='__main__':main()
