"""Scenario arithmetic for the extra measured component probes."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--decisions',type=Path,required=True)
    ap.add_argument('--runs',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    frozen=json.loads(a.decisions.read_bytes());queries={tuple(r['identity']):r for r in frozen['predictions']};runs=[]
    for path in a.runs:
        data=json.loads(path.read_bytes());mapping={tuple(r['identity']):r for r in data['rows']}
        if set(mapping)!=set(queries):raise ValueError('run inventory mismatch')
        runs.append(mapping)
    model_saving=baseline_saving=0.
    for identity,q in queries.items():
        ratios=[run[identity]['actual_speedup'] for run in runs]
        # Two-point median of measured speedups is exploratory; use the
        # separately measured component stock latency for this scenario.
        ratio=statistics.median(ratios)
        # Reduced-probe bundles do not measure target stock. Use model's
        # estimate and flag it below; this is not an observed baseline time.
        stock=q['query'].get('stock_ms',q['prediction'].get('estimated_stock_ms'))
        if stock is None or not math.isfinite(stock) or stock<=0:raise ValueError('missing stock time')
        baseline_saving+=stock*(1-1/ratio)
        if q['prediction']['choice']=='overlap':model_saving+=stock*(1-1/ratio)
    n=len(queries);model_saving/=n;baseline_saving/=n
    cost=sum(q['component_wall_ms'] for q in queries.values())
    advantage=model_saving-baseline_saving
    out={'classification':'illustrative round-robin amortization; no model E2E measurement',
         'warning':'assumes each of 40 shapes runs once per cycle, nonnegative net probe cost, equal rank-side wall time; excludes process launch; estimates not guarantees',
         'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [a.decisions,*a.runs]],
         'points':n,'total_component_wall_ms':cost,'mean_component_wall_ms':cost/n,
         'stock_time_source':'measured' if all('stock_ms' in q['query'] for q in queries.values()) else 'model_estimate',
         'model_mean_saved_ms_per_invocation_vs_stock':model_saving,
         'always_dx_mean_saved_ms_per_invocation_vs_stock':baseline_saving,
         'mean_advantage_ms_per_invocation_vs_always_dx':advantage,
         'breakeven_invocations_total_assuming_round_robin':math.ceil(cost/advantage) if advantage>0 else None}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:out[k] for k in out if k!='sources'},indent=2))


if __name__=='__main__':main()
