"""Illustrative round-robin probe amortization from measured target times."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--decisions',type=Path,required=True)
    ap.add_argument('--evals',nargs='+',type=Path,required=True);ap.add_argument('--operators',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    decision=json.loads(a.decisions.read_text());queries={tuple(r['identity']):r for r in decision['predictions']}
    evaluations=[{tuple(r['identity']):r for r in json.loads(p.read_text())['rows']} for p in a.evals]
    if any(set(e)!=set(queries) for e in evaluations):raise ValueError('evaluation inventory mismatch')
    measured={}
    for path in a.operators:
        d=json.loads(path.read_text());m=d['metadata'];signature=decision['signature']
        if (m['tp'],m['dtype'],m['gpu'],m['torch'],m['triton'],m['cuda_build'],m['cuda_device_max_connections'],m['batch'],m['warmup'])!=(
            signature[1],signature[2],signature[3],signature[4],signature[5],signature[6],signature[7],signature[8],signature[-1]):
            raise ValueError('timing context mismatch')
        for r in d['results']:
            ident=(r['model'],r['site'],r['M'],r['N'],r['K'])
            if ident in measured:raise ValueError('duplicate operator identity')
            measured[ident]=r['measurements']['stock']['median_ms']
    if set(measured)!=set(queries):raise ValueError('operator inventory mismatch')
    saved_local=saved_always=0.
    for ident,q in queries.items():
        ratio=statistics.median(r[ident]['actual_speedup'] for r in evaluations)
        gain=measured[ident]*(1-1/ratio);saved_always+=gain
        if q['prediction']['choice']=='overlap':saved_local+=gain
    n=len(queries);advantage=(saved_local-saved_always)/n;cost=sum(r['component_wall_ms'] for r in queries.values())
    result={'classification':'illustrative equal-frequency operator probe amortization',
            'warning':'target stock measured in first operator batch; speedup median over listed batches; '
                      'startup and varying invocation frequencies excluded; not E2E',
            'points':n,'component_wall_ms':cost,'stock_time_source':'first target operator batch',
            'local_vs_always_dx_ms_per_call':advantage,
            'breakeven_total_calls':math.ceil(cost/advantage) if advantage>0 else None,
            'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (a.decisions,*a.evals,*a.operators)]}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
