"""Grouped Flux shape-only evaluation on the user's existing TP4 PDF logs.

GPU model/topology and complete software versions are unknown. No runtime
adapter is claimed; this analysis must never be used for A800 TP2 deployment.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .orthogonal_predictor import Route,Sample
from .route_policy import GroupedSample,RoutePolicy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=a.input.read_bytes();data=json.loads(raw);sections={}
    for direction in ('ag','rs'):
        records=[r for r in data['rows'] if r['method']=='flux' and r['op']==direction]
        # Do not assert CUTLASS, NVSHMEM or transport from a PDF whose actual
        # binary path and topology were not recorded.
        route=Route('flux','unknown','unknown','unknown','unknown',direction,'pdf_reported_flux',
                    'unknown','unknown',4,'float16',(('source','user_pdf_unknown_version'),),
                    'single_or_10_iters_pdf',None,'pdf_reported_unknown')
        rows=[GroupedSample(f"{r['N']}:{r['K']}",Sample(route,(r['M'],r['N'],r['K']),
                      r['stock_max_rank_ms'],r['overlap_max_rank_ms'])) for r in records]
        out=[]
        for target in rows:
            train=[r for r in rows if r.family!=target.family]
            try:
                model=RoutePolicy(route).fit(train);p=model.predict(route,target.sample.shape)
                choice=p['choice'];audit=model.audit
            except ValueError as exc:
                choice='stock';audit={'reason':str(exc)}
            actual=target.sample.stock_ms/target.sample.candidate_ms
            out.append({'family':target.family,'shape':target.sample.shape,'choice':choice,
                        'oracle':'flux' if actual>1 else 'stock','speedup':actual if choice=='flux' else 1.,
                        'always_flux_speedup':actual,'audit':audit})
        sections[direction]={'count':len(out),'families':len({r.family for r in rows}),
            'correct':sum(r['choice']==r['oracle'] for r in out),
            'speedup_gmean':math.exp(statistics.mean(math.log(r['speedup']) for r in out)),
            'always_flux_gmean':math.exp(statistics.mean(math.log(r['always_flux_speedup']) for r in out)),
            'slower':sum(r['speedup']<1 for r in out),'rows':out}
    result={'classification':'exploratory TP4 Flux PDF grouped holdout, hardware unknown',
            'warning':'small N, AG single iteration, prior user data, no runtime claim or TP2 transfer',
            'source_sha256':hashlib.sha256(raw).hexdigest(),'sections':sections}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:{n:v[n] for n in ('count','families','correct','speedup_gmean','always_flux_gmean','slower')} for k,v in sections.items()},indent=2))


if __name__=='__main__':main()
