"""Replay route-local residual on user-provided TP4 Flux logs, not A800 calibration."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from .orthogonal_predictor import Route, Sample
from .route_policy import GroupedSample
from .route_local_residual import RouteLocalResidual
from .orthogonal_predictor import RouteRegressor


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();raw=a.input.read_bytes();d=json.loads(raw);sections={}
    for op in ('ag','rs'):
        route=Route('flux','unknown','unknown','unknown','unknown',op,'pdf_reported_flux',
                    'unknown','unknown',4,'float16',(('source','user_pdf_unknown_version'),),
                    'single_or_10_iters_pdf',None,'pdf_reported_unknown')
        rows=[GroupedSample(f"{r['N']}:{r['K']}",Sample(route,(r['M'],r['N'],r['K']),
              r['stock_max_rank_ms'],r['overlap_max_rank_ms'])) for r in d['rows'] if r['method']=='flux' and r['op']==op]
        results=[]
        for held in sorted({r.family for r in rows}):
            train=[r for r in rows if r.family!=held]
            local_available=len({r.family for r in train})>=4
            model=(RouteLocalResidual(route).fit(train) if local_available else
                   RouteRegressor(route).fit([r.sample for r in train]))
            for r in rows:
                if r.family!=held:continue
                s=r.sample;actual=s.stock_ms/s.candidate_ms
                if local_available:pred=model.predict(route,s.shape)
                else:
                    gain=model.predict(route,s.shape)
                    pred={'predicted_log_speedup':gain,'base_log_speedup':gain,'status':'insufficient families: base fallback'}
                results.append({'shape':s.shape,'family':held,'prediction':pred,'actual_speedup':actual})
        def score(field):
            choices=[r['prediction'][field]>0 for r in results];speeds=[r['actual_speedup'] if c else 1. for r,c in zip(results,choices)]
            return {'count':len(results),'correct':sum(c==(r['actual_speedup']>1) for r,c in zip(results,choices)),
                    'slower':sum(s<1 for s in speeds),'speedup_gmean':math.exp(statistics.mean(map(math.log,speeds)))}
        sections[op]={'local':score('predicted_log_speedup'),'base':score('base_log_speedup'),'rows':results}
    out={'classification':'user Flux TP4 PDF historical group holdout; device and transport unknown',
         'warning':'small historical dataset, AG single iteration, not deployable TP2 or source of GPU route metadata',
         'source_sha256':hashlib.sha256(raw).hexdigest(),'sections':sections}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:{x:v[x] for x in ('local','base')} for k,v in sections.items()},indent=2))


if __name__=='__main__':main()
