"""Freeze one previously measured route's generic policy as JSON."""
import argparse
import hashlib
import json
from pathlib import Path

from .evaluate_route_policy import flash
from .route_policy import RoutePolicy


def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--flash-report',type=Path);group.add_argument('--triton-report',type=Path)
    ap.add_argument('--op',choices=('ag_up','ag_gu'));ap.add_argument('--hybrid',action='store_true')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    from .hybrid_route_policy import HybridRoutePolicy
    def fit(route,rows):return HybridRoutePolicy(route).fit(rows) if a.hybrid else RoutePolicy(route).fit(rows)
    if a.triton_report:
        from .validate_tp2_new import load
        from .validate_orthogonal_tp2 import route_for
        from .orthogonal_predictor import Sample
        from .route_policy import GroupedSample
        if not a.op:raise ValueError('triton route requires --op')
        raw=a.triton_report.read_bytes();report=json.loads(raw);paths=[Path(s['path']) for s in report['sources']]
        cells,signature,sources=load(paths);route=route_for(signature,a.op)
        rows=[GroupedSample(r['model'],Sample(route,(r['M'],r['N'],r['K']),r['stock_b2b'],r['dx_b2b']))
              for r in cells if r['op']==a.op]
        model=fit(route,rows)
        payload={'classification':'experimental route policy; operator only',
                 'warning':'prior data used in development; not validated E2E',
                 'source_report':{'path':str(a.triton_report),'sha256':hashlib.sha256(raw).hexdigest()},
                 'sources':sources,'policy':model.to_dict()}
        encoded=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        with a.output.open('xb') as f:f.write(encoded)
        print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(encoded).hexdigest(),
                          'audit':model.fallback.audit if a.hybrid and model.fallback else getattr(model,'audit',None)},indent=2))
        return
    raw=a.flash_report.read_bytes();report=json.loads(raw)
    rows=[];sources=[]
    for item in report['train_sources']:
        path=Path(item['path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=item['sha256']:raise ValueError('source digest mismatch')
        route,row=flash(path);rows.append(row);sources.append({'path':str(path),'sha256':digest})
    if not rows or any(r.sample.route!=route for r in rows):raise ValueError('mixed routes')
    model=fit(route,rows)
    payload={'classification':'experimental route policy; operator only',
             'warning':'prior data used in development; model validation and runtime integration still required',
             'source_report':{'path':str(a.flash_report),'sha256':hashlib.sha256(raw).hexdigest()},
             'sources':sources,'policy':model.to_dict()}
    encoded=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(encoded)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(encoded).hexdigest(),
                      'audit':model.fallback.audit if a.hybrid and model.fallback else getattr(model,'audit',None)},indent=2))


if __name__=='__main__':main()
