"""Create a provenance-bound operator model for one observed TP2 route.

This exports an exploratory operator-only profile; it is not a training step
plan and must not be enabled automatically in Megatron/vLLM.
"""
import argparse
import json
from pathlib import Path

from .orthogonal_predictor import RouteRegressor, Sample
from .validate_orthogonal_tp2 import route_for
from .validate_tp2_new import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('inputs',nargs='+',type=Path)
    ap.add_argument('--op',choices=('rs','ag_up','ag_gu'),required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cells,signature,sources=load(a.inputs);rows=[r for r in cells if r['op']==a.op]
    route=route_for(signature,a.op)
    samples=[Sample(route,(r['M'],r['N'],r['K']),r['stock_b2b'],r['dx_b2b']) for r in rows]
    model=RouteRegressor(route).fit(samples)
    payload={'classification':'experimental operator-only calibration; not a validated E2E plan',
             'sources':sources,'source_count':len(rows),'families':len({r['model'] for r in rows}),
             'model':model.to_dict()}
    with a.output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(f"saved {route.method} {a.op}: {len(rows)} rows, {payload['families']} families")


if __name__=='__main__':main()
