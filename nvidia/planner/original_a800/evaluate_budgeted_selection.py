"""Replay a one-block, two-candidate diagnostic against later held-out blocks.

Only previously recorded full blocks are available: masking candidates here
simulates a smaller search but does not measure its actual elapsed cost.
The first calibration block has already been inspected in prior development.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=a.report.read_bytes();report=json.loads(raw);profile=json.loads(a.profile.read_bytes())
    selected=profile['model_audit']['selection']['best']['sites']
    mapping={(): 'stock',('proj',):'proj',('fc2',):'fc2',('fc2','proj'):'pair'}
    choice=mapping[tuple(selected)]
    if choice=='stock':raise ValueError('replay requires a non-stock frozen candidate')
    blocks=report['blocks']
    if len(blocks)<3:raise ValueError('three blocks required')
    calibration=blocks[0];delta=calibration['median_delta_ms'][choice]
    # Fixed rule for this retrospective replay: retain the candidate only if
    # the first block favors it. All three blocks were already inspected.
    picked=choice if delta<0 else 'stock'
    held=blocks[1:];results=[]
    for b in held:
        results.append({'sequence':b['sequence'],'choice':picked,'oracle':b['oracle'],
                        'speedup':b['paired_speedup'][picked],
                        'fixed_overlap_speedup':b['paired_speedup'][choice],
                        'fixed_pair_speedup':b['paired_speedup']['pair']})
    gm=lambda field:math.exp(statistics.mean(math.log(r[field]) for r in results))
    out={'classification':'exploratory simulated restricted search, earlier inspected data',
         'warning':'not a prospective test; all four arms actually ran and real reduced-calibration cost is unknown',
         'source_sha256':hashlib.sha256(raw).hexdigest(),'profile_sha256':hashlib.sha256(a.profile.read_bytes()).hexdigest(),
         'candidates':['stock',choice],'calibration_block':calibration['sequence'],'calibration_delta_ms':delta,
         'choice':picked,'evaluation':results,'correct':sum(r['choice']==r['oracle'] for r in results),
         'summary':{'count':len(results),'speedup_gmean':gm('speedup'),
                    'fixed_overlap_gmean':gm('fixed_overlap_speedup'),'fixed_pair_gmean':gm('fixed_pair_speedup')}}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:out[k] for k in ('candidates','calibration_delta_ms','choice','correct','summary')},indent=2))


if __name__=='__main__':main()
