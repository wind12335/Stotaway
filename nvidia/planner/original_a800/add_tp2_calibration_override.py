"""Add checked exact-shape training measurements to an online bundle."""
import argparse
import hashlib
import json
from pathlib import Path
from ..tp2_lookup import build


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    payload=json.loads(a.bundle.read_bytes());raw=a.calibration.read_bytes()
    lookup=build([a.calibration],objective='training')
    if payload['runtime']!=lookup['runtime']:raise ValueError('runtime mismatch')
    calibration=json.loads(raw);md=calibration['metadata']
    workload=payload['workload_requirements']
    if md.get('M')!=workload['seq_length']*workload['micro_batch_size']:
        raise ValueError('calibration shape/workload mismatch')
    if md.get('batch')!=16 or md.get('warmup')!=3:
        raise ValueError('override requires the checked b2b16/warmup3 protocol')
    for r in calibration['results']:
        from ..tp2_lookup import QWEN_SITES
        site=QWEN_SITES.get(r['site'],r['site']);model=payload['models'].get(site,{})
        route=model.get('route',{})
        if route.get('timing_mode')!='b2b16' or route.get('warmup') not in (None,3):
            raise ValueError('calibration and model timing protocol differ')
    supported={f"{q['site']}:{q['shape'][0]}:{q['shape'][2]}:{q['shape'][1]}" for q in payload['queries']}
    if not set(lookup['entries'])<=supported:raise ValueError('unknown shapes')
    digest=hashlib.sha256(raw).hexdigest();overrides=payload.setdefault('overrides',{})
    for name,entry in lookup['entries'].items():
        if name in overrides:raise ValueError('override already exists')
        overrides[name]={'choice':entry['choice'],'source_sha256':digest,'source_warmup':md['warmup'],
                         'model_warmup_unknown':payload['models'][name.split(':')[0]]['route']['warmup'] is None,
                         'guard':lookup['guard'],'stock_lower_ms':entry['stock_lower_ms'],
                         'dx_upper_ms':entry['dx_upper_ms']}
    encoded=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(encoded)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(encoded).hexdigest(),'overrides':overrides},indent=2))


if __name__=='__main__':main()
