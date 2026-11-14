"""Build an opt-in TP2 Qwen operator-model execution bundle."""
import argparse
import hashlib
import json
from pathlib import Path

from .export_paircal_mask import RUNTIME, WORKLOAD
from .tp2_online_bundle import build_bundle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',nargs='+',type=Path,required=True)
    ap.add_argument('--capabilities',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--hybrid',action='store_true')
    ap.add_argument('--runtime',type=Path,help='explicit runtime JSON; defaults to historical diagnostic')
    ap.add_argument('--workload',type=Path,help='explicit workload JSON; defaults to historical diagnostic')
    a=ap.parse_args()
    runtime=json.loads(a.runtime.read_text()) if a.runtime else RUNTIME
    workload=json.loads(a.workload.read_text()) if a.workload else WORKLOAD
    payload=build_bundle(a.train,a.capabilities,runtime,workload,hybrid=a.hybrid)
    raw=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(raw).hexdigest(),
                      'models':list(payload['models']),'queries':len(payload['queries'])},indent=2))


if __name__=='__main__':main()
