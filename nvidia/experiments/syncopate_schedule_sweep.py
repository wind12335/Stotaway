"""CPU-only upstream Syncopate schedule feature sweep; no GPU performance."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sys.path.insert(0,str(a.repo))
    import torch
    from syncopate.communication.code_gen import CommGenerator
    from syncopate.communication.common_descriptors.all_gather import build_all_gather_plan_1d_swizzle
    from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
    from planner.original_a800.syncopate_plan_features import summarize
    rows=[]
    for tp in (2,4,8):
        for m in (512,1024,2048,4096,8192):
            plans={rank:build_all_gather_plan_1d_swizzle(shape=(m,5120),dtype=torch.bfloat16,axis=0,
                   mesh_size=tp,rank=rank,buffer_name='x',transfer_kind='pull') for rank in range(tp)}
            gen=CommGenerator(plans);gen.plan_signals();schedule=lower_comm_plan_to_raw_schedules(gen)
            features=summarize(schedule,buffer_name='x',k=5120,n_local=7168,dtype_bytes=2)
            rows.append({'tp':tp,'M':m,'K':5120,'N_local':7168,'per_rank':[asdict(f) for f in features]})
    result={'classification':'CPU upstream Syncopate plan structure only','rows':rows,
            'warning':'not runtime transport, GPU correctness, timings, or predictor calibration; output width held fixed across TP'}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'rows':len(rows),'output':str(a.output)}))


if __name__=='__main__':main()
