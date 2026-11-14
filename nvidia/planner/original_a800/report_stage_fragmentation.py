"""Describe split/full GEMM component timings on frozen target experiment."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    evaluation=json.loads(a.evaluation.read_bytes());stage={}
    for p in a.stages:
        data=json.loads(p.read_bytes())
        for r in data['results']:
            k=(r['model'],r['site'],r['M'],r['K'],r['N'])
            if k in stage:raise ValueError('duplicate stage record')
            s=r['measurements'];stage[k]={'split_ratio':s['split']['median_ms']/s['full']['median_ms'],
                                        'elapsed_wall_ms':r['elapsed_wall_ms']}
    rows=[]
    for r in evaluation['rows']:
        key=tuple(r['identity']);s=stage[key]
        rows.append({'identity':key,'split_ratio':s['split_ratio'],'elapsed_wall_ms':s['elapsed_wall_ms'],
                     'actual_speedup':r['actual_speedup'],'prediction_correct':r['choice']==r['oracle']})
    correct=[r['split_ratio'] for r in rows if r['prediction_correct']];wrong=[r['split_ratio'] for r in rows if not r['prediction_correct']]
    out={'classification':'exploratory split GEMM diagnostic on frozen unseen-family evaluation',
         'warning':'post-hoc association, sequential split is not full overlapping execution; no threshold selected or validation',
         'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [a.evaluation,*a.stages]],
         'summary':{'count':len(rows),'correct_count':len(correct),'wrong_count':len(wrong),
                    'median_split_ratio_correct':statistics.median(correct),'median_split_ratio_wrong':statistics.median(wrong),
                    'median_stage_wall_ms':statistics.median(r['elapsed_wall_ms'] for r in rows)},'rows':rows}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':main()
