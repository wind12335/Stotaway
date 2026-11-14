"""Post-hoc check of local residual dispersion as an optional selection guard."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def evaluate(development_path, decision_path, evaluation_paths, multipliers=(0.,.5,1.,2.)):
    dev=json.loads(Path(development_path).read_bytes());dec=json.loads(Path(decision_path).read_bytes())
    dsha=hashlib.sha256(Path(decision_path).read_bytes()).hexdigest()
    evals=[]
    for p in evaluation_paths:
        e=json.loads(Path(p).read_bytes())
        if e['frozen_sha256']!=dsha:raise ValueError('decision digest mismatch')
        rows={tuple(x['identity']):x for x in e['rows']}
        if len(rows)!=len(e['rows']):raise ValueError('duplicate evaluation identity')
        evals.append(rows)
    predictions={tuple(x['identity']):x['prediction'] for x in dec['predictions']}
    if any(set(e)!=set(predictions) for e in evals):raise ValueError('inventory mismatch')
    def quantify(pred):
        ns=pred['neighbors']
        if len(ns)<2 or len({x['family'] for x in ns})!=len(ns):raise ValueError('distinct neighbor families required')
        values=[x['residual'] for x in ns]
        median=statistics.median(values)
        return statistics.median(abs(x-median) for x in values)
    def score(rows,multiplier):
        choices=[p['predicted_log_speedup']>multiplier*quantify(p) for p,_ in rows]
        speeds=[s if c else 1. for c,(_,s) in zip(choices,rows)]
        return {'count':len(rows),'correct':sum(c==(s>1) for c,(_,s) in zip(choices,rows)),
                'slower':sum(s<1 for s in speeds),'dx_selected':sum(choices),
                'speedup_gmean':math.exp(statistics.mean(math.log(s) for s in speeds))}
    groups={'development':[(r['prediction'],r['actual_speedup']) for r in dev['rows']]}
    for index,ev in enumerate(evals):groups[f'evaluation_{index+1}']=[(predictions[k],r['actual_speedup']) for k,r in ev.items()]
    return {'classification':'post-hoc neighbor-residual MAD heuristic',
            'summary':{g:{str(m):score(rows,m) for m in multipliers} for g,rows in groups.items()},
            'sources':[{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                       for p in (development_path,decision_path,*evaluation_paths)],
            'warning':'all results already inspected; multipliers uncalibrated; repeated points are not new shapes; '
                      'neighbor dispersion is not predictive confidence; costs excluded'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--development',required=True,type=Path);ap.add_argument('--decisions',required=True,type=Path)
    ap.add_argument('--evaluations',nargs='+',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();result=evaluate(a.development,a.decisions,a.evaluations)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
