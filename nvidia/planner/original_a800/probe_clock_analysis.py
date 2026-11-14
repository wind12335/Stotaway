"""Join wall-window component probes to sampled nvidia-smi SM clocks.

Sparse samples describe observations, not exact clocks for every GPU kernel.
"""
import argparse
import csv
import datetime
import json
from collections import Counter
from pathlib import Path


def analyze(clock_file, reports):
    with Path(clock_file).open(newline='') as stream:
        rows=list(csv.DictReader(stream,skipinitialspace=True))
    samples=[]
    for r in rows:
        stamp=datetime.datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f').timestamp()
        samples.append((stamp,r['index'],r['clocks.current.sm [MHz]']))
    output=[]
    for path in reports:
        d=json.loads(Path(path).read_bytes());components={}
        for name,value in (d.get('components') or {}).items():
            start,end=value['rank0_wall_window_unix_s']
            if start>end:raise ValueError('invalid time window')
            counts={}
            for gpu in ('0','1'):
                counts[gpu]=dict(Counter(clock for t,g,clock in samples if g==gpu and start<=t<=end))
            components[name]={'median_ms':value['median_ms'],'clock_counts':counts}
        output.append({'path':str(path),'seed':d['seed'],'median_speedup':d['median_speedup'],'components':components})
    return {'classification':'sampled clock and component window diagnostics','reports':output,
            'warning':'time sampling is sparse and asynchronous; no causal inference or per-kernel clock guarantee'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clocks',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('reports',nargs='+',type=Path);a=ap.parse_args()
    result=analyze(a.clocks,a.reports)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
