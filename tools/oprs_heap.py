"""Symmetric-heap size (DUSHMEM_SYMMETRIC_SIZE) for the OPRS arms of one comparison.

  python3 tools/oprs_heap.py <plan.json | plan-set dir | manifest.json> [...] [--deo-bwd ag|stock]
                             [--scale 1.25] [--extra-gib 1]

Prints one number: the heap bytes to export for EVERY arm of the comparison (the heap is reserved before
the framework sizes its own memory, so all arms get the same size). It is
    scale x (largest need among the plans)  +  extra,
where a plan's need is platform.heap_bytes summed over its allocations -- the same figure the
dispatcher checks before it allocates anything. A plan-set directory is what tools/oprs_plans.py wrote.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from stowaway.oprs.platform import allocations_heap_bytes  # noqa: E402


def plan_files(target):
    """Plan files named by one argument: a plan file, a plan-set directory or its manifest.json."""
    path = Path(target)
    if path.is_dir():
        path = path / 'manifest.json'
    if not path.exists():
        sys.exit(f'not found: {target}')
    if path.name == 'manifest.json':
        data = json.loads(path.read_text())
        return [path.parent / item['path'] for item in data['plans'].values()]
    return [path]


def main(argv=None):
    parser = argparse.ArgumentParser(description='symmetric-heap bytes for a set of OPRS plans')
    parser.add_argument('plans', nargs='+', help='plan.json, plan-set directory or manifest.json')
    parser.add_argument('--deo-bwd', default='ag', choices=('ag', 'stock'),
                        help='training default ag (row sites hold the fused AG+dgrad context too)')
    parser.add_argument('--scale', type=float, default=1.25)
    parser.add_argument('--extra-gib', type=float, default=1.0)
    args = parser.parse_args(argv)
    files = [f for target in args.plans for f in plan_files(target)]
    largest = (0, None, None)
    for file in files:
        plan = json.loads(file.read_text())
        need = allocations_heap_bytes([tuple(item) for item in plan['allocate']], plan['tp'],
                                      deo_bwd=args.deo_bwd == 'ag')
        if need > largest[0]:
            largest = (need, file, plan['tp'])
    if largest[1] is None:
        sys.exit('no plan file found')
    need, file, tp = largest
    size = int(need * args.scale + args.extra_gib * 2**30)
    print(f'# largest need {need / 2**20:.0f} MiB (tp={tp}, deo_bwd={args.deo_bwd}) in {file}; '
          f'{len(files)} plans considered', file=sys.stderr)
    print(size)


if __name__ == '__main__':
    main()
