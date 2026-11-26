"""Bit-for-bit agreement of each overlap method with the vendor GEMM (one process per card count).

  bash bench/run_oprs_bitwise.sh [--shapes tiny_vllm | bw27b | tiny_megatron | site:op:M:N:K,...]
                                 [--methods deo,async_tp,triton_dist] [--json out.jsonl]

For every (site, method) the fused operator of the runtime dispatcher is run on seeded random bf16
operands next to four vendor GEMMs on the same operands, and the output is compared BIT for BIT:
    B^T     torch.matmul(x, w.t())            w [N, K] row-major, transposed view   (= F.linear)
    B       torch.matmul(x, w_kn)             w_kn [K, N] row-major                 (the das fork's weight layout)
Nothing is timed. What it says on BW1000 / DTK 26.04 (one card, bf16, the shapes of the two tiny
vehicles: results/bw1000_oprs_20261126/raw/smoke/oprs_bitwise_tp1.log):
  * the vendor GEMM is not one arithmetic: B^T and B differ in about 0.02 % of the output elements
    (0 at K = 512) by one bf16 step;
  * DEO and Async-TP reproduce B^T exactly, Triton-distributed reproduces B exactly -- at every site;
  * so which method is bit-identical to a framework's stock path follows from where that framework
    keeps its weights: vLLM (das fork) stores [K, N] and computes with B, and its three overlap arms
    are bit-identical to stock (zero prompt-logprob difference in the seeded-weights gate);
    Megatron stores [N, K] and computes with B^T, DEO and Async-TP are bit-identical to it, and
    Triton-distributed is not (training loss 1.182467 against 1.182455 in the single-card smoke).
At TP=1 the fused operators are the plain GEMM, so agreement is expected and any difference is a
finding. At TP>1 the stock path (partial GEMMs + AllReduce) and the fused one (GEMM on the gathered
rows) sum in different orders, so a count of differing elements is information, not a failure; the
correctness bar of record is the operator bench (bench/oprs_matrix.py). Nothing here is timed.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

# site -> (op, M, N, K)
SHAPES = {
    'tiny_vllm': {'qkv': ('ag', 2048, 1536, 1024), 'proj': ('rs', 2048, 1024, 512),
                  'fc1': ('ag', 2048, 7168, 1024), 'fc2': ('rs', 2048, 1024, 3584)},
    'bw27b': {'qkv': ('ag', 32768, 896, 5120), 'proj': ('rs', 32768, 5120, 640),
              'fc1': ('ag', 32768, 4352, 5120), 'fc2': ('rs', 32768, 5120, 2176)},
    'tiny_megatron': {'qkv': ('ag', 4096, 1536, 1024), 'proj': ('rs', 4096, 1024, 1024),
                      'fc1': ('ag', 4096, 4096, 1024), 'fc2': ('rs', 4096, 1024, 2048)},
}


def parse_shapes(text):
    out = {}
    for item in text.split(','):
        if item in SHAPES:
            out.update(SHAPES[item])
            continue
        site, op, m, n, k = item.split(':')                      # site:op:M:N:K
        out[site] = (op, int(m), int(n), int(k))
    return out


def plan(method, tp, sites, op):
    from stowaway.oprs.auto_plan import SCHEMA
    from stowaway.oprs.platform import runtime_platform
    from stowaway.oprs.predictor import SERIAL
    mine = {s: v for s, v in sites.items() if v[0] == op}
    return {'schema': SCHEMA, 'platform': runtime_platform(tp=tp), 'workload': 'bitwise', 'tp': tp,
            'dtype': 'bfloat16', 'external_backend': 'none' if method == 'deo' else method, 'default': SERIAL,
            'candidates': [SERIAL, method],
            'entries': [{'site': s, 'op': op, 'M': m, 'N': n, 'K': k, 'choice': method, 'predicted_log_speedup': 0.0,
                         'scores': {SERIAL: 0.0, method: 0.0}} for s, (o, m, n, k) in mine.items()],
            'allocate': [[method, op, m, n, k] for s, (o, m, n, k) in mine.items()]}


def main():
    parser = argparse.ArgumentParser(description='bit-for-bit agreement of the overlap methods with the vendor GEMM')
    parser.add_argument('--shapes', default='tiny_vllm',
                        help='comma list: a named set (tiny_vllm | bw27b | tiny_megatron) or site:op:M,N,K')
    parser.add_argument('--methods', default='deo,async_tp,triton_dist')
    parser.add_argument('--seed', type=int, default=20261126)
    parser.add_argument('--scale', type=float, default=0.5, help='operand scale (fp32 normal, then bf16)')
    parser.add_argument('--port', type=int, default=29989)
    parser.add_argument('--json', help='write the result rows to this file')
    args = parser.parse_args()
    from stowaway.oprs import dispatcher as D
    from stowaway.oprs.engine import EngineWorld
    world = EngineWorld.standalone(port=args.port)
    tp, rank = world.size, world.rank
    import torch
    sites = parse_shapes(args.shapes)
    rows = []

    def neq(a, b):
        return a.numel() - int((a.view(torch.int16) == b.view(torch.int16)).sum())

    ranks = world.size
    for method in [m for m in args.methods.split(',') if m]:
        for op in ('ag', 'rs'):
            path = os.path.join(tempfile.mkdtemp(prefix=f'oprs_bitwise_{rank}_'), f'{method}_{op}.json')
            with open(path, 'w') as handle:
                json.dump(plan(method, tp, sites, op), handle)
            disp = D.Dispatcher(path, 'bitwise', group=world.group, methods=(method,), deo_bwd='stock',
                                engine=world)
            with torch.no_grad():
                for index, (site, (o, m, n, k)) in enumerate(sites.items()):
                    if o != op:
                        continue
                    # the same operands on every rank (one seed, no Python hash): the rank's shard of a
                    # global [M, K] (ag) or [M, K x ranks] (rs) matrix is what a call site hands over
                    gen = torch.Generator(device='cuda').manual_seed(args.seed + index)

                    def normal(*shape):
                        return (torch.randn(*shape, generator=gen, device='cuda', dtype=torch.float32)
                                * args.scale).to(torch.bfloat16)
                    w = normal(n, k)
                    rows_m = m // ranks
                    if op == 'ag':
                        full = normal(m, k)
                        local = full[rank * rows_m:(rank + 1) * rows_m]
                        whole_t, whole_n = torch.matmul(full, w.t()), torch.matmul(full, w.t().contiguous())
                    else:
                        full = normal(m, k * ranks)
                        local = full[:, rank * k:(rank + 1) * k]
                        whole_t, whole_n = torch.matmul(local, w.t()), torch.matmul(local, w.t().contiguous())
                    ref_t = whole_t[rank * rows_m:(rank + 1) * rows_m]      # what the rank keeps
                    ref_n = whole_n[rank * rows_m:(rank + 1) * rows_m]
                    out = disp.run(site, op, local, w)
                    torch.cuda.synchronize()
                    counts = torch.tensor([neq(out, ref_t), neq(out, ref_n), neq(ref_t, ref_n)], dtype=torch.int64,
                                          device='cuda')
                    torch.distributed.all_reduce(counts, group=world.group)   # summed over the ranks
                    row = {'tp': tp, 'method': method, 'site': site, 'op': op, 'M': m, 'N': n, 'K': k,
                           'elements': out.numel() * ranks, 'differing_vs_BT': int(counts[0]),
                           'differing_vs_B': int(counts[1]), 'vendor_BT_vs_B': int(counts[2]),
                           'max_abs_diff_vs_BT': (out.float() - ref_t.float()).abs().max().item(),
                           'max_abs_diff_vs_B': (out.float() - ref_n.float()).abs().max().item()}
                    rows.append(row)
                    if rank == 0:
                        print(f"[bitwise] {method:<12} {site:<5} {op} M={m} N={n} K={k} TP={tp}: differing elements "
                              f"of {row['elements']}: vs B^T {row['differing_vs_BT']}, vs B {row['differing_vs_B']} "
                              f"(vendor itself, B^T vs B: {row['vendor_BT_vs_B']})", flush=True)
            disp.close()
    if args.json and rank == 0:
        with open(args.json, 'w') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
    torch.distributed.barrier(group=world.group)
    try:
        world.mpi.MPI_Finalize()                 # standalone() brought the MPI world up: leave it properly
    except Exception:                            # noqa: BLE001
        pass


if __name__ == '__main__':
    main()
