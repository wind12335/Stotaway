"""Plumbing check of the OPRS runtime on BW (any card count; the 1-card run is a smoke, not a speed test).

Every candidate method (deo / async_tp / triton_dist) x direction (ag / rs) goes through the REAL
dispatcher and autograd wrappers and is compared with the serial path:
  forward      Dispatcher.run()            vs Dispatcher.serial()
  training     row_parallel / column_parallel (fused forward + backward) vs analytic gradients
  fallback     a shape that is not in the plan -> serial, counted with its reason
  release      Dispatcher.close(), then a second dispatcher on the same heap

Plans are written by hand into a scratch dir (no measured sample behind them -- this is NOT how
deployment plans are made; see export_plan.py). Nothing is timed here.

  python3 bench/oprs_smoke.py                       under `mpirun -np R` (env/env.sh, SW_NP=R)
  --methods deo,async_tp,triton_dist  --m 4096  --deo-bwd ag|stock  --dtype bfloat16|float16
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from stowaway.oprs import dispatcher as D
from stowaway.oprs.autograd import column_parallel, row_parallel
from stowaway.oprs.auto_plan import SCHEMA
from stowaway.oprs.engine import EngineWorld
from stowaway.oprs.platform import BACKENDS, DEO, METHODS, runtime_platform
from stowaway.oprs.predictor import SERIAL


def plan_dict(method, op, tp, dtype, m, n, k):
    site = {'ag': 'qkv', 'rs': 'proj'}[op]
    return {'schema': SCHEMA, 'platform': runtime_platform(tp=tp), 'workload': 'smoke', 'tp': tp,
            'dtype': dtype, 'external_backend': 'none' if method == DEO else method, 'default': SERIAL,
            'candidates': [SERIAL, method],
            'entries': [{'site': site, 'op': op, 'M': m, 'N': n, 'K': k, 'choice': method,
                         'predicted_log_speedup': 0.0, 'scores': {SERIAL: 0.0, method: 0.0}}],
            'allocate': [[method, op, m, n, k]]}


def rel(x, ref):
    return ((x.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6)).item()


def main():
    parser = argparse.ArgumentParser(description='OPRS runtime plumbing check on BW')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--m', type=int, default=4096, help='global rows (triton_dist: M/TP a multiple of 1024)')
    parser.add_argument('--dtype', default='bfloat16', choices=('bfloat16', 'float16'))
    parser.add_argument('--deo-bwd', default='ag', choices=('ag', 'stock'))
    parser.add_argument('--tol', type=float, default=2e-2)
    parser.add_argument('--port', type=int, default=29987)
    args = parser.parse_args()
    world = EngineWorld.standalone(port=args.port)
    tp, rank = world.size, world.rank
    import torch
    import torch.distributed as dist
    dt = getattr(torch, args.dtype)
    M = args.m
    shapes = {'ag': (M, 2048, 1024), 'rs': (M, 1024, 2048)}          # (M, N, K)
    scratch = Path(tempfile.mkdtemp(prefix=f'oprs_smoke_{rank}_'))
    failures = 0
    for method in [x for x in args.methods.split(',') if x]:
        for op, (m, n, k) in shapes.items():
            path = scratch / f'{method}_{op}.json'
            path.write_text(json.dumps(plan_dict(method, op, tp, args.dtype, m, n, k)))
            tag = f'[{method}/{op}] rank={rank}'
            try:
                disp = D.Dispatcher(path, 'smoke', group=world.group, methods=(method,), deo_bwd=args.deo_bwd,
                                    engine=world)
            except Exception as exc:                                  # noqa: BLE001
                if method in BACKENDS:
                    print(f'{tag} NOT AVAILABLE: {type(exc).__name__}: {exc}', flush=True)
                    continue
                raise
            ex = disp.exchanges[(method, op, m, n, k)]
            torch.manual_seed(100 + rank)
            site = {'ag': 'qkv', 'rs': 'proj'}[op]
            rows = m // tp if op == 'ag' else m
            cols = k if op == 'ag' else k // tp
            # ---- forward through the dispatcher -------------------------------
            x = (torch.randn(rows, cols, dtype=dt, device='cuda') * 0.05)
            w = (torch.randn(n, cols, dtype=dt, device='cuda') * 0.05)
            with torch.no_grad():
                ref = disp.serial(op, x, w)
                out = disp.run(site, op, x, w)
            torch.cuda.synchronize()
            r_fwd = rel(out, ref)
            # ---- training: fused forward + backward vs analytic gradients ------
            a = (torch.randn(rows, cols, dtype=torch.float32, device='cuda') * 0.05).to(dt).requires_grad_()
            wa = (torch.randn(n, cols, dtype=torch.float32, device='cuda') * 0.05).to(dt).requires_grad_()
            g = (torch.randn(out.shape[0], n, dtype=torch.float32, device='cuda') * 0.05).to(dt)
            if op == 'rs':
                full_g = torch.empty(m, n, dtype=dt, device='cuda')
                dist.all_gather_into_tensor(full_g, g)
                ref_a, ref_w = torch.matmul(full_g, wa.detach()), full_g.t().matmul(a.detach())
                y = row_parallel(a, wa, ex, world.group)
                y.backward(g)
                kind = ex.backward_kind
            else:
                full_x = torch.empty(m, cols, dtype=dt, device='cuda')
                dist.all_gather_into_tensor(full_x, a.detach())
                grad_x = torch.matmul(g, wa.detach())
                ref_a, ref_w = torch.empty_like(a), g.t().matmul(full_x)
                dist.reduce_scatter_tensor(ref_a, grad_x.contiguous(), group=world.group)
                y = column_parallel(a, wa, ex, world.group)
                y.backward(g)
                kind = 'stock'
            torch.cuda.synchronize()
            r_a, r_w = rel(a.grad, ref_a), rel(wa.grad, ref_w)
            finite = bool(torch.isfinite(a.grad).all() and torch.isfinite(wa.grad).all())
            # ---- fallback: a shape the plan does not know ---------------------
            xs, ws = x[:rows // 2].contiguous(), w[:n // 2].contiguous()
            with torch.no_grad():
                out_small = disp.run(site, op, xs, ws)
                ref_small = disp.serial(op, xs, ws)
            r_fb = rel(out_small, ref_small)
            # ---- host cost of one routing decision (the dispatcher's own per-call overhead; handoff 3.4).
            # peek() is what exchange_for() does before it hands out a context, and it counts nothing.
            def per_call_us(fn, calls=20000):
                fn()
                start = time.perf_counter()
                for _ in range(calls):
                    fn()
                return (time.perf_counter() - start) / calls * 1e6
            hit_us = per_call_us(lambda: disp.peek(site, op, x, w))                # a shape the plan has
            miss_us = per_call_us(lambda: disp.peek(site, op, xs, ws))            # a shape it does not have
            ok = max(r_fwd, r_a, r_w, r_fb) < args.tol and finite
            failures += not ok
            print(f'{tag} M={m} N={n} K={k} {"PASS" if ok else "FAIL"} fwd={r_fwd:.2e} '
                  f'grad_a={r_a:.2e} grad_w={r_w:.2e} (bwd={kind}) fallback={r_fb:.2e} finite={finite} '
                  f'runs={dict(disp.runs)}', flush=True)
            print(f'{tag} host overhead per call: peek(hit) {hit_us:.2f} us, peek(unknown shape) {miss_us:.2f} us',
                  flush=True)
            disp.print_report()
            disp.close()
    # heap released by close() must be usable again
    path = scratch / 'again.json'
    path.write_text(json.dumps(plan_dict(DEO, 'rs', tp, args.dtype, *shapes['rs'])))
    disp = D.Dispatcher(path, 'smoke', group=world.group, methods=(DEO,), deo_bwd=args.deo_bwd, engine=world)
    with torch.no_grad():
        x = torch.randn(M, 2048 // tp, dtype=dt, device='cuda')
        w = torch.randn(1024, 2048 // tp, dtype=dt, device='cuda')
        r = rel(disp.run('proj', 'rs', x, w), disp.serial('rs', x, w))
    disp.close()
    print(f'[deo/rs-again] rank={rank} after close(): rel={r:.2e} heap_need_mb={disp.heap_bytes / 2**20:.0f} '
          f'{"PASS" if r < args.tol else "FAIL"}', flush=True)
    failures += r >= args.tol
    checks = [torch.tensor([failures], device='cuda')]
    dist.all_reduce(checks[0], op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f'OPRS_SMOKE {"PASS" if checks[0].item() == 0 else "FAIL"} failures={int(checks[0].item())} '
              f'tp={tp} dtype={args.dtype} deo_bwd={args.deo_bwd}', flush=True)
    try:
        world.mpi.MPI_Finalize()        # a rank that leaves without it is an "abnormal termination" to
    except Exception:                   # noqa: BLE001 - prterun, and mpirun would return 1 after a PASS
        pass
    sys.exit(1 if checks[0].item() else 0)


if __name__ == '__main__':
    main()
