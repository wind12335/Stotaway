"""Execute saved three-candidate TP2 plans and verify real dispatch."""
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from stowaway_nvidia.replacement_plan import ReplacementExecutor


def reference(op, x, w):
    if op == 'ag':
        full = torch.empty((x.shape[0] * 2, x.shape[1]), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(full, x)
        return full @ w.T
    partial = x @ w.T
    out = torch.empty((x.shape[0] // 2, w.shape[0]), device=x.device, dtype=x.dtype)
    dist.reduce_scatter_tensor(out, partial)
    return out


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    plan = Path(os.environ['REPLACEMENT_PLAN'])
    mode = os.environ.get('REPLACEMENT_TEST_MODE', 'coverage')
    out = Path(os.environ['REPLACEMENT_RESULT'])
    if mode == 'mismatch':
        if rank == 1:
            plan = Path(os.environ['REPLACEMENT_OTHER_PLAN'])
        try:
            ReplacementExecutor(plan)
            raise AssertionError('rank mismatch not rejected')
        except RuntimeError as exc:
            if 'plan mismatch' not in str(exc):
                raise
        print(f'rank={rank} plan_mismatch_rejected=PASS', flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return
    exe = ReplacementExecutor(plan)
    gen = torch.Generator(device='cuda').manual_seed(6020 + rank)
    w = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda', generator=gen) * .05
    calls = ([('qkv', 'ag'), ('o_proj', 'rs'), ('fc1', 'ag'), ('fc2', 'rs'),
              ('unregistered', 'ag'), ('qkv', 'rs')] if mode == 'coverage'
             else [('measured_ag', 'ag'), ('measured_rs', 'rs')])
    retained = []
    checks = 0
    for turn in range(3):
        for site, op in calls:
            x = torch.randn((512 if op == 'ag' else 1024, 4096), dtype=torch.bfloat16,
                            device='cuda', generator=gen) * .05
            expected = reference(op, x, w)
            actual = exe.run(site, op, x, w)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
            retained.append((actual, actual.clone()))
            checks += 1
    for actual, saved in retained:
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    batched_calls, batch_expected = [], []
    for site, op in calls:
        bx = torch.randn((512 if op == 'ag' else 1024, 4096), dtype=torch.bfloat16,
                         device='cuda', generator=gen) * .05
        batched_calls.append((site, op, bx, w))
        batch_expected.append(reference(op, bx, w))
    batch_actual = exe.run_batch(batched_calls)
    for actual, expected in zip(batch_actual, batch_expected):
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    try:
        mismatch_calls = batched_calls if rank == 0 else list(reversed(batched_calls))
        exe.run_batch(mismatch_calls)
        raise AssertionError('rank batch mismatch not rejected')
    except RuntimeError as exc:
        if 'inconsistent call batch' not in str(exc):
            raise
    # Both ranks enter with different but individually valid metadata.  The
    # control group must reject before any data collective or backend call.
    x = torch.randn((512, 4096), dtype=torch.bfloat16, device='cuda', generator=gen)
    try:
        exe.run('qkv' if rank == 0 else 'fc1', 'ag', x, w)
        raise AssertionError('rank call mismatch not rejected')
    except RuntimeError as exc:
        if 'inconsistent call metadata' not in str(exc):
            raise
    expected_hits = ({'stock': 12, 'async_tp': 8, 'dx': 4} if mode == 'coverage'
                     else {'stock': 0, 'async_tp': 0, 'dx': 8})
    if exe.hits != expected_hits or exe.fallbacks != (8 if mode == 'coverage' else 0):
        raise AssertionError((exe.hits, exe.fallbacks, expected_hits))
    torch.cuda.synchronize()
    report = {'rank': rank, 'mode': mode, 'plan_sha256': exe.digest, 'hits': exe.hits,
              'fallbacks': exe.fallbacks, 'reference_checks': checks, 'retained_outputs': len(retained),
              'batch_reference_checks': len(batch_actual), 'rank_batch_mismatch_rejected': True,
              'rank_call_mismatch_rejected': True, 'selection_source': exe.plan['selection_source']}
    all_reports = [None, None]
    dist.all_gather_object(all_reports, report, group=exe.group)
    exe.close()
    if rank == 0:
        result = {'classification': 'TP2 three-candidate operator-carrier execution', 'reports': all_reports,
                  'warning': 'direct symmetric-memory ops, not Inductor model E2E; per-call rank metadata checks included'}
        with out.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
