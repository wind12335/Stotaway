"""Execute a previously frozen forward plan on both GPUs with synthetic layers."""
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from stowaway_nvidia.vllm_sp import LlamaSequenceParallel


class QuantMethod:
    def apply(self, layer, value, bias):
        return value @ layer.weight.T


class Layer:
    def __init__(self, weight):
        self.weight = weight
        self.quant_method = QuantMethod()


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    workload = json.loads(Path(os.environ['PLAN_WORKLOAD']).read_text())
    # First check that a rank-local validation failure is communicated to both ranks.
    invalid_workload = workload if rank == 0 else {**workload, 'extra_mismatch': True}
    invalid = LlamaSequenceParallel(None, rank, 'dx', [1024], forward_plan=os.environ['PLAN_PATH'],
                                    plan_workload=invalid_workload)
    try:
        invalid._prepare_plan()
        raise AssertionError('rank workload mismatch accepted')
    except RuntimeError as exc:
        if 'forward plan invalid or inconsistent' not in str(exc):
            raise
    adapter = LlamaSequenceParallel(None, rank, 'dx', [1024], forward_plan=os.environ['PLAN_PATH'],
                                    plan_workload=workload)
    # Replace only vLLM's layer-type check; both actual execution methods remain unchanged.
    adapter._weight = lambda layer: layer.weight
    adapter._prepare_plan()
    sites = [('qkv', 'ag', 3456, 2496), ('proj', 'rs', 1728, 3456),
             ('fc1', 'ag', 3456, 9472), ('fc2', 'rs', 4736, 3456)]
    generator = torch.Generator(device='cuda').manual_seed(2000 + rank)
    for site, op, k, n in sites:
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16, generator=generator) * .05
        layer = Layer(w)
        if op == 'ag':
            x = torch.randn((512, k), device='cuda', dtype=torch.bfloat16, generator=generator) * .05
            full = torch.empty((1024, k), device='cuda', dtype=x.dtype)
            dist.all_gather_into_tensor(full, x)
            reference = layer.quant_method.apply(layer, full, None)
            actual = adapter._col(layer, x, 1024, dist.group.WORLD, site)
        else:
            x = torch.randn((1024, k), device='cuda', dtype=torch.bfloat16, generator=generator) * .05
            partial = layer.quant_method.apply(layer, x, None)
            reference = torch.empty((512, n), device='cuda', dtype=x.dtype)
            dist.reduce_scatter_tensor(reference, partial)
            actual = adapter._row(layer, x, 1024, dist.group.WORLD, site)
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)
        torch.cuda.synchronize()
        del actual, reference, x, w, layer
    choices = [{'site': k[0], 'M': k[1], 'K': k[2], 'N': k[3], 'choice': k[4], 'count': v}
               for k, v in adapter.plan_decisions.items()]
    all_choices = [None] * 2
    dist.all_gather_object(all_choices, choices)
    if all_choices[0] != all_choices[1] or {c['choice'] for c in choices} != {'dx', 'stock'}:
        raise RuntimeError(f'invalid rank decisions: {all_choices}')
    if {c['site']: c['choice'] for c in choices} != {'qkv':'dx', 'proj':'dx', 'fc1':'dx', 'fc2':'stock'}:
        raise RuntimeError(f'decisions do not match frozen plan: {choices}')
    if rank == 0:
        result = {'classification': 'synthetic TP2 forward routing correctness smoke, not E2E performance',
                  'plan_sha256': adapter.lookup.digest, 'rank_choices': all_choices,
                  'tolerance': {'rtol': .02, 'atol': .02}, 'rank_mismatch_rejected': True}
        with Path(os.environ['PLAN_TEST_OUT']).open('x') as stream:
            json.dump(result, stream, indent=2);stream.write('\n')
        print(json.dumps(result), flush=True)
    adapter.close()
    if adapter.cuda is not None:
        import nvshmem.core
        nvshmem.core.finalize()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
