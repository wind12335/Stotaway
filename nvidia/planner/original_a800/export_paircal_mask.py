"""Export the measured Qwen TP2 site-mask diagnostic to existing lookup JSON.

The paired report is bound to one concrete four-layer M2048 mock workload;
the supplied lookup gives its explicit runtime and shape inventory. This is
not the operator-regression step model and cannot generalize across contexts.
"""
import argparse
import hashlib
import json
from pathlib import Path

from ..tp2_lookup import TRAIN_SCHEMA
from .step_mask_calibration import StepBlock, select


CONTEXT = 'qwen36-local4-random-mock-training-tp2-m2048-conn1-dx-sites'
MASKS = {'stock': set(), 'proj': {'proj'}, 'fc2': {'fc2'}, 'pair': {'proj', 'fc2'}}
RUNTIME = {'tp': 2, 'dtype': 'bfloat16', 'gpu': 'NVIDIA A800-SXM4-80GB', 'torch': '2.7.0+cu126',
           'triton': '3.4.0', 'cuda_build': '12.6', 'cuda_device_max_connections': '1'}
INVENTORY = {'qkv:2048:5120:7168', 'proj:2048:3072:5120', 'fc1:2048:5120:8240',
             'fc2:2048:3072:5120', 'fc1:2048:5120:17408', 'fc2:2048:8704:5120'}
WORKLOAD = {'num_layers': 4, 'hidden_size': 5120, 'ffn_hidden_size': 17408, 'num_attention_heads': 24,
            'num_query_groups': 4, 'kv_channels': 256, 'seq_length': 2048, 'micro_batch_size': 1,
            'global_batch_size': 2, 'tensor_model_parallel_size': 2, 'pipeline_model_parallel_size': 1,
            'experimental_attention_variant': 'gated_delta_net', 'linear_attention_freq': 4,
            'transformer_impl': 'local', 'recompute_granularity': None, 'recompute_method': None,
            'mock_data': True, 'seed': 1234, 'sequence_parallel': True, 'bf16': True, 'optimizer': 'adam',
            'use_distributed_optimizer': False, 'optimizer_cpu_offload': False,
            'stowaway_qwen36_local_spec': True}


def export(report, inventory, *, report_sha, inventory_sha):
    if report.get('context') != CONTEXT or inventory.get('schema') != TRAIN_SCHEMA:
        raise ValueError('unsupported workload or inventory schema')
    runtime = inventory['runtime']
    if runtime != RUNTIME:
        raise ValueError('runtime differs from the recorded diagnostic workload')
    measured = []
    for record in report['blocks']:
        deltas = record['median_delta_ms']
        if set(deltas) != set(MASKS) or deltas['stock'] != 0:
            raise ValueError('incomplete block mask inventory')
        # Constant reference cancels; requires positive StepBlock inputs.
        ref = 1000 + max(abs(float(x)) for x in deltas.values())
        measured.append(StepBlock(CONTEXT, str(record['block']), {k: ref + v for k, v in deltas.items()}))
    result = select(measured, context=CONTEXT, allowed_masks=MASKS, mad_penalty=1., min_blocks=3)
    chosen = MASKS[result['choice']]
    entries = {}
    for k in inventory['entries']:
        parts = k.split(':')
        if len(parts) != 4 or parts[0] not in {'qkv','proj','fc1','fc2'} or not all(x.isdigit() and int(x)>0 for x in parts[1:]):
            raise ValueError('invalid runtime inventory shape')
        if int(parts[1]) == 2048:
            entries[k] = {'choice': 'dx' if parts[0] in chosen else 'stock'}
    if set(entries) != INVENTORY:
        raise ValueError('M2048 shape inventory differs from the recorded diagnostic workload')
    return {'schema': TRAIN_SCHEMA, 'classification': 'diagnostic-forward-backward', 'runtime': runtime,
            'entries': entries, 'plan_source': 'experimental-measured-step-mask', 'workload': CONTEXT,
            'workload_requirements': WORKLOAD,
            'selected_mask': result['choice'], 'selection': result,
            'sources': {'paired_report_sha256': report_sha, 'shape_inventory_sha256': inventory_sha},
            'warning': 'exploratory 3-block selection; four-layer mock workload only; no E2E gain guarantee'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--paired-report',type=Path,required=True)
    ap.add_argument('--inventory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    report, inventory = a.paired_report.read_bytes(), a.inventory.read_bytes()
    payload=export(json.loads(report),json.loads(inventory),report_sha=hashlib.sha256(report).hexdigest(),
                   inventory_sha=hashlib.sha256(inventory).hexdigest())
    encoded=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(encoded)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(encoded).hexdigest(),
                      'selected_mask':payload['selected_mask'],'entries':len(payload['entries'])},indent=2))


if __name__=='__main__':
    main()
