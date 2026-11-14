"""Single-host TP8 Llama real-weight prefill and natural-text correctness gate.

Per-process JSON records, max across ranks wall clock, and explicit adapter hits.
SP/DX arms remain diagnostic until token/logprob comparisons pass.
"""
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import get_world_group

rank = int(os.environ['RANK'])
if int(os.environ['WORLD_SIZE']) != 8:
    raise ValueError('TP8 required')
torch.cuda.set_device(rank)
m = int(os.getenv('V8_M', '16384'))
arm = os.getenv('V8_ARM', 'stock')
rope_factor = float(os.getenv('V8_ROPE_FACTOR', '0'))
model = Path(os.environ['V8_MODEL'])
if arm not in ('stock', 'stock_sp', 'row', 'all', 'plan', 'async_tp', 'async_dx', 'auto_plan', 'multi_plan'):
    raise ValueError('unsupported arm')
adapter = None
if arm != 'stock':
    from stowaway_nvidia.vllm_sp import LlamaSequenceParallel
    if arm in ('async_tp', 'async_dx', 'auto_plan'):
        from stowaway_nvidia.vllm_async_tp import AsyncTPLlamaSequenceParallel
        LlamaSequenceParallel = AsyncTPLlamaSequenceParallel
        os.environ['VS_ASYNC_DX_SITES'] = 'proj,fc2' if arm == 'async_dx' else ''
        if arm == 'auto_plan':
            os.environ['VS_AUTO_PLAN8'] = os.environ['V8_AUTO_PLAN']
    if arm == 'multi_plan':
        from stowaway_nvidia.vllm_multi_tp8 import MultiBackendLlamaSequenceParallel
        LlamaSequenceParallel = MultiBackendLlamaSequenceParallel
        os.environ['VS_MULTI_PLAN8'] = os.environ['V8_AUTO_PLAN']
    selected = None
    if arm == 'plan':
        report_path = Path(os.environ['V8_PLAN_JSON'])
        report = json.loads(report_path.read_text())
        if report.get('platform') != 'A800-TP8' or report.get('M') != m or report.get('model') != '70b':
            raise ValueError('incompatible predictor plan')
        selected = [site for site, choice in report['choices'].items() if choice == 'dx']
        if any(choice not in ('stock', 'dx') for choice in report['choices'].values()):
            raise ValueError('invalid predictor choice')
        print(f'V8_PREDICTOR_PLAN rank={rank} sites={selected} source={report_path}',flush=True)
    os.environ['VS_DX_COMPONENTS'] = 'row,col' if arm in ('all','plan') else 'row'
    os.environ['VS_ROW_EXACT'] = '0'
    adapter = LlamaSequenceParallel(None, rank, 'stock_sp' if arm == 'stock_sp' else 'dx', [m], world=8, selected_sites=selected)
    adapter.install()
overrides = {'rope_scaling': {'rope_type': 'linear', 'factor': rope_factor}} if rope_factor else {}
llm = LLM(model=str(model), tensor_parallel_size=8, pipeline_parallel_size=1,
          distributed_executor_backend='external_launcher', seed=0, dtype='bfloat16',
          max_model_len=2*m+64, gpu_memory_utilization=0.8, enforce_eager=True,
          enable_chunked_prefill=True, max_num_batched_tokens=m, disable_log_stats=True,
          enable_prefix_caching=False, hf_overrides=overrides)
cpu = get_world_group().cpu_group
tok = llm.get_tokenizer()
text = '\n\n'.join((model / name).read_text(errors='ignore') for name in
                    ('README.md', 'LICENSE', 'USE_POLICY.md') if (model / name).exists())
base = tok.encode(text, add_special_tokens=False)
if len(base) <= 2000:
    raise RuntimeError('natural text is too short')
copies = m // len(base) + 2
stream = base * copies
def natural(i):
    end = (copies - 1) * len(base) + (i + 1) * len(base) // 5
    return TokensPrompt(prompt_token_ids=[tok.bos_token_id] + stream[end-m+1:end])

def random_prompts(nreq, length, seed):
    prompts = []
    for r in range(nreq):
        g = torch.Generator().manual_seed(seed * 100003 + r)
        prompts.append(TokensPrompt(prompt_token_ids=torch.randint(1000, tok.vocab_size-1, (length,), generator=g).tolist()))
    return prompts

def generate(ps, n, logprobs=None):
    dist.barrier(group=cpu)
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=n, ignore_eos=True,
                                               logprobs=logprobs), use_tqdm=False)
    torch.cuda.synchronize()
    times = [None] * 8
    dist.all_gather_object(times, (time.perf_counter() - start)*1000, group=cpu)
    return max(times), outputs

generate(random_prompts(2, 256, 999), 4)
generate(random_prompts(1, m, 998), 1)
results = []
reps = int(os.getenv('V8_REPS', '3'))
for cell, nreq, length, repeats in (('ttft', 1, m, reps), ('ttft2', 1, 2*m, reps),
                                     ('thru', 16, m, max(1, reps-1))):
    for rep in range(repeats):
        ms, _ = generate(random_prompts(nreq, length, 100+rep), 1)
        results.append(dict(cell=cell, rep=rep, nreq=nreq, prompt_tokens=length, ms=ms))
        if rank == 0:
            print('V8_BWMATCH', results[-1], flush=True)
gates = []
for i in range(4):
    prompt = natural(i)
    _, outputs = generate([prompt], 48, 5)
    out = outputs[0].outputs[0]
    gates.append(dict(ids=list(out.token_ids),
                      top5=[{str(k): float(v.logprob) for k, v in lp.items()} for lp in out.logprobs],
                      prompt_sha256=hashlib.sha256(json.dumps(prompt['prompt_token_ids'], separators=(',', ':')).encode()).hexdigest()))
if adapter:
    print(f'V8_ADAPTER rank={rank} arm={arm} hits={adapter.hits} fallbacks={adapter.fallbacks} backend_hits={getattr(adapter, "backend_hits", None)}', flush=True)
if rank == 0:
    summary = {cell: statistics.median(r['ms'] for r in results if r['cell'] == cell)
               for cell in ('ttft', 'ttft2', 'thru')}
    dest = Path(os.environ['V8_OUT'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as f:
        json.dump(dict(model=str(model), arm=arm, tp=8, M=m, rope_factor=rope_factor, summary_ms=summary,
                       plan_source=os.getenv('V8_PLAN_JSON') if arm == 'plan' else None,
                       selected_sites=selected if arm == 'plan' else None,
                       auto_plan_source=os.getenv('V8_AUTO_PLAN') if arm == 'auto_plan' else None,
                       auto_hits=getattr(adapter, 'auto_hits', None) if adapter else None,
                       multi_hits={str(k): v for k, v in adapter.multi.hits.items()} if arm == 'multi_plan' and adapter.multi else None,
                       backend_hits=getattr(adapter, 'backend_hits', None) if adapter else None,
                       results=results, gates=gates), f, indent=2)
    print('V8_BWMATCH_RESULT', summary, flush=True)
dist.barrier(group=cpu)
if adapter:
    adapter.close()
    if adapter.cuda is not None:
        import nvshmem.core
        nvshmem.core.finalize()
