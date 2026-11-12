"""Eight-rank real-weights offline vLLM baseline."""
import hashlib,json,os,statistics,time
from pathlib import Path
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0')
os.environ.setdefault('VLLM_LOGGING_LEVEL','WARNING')
import torch
import torch.distributed as dist
from vllm import LLM,SamplingParams
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import get_world_group
rank=int(os.environ['RANK'])
if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('TP8 required')
torch.cuda.set_device(rank)
model=Path(os.environ['V8_MODEL']);m=int(os.getenv('V8_M','1024'));gen=int(os.getenv('V8_GEN','32'))
arm=os.getenv('V8_ARM','stock')
adapter=None
if arm not in ('stock','stock_sp','dx'): raise ValueError('unsupported V8_ARM')
if arm != 'stock':
 from stowaway_nvidia.vllm_sp import LlamaSequenceParallel
 adapter=LlamaSequenceParallel(None,rank,arm,[m],architecture=os.getenv('V8_ARCH','llama'),world=8)
 adapter.install()
llm=LLM(model=str(model),tensor_parallel_size=8,pipeline_parallel_size=1,distributed_executor_backend='external_launcher',
        seed=0,dtype='bfloat16',max_model_len=m+gen+64,gpu_memory_utilization=.8,enforce_eager=True,
        enable_chunked_prefill=True,max_num_batched_tokens=m,disable_log_stats=True,enable_prefix_caching=False)
tokenizer=llm.get_tokenizer();text='\n'.join((model/n).read_text(errors='ignore') for n in ('README.md','LICENSE','USE_POLICY.md') if (model/n).exists())
tokens=tokenizer.encode(text,add_special_tokens=False)
if len(tokens)<2000:raise RuntimeError('source too short')
reps=m//len(tokens)+2;stream=tokens*reps
def prompt(i):
 end=(reps-1)*len(tokens)+(i+1)*len(tokens)//5
 prefix=[] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
 return TokensPrompt(prompt_token_ids=prefix+stream[end-m+len(prefix):end])
params={v:SamplingParams(temperature=0,max_tokens=v,ignore_eos=True) for v in (1,gen)}
for _ in range(2):llm.generate([prompt(0)],params[gen],use_tqdm=False)
cpu=get_world_group().cpu_group;results=[]
hits_before=adapter.hits if adapter else 0
for i in range(4):
 for n in (1,gen):
  p=prompt(i);dist.barrier(group=cpu);torch.cuda.synchronize();t=time.perf_counter()
  out=llm.generate([p],params[n],use_tqdm=False)[0].outputs[0];torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
  times=[None]*8;dist.all_gather_object(times,elapsed,group=cpu)
  results.append({'index':i,'n':n,'ms':max(times),'ids':list(out.token_ids),'prompt_sha256':hashlib.sha256(json.dumps(p['prompt_token_ids'],separators=(',',':')).encode()).hexdigest()})
if adapter:
 hits=adapter.hits-hits_before
 print(f'V8_ADAPTER rank={rank} arm={arm} measured_hits={hits} fallback={adapter.fallbacks}',flush=True)
 if hits < 8: raise RuntimeError('not all measured prefills hit adapter')
if rank==0:
 summary={str(n):statistics.median(x['ms'] for x in results if x['n']==n) for n in (1,gen)}
 dest=Path(os.environ['V8_OUT']);dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('x') as f:json.dump({'model':str(model),'arm':arm,'tp':8,'M':m,'output':gen,'summary_ms':summary,'results':results},f,indent=2)
 print('V8_RESULT',summary,flush=True)
dist.barrier(group=cpu)
if adapter:
 adapter.close()
 if adapter.cuda is not None:
  import nvshmem.core
  nvshmem.core.finalize()
