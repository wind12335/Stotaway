"""Megatron adapter of the OPRS runtime (training): Hygon port of the multi-backend branch of
nvidia/experiments/megatron_async_prelude.py.

What it does, in order:
  arm()        class-level patches of RowParallelLinear.forward and ColumnParallelLinear._forward_impl
               (the same two entry points the BW vehicle patches have always used), a constructor
               wrap that keeps the site name, and a wrap of megatron.training.training.train -- after
               the das adaptor has replaced that function by its own train(), which is the one pretrain
               runs (the BW Megatron is the das fork; see the handoff doc).
  train() entry   (all ranks in lockstep) the call sites are named, the engine world is brought up
               over the TP group (MPI_Comm_split + DUSHMEM heap, same bootstrap as the vehicle) and
               the dispatcher loads the plan and allocates its contexts in canonical order. Nothing
               is allocated while the model is built.
  per call     eligible call site -> Dispatcher.exchange_for(): None means the plan says `serial`
               (or the shape / dtype is not in the plan) -> the framework's ORIGINAL forward and
               backward; otherwise the fused forward of the chosen method with the backward of
               autograd.py. Ineligible calls (other sites, bias, grad-accum fusion, ...) always take
               the original.

Site naming, the one place where the BW stack differs from NVIDIA: upstream Megatron accepts
tp_comm_buffer_name but does not keep it, NVIDIA catches it in the constructor. The BW Megatron
(mcore 0.18) also passes it, so the constructor wrap is kept -- and every site is cross-checked at
train() entry against the module path in the model chunks (self_attention.linear_qkv / _proj,
mlp.linear_fc1 / _fc2), which is the ground truth of the local backend. A layer the two disagree on
is never dispatched. Expert linears of MoE layers are never dispatched (MoE is not in the operator
table). A model with zero named sites prints a loud warning.

Hit / fallback accounting per rank (printed after train() and at exit):
  OPRS_HIT / OPRS_FALLBACK / OPRS_RUN        dispatcher.py (plan digest, contexts, heap)
  OPRS_ADAPTER                               row / column hits, skips, calls before train()
Environment: OPRS_PLAN (required), OPRS_WORKLOAD, OPRS_METHODS (comma list of the methods this process may run,
                default all: deo async_tp triton_dist),
OPRS_DEO_BWD = ag (default: fused AG + dgrad backward at the row sites) | stock.

Layout law of the call sites: they are dispatched in their sequence-parallel (SP) layout -- column:
gathered rows -> GEMM -> per-rank shards; row: full rows -> GEMM -> scattered sum. At TP > 1 that
means the layer must be in SP form. At TP=1 Megatron turns SP off (arguments, layer and model config
all refuse it) and the plain layout IS that layout: every gather / scatter is the identity and the
tensors are the same, so the SP flags are not required there. A TP=1 plan is therefore a single-card
plumbing check of the hit path (one card has nothing to overlap), never a measurement arm.

Entry through e2e/megatron/prelude.py (OPRS_PLAN set): engine world, heap size and the old
MGD_*/OPUS_* switches are mutually exclusive with it. No part of the old control plane (BRS,
standalone RS) belongs to the OPRS arm.
"""
import atexit
import os
import sys
from collections import Counter

SITES = ('qkv', 'proj', 'fc1', 'fc2')
# module path inside a model chunk -> site; (attention, MLP) of a dense layer
BY_PATH = {'self_attention.linear_qkv': 'qkv', 'self_attention.linear_proj': 'proj',
           'mlp.linear_fc1': 'fc1', 'mlp.linear_fc2': 'fc2'}
BY_LEAF = {'linear_qkv': 'qkv', 'linear_proj': 'proj', 'linear_fc1': 'fc1', 'linear_fc2': 'fc2'}

_ST = {'armed': False, 'disp': None, 'rank': None, 'row': 0, 'column': 0, 'row_skip': 0,
       'column_skip': 0, 'early': 0, 'sites': {}, 'conflicts': 0, 'untagged': 0,
       'group_mismatch': 0, 'reported': False, 'snapshot': None, 'plan': None, 'workload': None,
       'methods': ()}

log = lambda message: print(f'[oprs] {message}', flush=True)      # noqa: E731


def site_of(path):
    """Site named by a module path (last two components), or None."""
    parts = path.split('.')
    return BY_PATH.get('.'.join(parts[-2:])) if len(parts) >= 2 else None


def _name_sites(model):
    """Give every Column/RowParallelLinear of the model its site (constructor name cross-checked
    with the module path). Returns {site: count}; expert linears are left untagged."""
    import torch
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    linear, per_site, seen = (ColumnParallelLinear, RowParallelLinear), Counter(), set()
    for chunk in chunks:
        if not isinstance(chunk, torch.nn.Module):
            continue
        for path, module in chunk.named_modules():
            if id(module) in seen or not isinstance(module, linear):
                continue
            seen.add(id(module))
            by_path, by_ctor = site_of(path), getattr(module, '_oprs_site_ctor', None)
            # expert linears are never dispatched (MoE is not in the operator table), by path no more
            # than by constructor name; and the constructor name alone is trusted only on the leaf it names
            if getattr(module, 'is_expert', False):
                by_path = by_ctor = None
            if by_ctor and BY_LEAF.get(path.rsplit('.', 1)[-1]) != by_ctor:
                by_ctor = None
            site = by_path or by_ctor
            if by_path and by_ctor and by_path != by_ctor:
                _ST['conflicts'] += 1
                site = None
            module._oprs_site = site
            if site:
                per_site[site] += 1
            else:
                _ST['untagged'] += 1
    return dict(sorted(per_site.items()))


def _enter(model):
    """At train() entry, on every rank in lockstep: name the sites, start the engine world over
    the TP group, build the dispatcher."""
    import torch.distributed as dist
    from megatron.core import parallel_state

    from .dispatcher import Dispatcher
    from .engine import EngineWorld

    if _ST['disp'] is not None:
        return
    _ST['rank'] = dist.get_rank()
    _ST['sites'] = _name_sites(model)
    total = sum(_ST['sites'].values())
    if total == 0:
        log(f'OPRS_ADAPTER WARNING rank={_ST["rank"]}: no call site named in the model -- every '
            f'call takes the original path (untagged={_ST["untagged"]})')
    group = parallel_state.get_tensor_model_parallel_group()
    engine = EngineWorld.over_group(group)
    disp = Dispatcher(_ST['plan'], _ST['workload'], group=group, methods=_ST['methods'],
                      deo_bwd=_ST['bwd'], engine=engine)
    _ST['disp'] = disp
    log(f'OPRS_ADAPTER rank={_ST["rank"]} tp={disp.tp} sites={_ST["sites"]} conflicts={_ST["conflicts"]} '
        f'untagged={_ST["untagged"]} deo_bwd={_ST["bwd"]} methods={sorted(_ST["methods"])}')


def _same_group(layer):
    """layer.tp_group is the dispatcher's group (checked once per layer)."""
    import torch.distributed as dist
    disp = _ST['disp']
    if getattr(layer, '_oprs_group_for', None) != id(disp):
        try:
            ok = dist.get_process_group_ranks(layer.tp_group) == dist.get_process_group_ranks(disp.group)
        except (RuntimeError, ValueError, AttributeError):                 # noqa: BLE001
            ok = False
        layer._oprs_group_for, layer._oprs_group_ok = id(disp), ok
        if not ok:                                  # counted once per layer
            _ST['group_mismatch'] += 1
    return layer._oprs_group_ok


def report():
    disp = _ST['disp']
    if _ST['reported']:
        return
    _ST['reported'] = True
    if disp is not None:
        disp.print_report()
    _ST['snapshot'] = _snapshot()
    log(f'OPRS_ADAPTER rank={_ST["rank"]} row={_ST["row"]} column={_ST["column"]} '
        f'row_skip={_ST["row_skip"]} column_skip={_ST["column_skip"]} early={_ST["early"]} '
        f'sites={_ST["sites"]} conflicts={_ST["conflicts"]} untagged={_ST["untagged"]} '
        f'group_mismatch={_ST["group_mismatch"]}')


def _snapshot():
    disp = _ST['disp']
    return ({k: v for k, v in _ST.items() if k in ('row', 'column', 'row_skip', 'column_skip', 'early')},
            dict(disp.hits), dict(disp.fallbacks)) if disp is not None else None


def _report_delta():
    """What ran after report(): the framework evaluates after train() returned (its validation and
    test passes), and those calls go through the same hook. Printed in the same line formats with the
    increments, so that a sum over the log is the count of the whole process."""
    now = _snapshot()
    before = _ST.get('snapshot')
    if now is None or before is None or now == before:
        return
    for key, value in now[0].items():
        if value != before[0][key]:
            _ST[key] = value - before[0][key]           # the adapter line prints the increments
    log(f'OPRS_ADAPTER rank={_ST["rank"]} after_report+ ' + ' '.join(
        f'{k}={_ST[k]}' for k in ('row', 'column', 'row_skip', 'column_skip', 'early')) + ' '
        f'sites={_ST["sites"]} conflicts={_ST["conflicts"]} untagged={_ST["untagged"]} '
        f'group_mismatch={_ST["group_mismatch"]}')
    for key, value in now[1].items():
        if value != before[1].get(key):
            site, op, m, n, k, choice = key
            log(f'OPRS_HIT rank={_ST["rank"]} site={site} op={op} M={m} N={n} K={k} choice={choice} '
                f'count={value - before[1].get(key, 0)} after_report')
    for reason, value in now[2].items():
        if value != before[2].get(reason):
            log(f'OPRS_FALLBACK rank={_ST["rank"]} reason={reason} count={value - before[2].get(reason, 0)} '
                f'after_report')
    for key, value in now[0].items():                  # the counters themselves keep their true totals
        _ST[key] = value
    _ST['snapshot'] = now


@atexit.register
def _exit():
    """Teardown-order law of this stack: once the engine's MPI world is up, letting the interpreter
    reach the NCCL destructors segfaults (139). Print, flush, leave with the training exit code.

    mpirun then reports an abnormal termination (the rank leaves without MPI_Finalize) and returns 1
    whatever this code is: the arm's exit code is the `EXIT <n>` its rank printed."""
    if _ST['disp'] is None:
        return                                      # nothing was started: normal interpreter exit
    try:
        report()
        _report_delta()
        if os.environ.get('OPRS_CLOSE_AT_EXIT', '') == '1':
            _ST['disp'].close()
    except Exception as exc:                                       # noqa: BLE001
        log(f'OPRS_ADAPTER exit: {type(exc).__name__}: {exc}')
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(getattr(sys, '_oprs_exit_code', 0) or 0)


def arm(plan_path, workload=None, *, methods=None, deo_bwd='ag'):
    """Install the patches and the train() hook (once per process). `methods` = overlap methods
    this process may run (default all); a plan that needs another one is refused on every rank."""
    import torch
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    import megatron.training.training as mt

    from .autograd import column_parallel, row_parallel     # closures below keep these
    from .backends import METHODS

    if _ST['armed']:
        raise RuntimeError('the OPRS Megatron adapter is already armed')
    enabled = tuple(METHODS if methods is None else methods)
    unknown = sorted(set(enabled) - set(METHODS))
    if unknown:
        raise ValueError(f'unknown methods {unknown}; choose from {METHODS}')
    if deo_bwd not in ('ag', 'stock'):
        raise ValueError(f'deo_bwd must be ag or stock, got {deo_bwd!r}')
    _ST.update(armed=True, plan=str(plan_path), workload=workload or '', methods=enabled, bwd=deo_bwd)

    original_row = RowParallelLinear.forward
    original_column = ColumnParallelLinear._forward_impl
    original_init = {cls: cls.__init__ for cls in (ColumnParallelLinear, RowParallelLinear)}
    for cls, init in original_init.items():
        def constructor(layer, *args, _init=init, **kwargs):
            _init(layer, *args, **kwargs)
            layer._oprs_site_ctor = kwargs.get('tp_comm_buffer_name')
        cls.__init__ = constructor

    def row_forward(layer, input_):
        disp = _ST['disp']
        if disp is None:
            _ST['early'] += 1
            return original_row(layer, input_)
        eligible = (getattr(layer, '_oprs_site', None) in SITES and (layer.sequence_parallel or disp.tp == 1)
                    and layer.input_is_parallel and not layer.explicit_expert_comm
                    and not layer.gradient_accumulation_fusion and layer.bias is None
                    and input_.ndim == 3 and input_.dtype in (torch.float16, torch.bfloat16)
                    and input_.shape[0] % disp.tp == 0 and input_.is_contiguous()
                    and layer.weight.requires_grad)
        if not eligible or not _same_group(layer):
            _ST['row_skip'] += 1
            return original_row(layer, input_)
        rows = input_.shape[0] * input_.shape[1]          # S (full) x B
        exchange = disp.exchange_for(layer._oprs_site, 'rs', rows, layer.weight.shape[0],
                                     input_.shape[2], input_.dtype)
        if exchange is None:
            _ST['row_skip'] += 1
            return original_row(layer, input_)
        output = row_parallel(input_.view(rows, -1), layer.weight, exchange, layer.tp_group)
        _ST['row'] += 1
        # SP contract: the RS row block of the flat [S*B, N] is the sequence chunk [S/R, B, N]
        return (output.view(input_.shape[0] // disp.tp, input_.shape[1], -1),
                layer.bias if layer.skip_bias_add else None)

    def column_forward(layer, input, weight, *args, **kwargs):
        disp = _ST['disp']
        if disp is None:
            _ST['early'] += 1
            return original_column(layer, input, weight, *args, **kwargs)
        eligible = (not args and getattr(layer, '_oprs_site', None) in SITES
                    and (kwargs.get('sequence_parallel') or disp.tp == 1) and kwargs.get('bias') is None
                    and not kwargs.get('gradient_accumulation_fusion')
                    and not kwargs.get('allreduce_dgrad') and kwargs.get('grad_output_buffer') is None
                    and input.ndim == 3 and input.is_contiguous() and weight.requires_grad
                    and input.dtype in (torch.float16, torch.bfloat16))
        if not eligible or not _same_group(layer):
            _ST['column_skip'] += 1
            return original_column(layer, input, weight, *args, **kwargs)
        group = kwargs.get('tp_group')
        rows = input.shape[0] * input.shape[1] * disp.tp     # (S/R) x B x R
        exchange = disp.exchange_for(layer._oprs_site, 'ag', rows, weight.shape[0], input.shape[2],
                                     input.dtype)
        if exchange is None:
            _ST['column_skip'] += 1
            return original_column(layer, input, weight, *args, **kwargs)
        output = column_parallel(input.view(-1, input.shape[2]), weight, exchange, group or disp.group)
        _ST['column'] += 1
        return output.view(input.shape[0] * disp.tp, input.shape[1], -1)

    def train(*args, **kwargs):
        model = args[1] if len(args) > 1 else kwargs.get('model')
        try:
            _enter(model)
            result = mt_train(*args, **kwargs)
        except SystemExit as exc:                 # the framework's own exit: keep its code
            sys._oprs_exit_code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
            raise
        except BaseException:
            sys._oprs_exit_code = 1                  # propagated through the atexit os._exit
            raise
        try:
            report()                                # R=8: rank 0's teardown can crash before atexit
        except Exception as exc:                                       # noqa: BLE001
            log(f'OPRS_ADAPTER report failed: {type(exc).__name__}: {exc}')
            sys._oprs_exit_code = 1
            raise
        sys._oprs_exit_code = 0
        return result

    mt_train = mt.train
    RowParallelLinear.forward = row_forward
    ColumnParallelLinear._forward_impl = column_forward
    mt.train = train
    log(f'OPRS_ADAPTER armed plan={plan_path} workload={_ST["workload"] or "<none>"} '
        f'methods={sorted(enabled)} deo_bwd={deo_bwd}')
