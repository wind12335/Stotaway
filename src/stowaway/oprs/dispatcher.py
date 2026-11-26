"""Runtime dispatcher: load one plan, allocate its contexts in one canonical order, route every call.

Hygon port of nvidia/stowaway_nvidia/multi_backend8.py. Differences that the BW stack forces:
  * vocabulary: the no-overlap choice is `serial` (upstream `stock`), the own method is `deo`;
  * TP, dtype and platform come from the plan (upstream fixes 8 / A800 / BF16) and are compared with
    the running process -- a plan for another card count, topology or software stack is refused;
  * the symmetric heap is explicit: before the first context is allocated the plan's allocations are
    converted to heap bytes and checked against DUSHMEM_SYMMETRIC_SIZE (a NULL heap malloc inside a
    kernel constructor is otherwise an assertion on one rank and a hang on the others);
  * a training call site whose choice is `serial` returns None from exchange_for() and the framework
    adapter runs the framework's ORIGINAL forward and backward (nothing is re-implemented there).

Rules kept from the handoff: all TP ranks read the same plan (digest compared with a collective
before anything is allocated); DUSHMEM world, process groups and backend contexts are initialised
in one consistent order; the hit path keeps the input/output layout, the communication semantics
and the training backward; unmatched calls take the explicit default `serial` and are counted with
a reason; actual hits and fallback reasons are reported per path (OPRS_HIT / OPRS_FALLBACK lines).
"""
import time
from collections import Counter

from .auto_plan import AutoPlan
from .backends import METHODS, create_exchange, dtype_name
from .engine import configured_heap
from .platform import DEO, allocations_heap_bytes, runtime_platform
from .predictor import SERIAL


class Dispatcher:
    def __init__(self, path, workload, *, group=None, methods=None, deo_bwd='ag', engine=None, topology=None):
        """`methods` = overlap methods this process may run (default: every integrated one, `deo`
        included). A plan that needs another one is refused on every rank before anything is
        allocated."""
        import torch
        import torch.distributed as dist
        self.group = group if group is not None else dist.group.WORLD
        self.tp = self.group.size()
        try:
            self.grank = dist.get_rank(self.group)
        except (RuntimeError, ValueError):
            self.grank = dist.get_rank()
        self.workload, self.path = workload, str(path)
        self.deo_bwd = deo_bwd
        self.exchanges, self.deo_contexts = {}, None
        self.hits, self.fallbacks, self.runs = Counter(), Counter(), Counter()
        self.closed = False
        plan, error, t0 = None, None, time.perf_counter()
        try:
            plan = AutoPlan(path, workload, tp=self.tp,
                            runtime_platform=runtime_platform(tp=self.tp, topology=topology))
        except Exception as exc:                                  # noqa: BLE001 - reported on every rank
            error = f'{type(exc).__name__}: {exc}'
        # No rank may allocate while another one refused the plan: agree first, then raise everywhere
        # or nowhere (a rank that went ahead would hang in a context constructor's barrier).
        peers = [None] * self.tp
        dist.all_gather_object(peers, (plan.digest if plan else None, error), group=self.group)
        problems = [f'rank {r}: {e}' for r, (digest, e) in enumerate(peers) if e]
        if plan is not None and any(digest != plan.digest for digest, _ in peers):
            problems.append(f'ranks read different plan files: {sorted({d for d, _ in peers if d})}')
        if problems:
            raise RuntimeError('automatic plan not usable on all ranks -- ' + '; '.join(problems))
        self.plan = plan
        methods = tuple(METHODS if methods is None else methods)
        disabled = {method for method in METHODS if method not in methods}
        used = {choice for choice in plan.entries.values() if choice != SERIAL} | {a[0] for a in plan.allocate}
        if used & disabled:
            raise RuntimeError(f'plan needs {sorted(used & disabled)} but backends enabled here are '
                               f'{sorted(set(METHODS) - disabled)}')
        if DEO in used and engine is None:
            raise RuntimeError('deo needs an EngineWorld (symmetric-heap world)')
        self.dtype = getattr(torch, plan.dtype)
        # the Null arm allocates the contexts of the plan it derives from and replaces nothing:
        # allocations = non-serial entries of THIS plan + the copied allocation list
        needed = sorted({(choice, op, m, n, k) for (_, op, m, n, k), choice in plan.entries.items()
                         if choice != SERIAL} | set(plan.allocate))
        self.needed = needed
        need = allocations_heap_bytes(needed, self.tp, deo_bwd=deo_bwd == 'ag')
        have = configured_heap()
        self.heap_bytes = need
        if need > have:
            raise RuntimeError(f'plan allocations need {need / 2**30:.2f} GiB of symmetric heap but '
                               f'DUSHMEM_SYMMETRIC_SIZE is {have / 2**30:.2f} GiB')
        # contexts in one canonical order on every rank (the constructors are collective)
        self.load_ms = (time.perf_counter() - t0) * 1e3       # plan read + agreement, before any context
        error = None
        try:
            if any(method == DEO for method, *_ in needed):
                from .backends.deo import DeoContexts
                self.deo_contexts = DeoContexts(engine, bwd=deo_bwd)
            for method, op, m, n, k in needed:
                self.exchanges[(method, op, m, n, k)] = create_exchange(
                    method, engine, op, m, n, k, self.dtype, group=self.group,
                    deo_contexts=self.deo_contexts)
        except Exception as exc:                                  # noqa: BLE001 - reported on every rank
            error = f'{type(exc).__name__}: {exc}'
        peers = [None] * self.tp
        dist.all_gather_object(peers, error, group=self.group)
        failed = [f'rank {r}: {e}' for r, e in enumerate(peers) if e]
        if failed:
            raise RuntimeError('context allocation failed -- ' + '; '.join(failed))
        self.prepare_ms = (time.perf_counter() - t0) * 1e3 - self.load_ms   # context construction
        kinds = {}
        for (method, op, m, n, k), exchange in self.exchanges.items():
            kinds[f'{method}/{op}'] = kinds.get(f'{method}/{op}', 0) + 1
        self.log(f'OPRS_READY rank={self.grank} tp={self.tp} workload={workload} plan={plan.digest[:12]} '
                 f'backend={plan.backend} entries={len(plan.entries)} deo_bwd={deo_bwd} '
                 f'contexts={len(self.exchanges)} {dict(sorted(kinds.items()))} '
                 f'heap_need_mb={need / 2**20:.0f} heap_mb={have / 2**20:.0f} '
                 f'load_ms={self.load_ms:.1f} prepare_ms={self.prepare_ms:.1f}')

    @staticmethod
    def log(message):
        print(f'[oprs] {message}', flush=True)

    # ---- routing --------------------------------------------------------------
    def locate(self, site, op, x, weight):
        """(choice, reason, M, N, K) of one call. M = global rows: op 'ag' gets the [M/R, K] input
        shard, op 'rs' the [M, K/R] full-row activation."""
        m = x.shape[0] * self.tp if op == 'ag' else x.shape[0]
        n, k = weight.shape[0], x.shape[-1]
        choice, reason = self.plan.choose(site, op, m, n, k, dtype_name(x.dtype))
        return choice, reason, m, n, k

    def exchange_for(self, site, op, m, n, k, dtype):
        """Context for a training call site (framework adapters in autograd.py / megatron_hook.py):
        None = the plan says `serial` (or the shape / dtype is not in the plan) -> original path."""
        choice, reason = self.plan.choose(site, op, m, n, k, dtype_name(dtype))
        if choice == SERIAL:
            self.fallbacks[reason] += 1
            return None
        self.hits[(site, op, m, n, k, choice)] += 1
        return self.exchanges[(choice, op, m, n, k)]

    def peek(self, site, op, x, weight):
        """(choice, reason, M, N, K) without counting: lets a caller prepare method-specific inputs."""
        return self.locate(site, op, x, weight)

    def serial(self, op, x, weight, gemm=None):
        """The serial path with the plan's own definition: AG -> GEMM, or GEMM -> RS (RCCL collectives).
        `gemm` = the framework's own layer computation, if it must be kept."""
        import torch
        import torch.distributed as dist
        if gemm is None:
            gemm = lambda a: torch.matmul(a, weight.detach().t())    # noqa: E731
        if op == 'ag':
            full = torch.empty(x.shape[0] * self.tp, x.shape[1], dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(full, x.contiguous(), group=self.group)
            return gemm(full)
        part = gemm(x)
        out = torch.empty(x.shape[0] // self.tp, weight.shape[0], dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, part.contiguous(), group=self.group)
        return out

    def run(self, site, op, x, weight, gemm=None):
        """Forward-only execution of one call site (inference): plan choice or the serial path."""
        if x.requires_grad or weight.requires_grad:
            raise RuntimeError('Dispatcher.run is forward-only; training goes through autograd.py')
        choice, reason, m, n, k = self.locate(site, op, x, weight)
        self.runs[(site, op, m, n, k, choice, reason)] += 1
        if choice == SERIAL:
            return self.serial(op, x, weight, gemm)
        exchange = self.exchanges[(choice, op, m, n, k)]
        if op == 'ag':
            return exchange.all_gather_gemm(x.contiguous(), weight.detach())
        return exchange.gemm_reduce_scatter(x.contiguous(), weight.detach())

    # ---- reporting and teardown ---------------------------------------------
    def report(self):
        """Actual hits, fallback reasons and forward-only runs of this rank (one line each)."""
        lines = [f'OPRS_SUMMARY rank={self.grank} tp={self.tp} workload={self.workload} '
                 f'plan={self.plan.digest[:12]} backend={self.plan.backend} deo_bwd={self.deo_bwd} '
                 f'entries={len(self.plan.entries)} contexts={len(self.exchanges)}']
        for (site, op, m, n, k, choice), count in sorted(self.hits.items()):
            lines.append(f'OPRS_HIT rank={self.grank} site={site} op={op} M={m} N={n} K={k} '
                         f'choice={choice} count={count}')
        for reason, count in sorted(self.fallbacks.items()):
            lines.append(f'OPRS_FALLBACK rank={self.grank} reason={reason} count={count}')
        for (site, op, m, n, k, choice, reason), count in sorted(self.runs.items()):
            lines.append(f'OPRS_RUN rank={self.grank} site={site} op={op} M={m} N={n} K={k} '
                         f'choice={choice} reason={reason} count={count}')
        return lines

    def print_report(self):
        for line in self.report():
            self.log(line)

    def close(self):
        """Collective: release every context, reverse order of creation."""
        if self.closed:
            return
        self.closed = True
        import torch
        for key in sorted(self.exchanges, reverse=True):
            self.exchanges[key].close()
        if self.deo_contexts is not None:
            self.deo_contexts.close()
        torch.cuda.empty_cache()
