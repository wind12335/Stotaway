"""MGD t21 RS attribution: count torch.distributed reduce_scatter calls by
CALLER SITE + bytes, dump at exit.

Why: the STOCK step trace shows 411 ReduceScatter_b16 kernels / 3.1s per
step (median 4.1ms, p10 0.03ms) and the fix pool depends on WHERE they
fire: layers.py:349 SP-dgrad RS (512MB class, wire-bound) vs per-param
wgrad smalls (batchable) vs vocab/param_and_grad_buffer strays.  Kernel
names carry no payload or site info, so count at the API boundary.

Install: mgd_prelude2 imports this BEFORE runpy(pretrain_gpt.py), so
megatron's module-level aliases
    dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
(3rdparty param_and_grad_buffer.py:40 / mappings.py:13) capture the
wrapper at THEIR import time; live attribute lookups hit it too.
MGD_RS_ATTR=1 gate.  Teardown: atexit (vehicle exits clean).
"""
import atexit
import os
import sys
from collections import defaultdict

import torch
import torch.distributed as _td

_THIS = os.path.abspath(__file__)
_C = defaultdict(lambda: [0, 0, 0])     # "file:func" -> [calls, numel, max]
_HIST = defaultdict(lambda: defaultdict(int))  # site -> log2(numel) -> n


def _caller():
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if fn != _THIS and "torch/distributed" not in fn:
            return f"{os.path.basename(fn)}:{f.f_code.co_name}"
        f = f.f_back
    return "?"


def _wrap(orig, name):
    def w(output, input, *a, **kw):
        try:
            n = input.numel()
        except Exception:                          # noqa: BLE001
            n = 0
        site = _caller()
        c = _C[site]
        c[0] += 1
        c[1] += n
        c[2] = max(c[2], n)
        _HIST[site][max(0, n.bit_length() - 1)] += 1
        return orig(output, input, *a, **kw)
    w.__name__ = f"mgd_attr_{name}"
    return w


for _n in ("reduce_scatter_tensor", "_reduce_scatter_base"):
    _o = getattr(_td, _n, None)
    if _o is not None:
        setattr(_td, _n, _wrap(_o, _n))


_MG_MODULES = (
    "megatron.core.distributed.param_and_grad_buffer",
    "megatron.core.tensor_parallel.layers",
    "megatron.core.tensor_parallel.mappings",
    "megatron.core.tensor_parallel.vocab_output",
)


def install_megatron_bindings():
    """Rebind the module-attribute aliases AFTER megatron imported them
    (the prelude imports _gls -> layers/mappings before this can run;
    module-level `from X import f` copies froze the original function).
    Late importers (vocab_output) copy from the rebound source module."""
    import sys as _sys
    for mn in _MG_MODULES:
        m = _sys.modules.get(mn)
        if m is None:
            continue
        f = getattr(m, "dist_reduce_scatter_func", None)
        if f is None or getattr(f, "__name__", "").startswith("mgd_attr"):
            continue
        setattr(m, "dist_reduce_scatter_func", _wrap(f, f"rs_{mn.rsplit('.',1)[-1]}"))
        g = getattr(m, "dist_all_gather_func", None)
        if g is not None and not getattr(g, "__name__", "").startswith("mgd_attr"):
            setattr(m, "dist_all_gather_func", _wrap(g, f"ag_{mn.rsplit('.',1)[-1]}"))


@atexit.register
def _dump():
    if os.environ.get("OMPI_COMM_WORLD_RANK", "0") not in ("0",):
        return
    if not _C:
        print("[mgd-rs-attr] NO CALLS CAPTURED (install-order bug?)",
              flush=True)
        return
    print("\n[mgd-rs-attr] site | calls | total_MB | avg_MB | max_MB | "
          "log2B-histogram(numel)", flush=True)
    for site, (n, tot, mx) in sorted(_C.items(), key=lambda kv: -kv[1][1]):
        h = _HIST[site]
        span = f"{min(h)}-{max(h)}" if h else "-"
        print(f"[mgd-rs-attr] {site:48s} {n:5d} {tot*2/1e6:9.1f} "
              f"{tot*2/n/1e6:7.2f} {mx*2/1e6:7.1f} exp2[{span}] "
              f"n={sum(h.values())}", flush=True)
