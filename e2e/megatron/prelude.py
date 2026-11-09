"""T21 prelude -- mgd_prelude.py (the proven C1 vehicle prelude, kept
untouched) extended for the 4-group A/B: granular engine arms.
  BASE : no MGD_*            -> fully stock
  RS   : MGD_FUSED=1         -> mgd_tp_patch (RowParallel RS engine)
  AG   : MGD_AG=1            -> mgd_ag_patch (ColumnParallel AG engine)
  AGRS : both                -> shared engine world
Same 3.12 shims as mgd_prelude (typing.override, WrappedTorchNorm SP
bypass, hyper_connection strip); engine imports LAST after every shim.
"""
import os
import runpy
import sys
import typing

# MGD_FH_SPY=1: no-ptrace host-stack sampler (py-spy is dead here -- container
# drops CAP_SYS_PTRACE, yama scope=1).  kill -USR1 <pid> appends all-thread
# python stacks to /tmp/r0_fh_<rank>.log; the shell sampler fires ~1/1.2s.
if os.environ.get("MGD_FH_SPY", "") == "1":
    import faulthandler
    import signal
    _rk = "x"
    for _i, _a in enumerate(sys.argv):
        if _a == "--rank" and _i + 1 < len(sys.argv):
            _rk = sys.argv[_i + 1]
    _fhf = open("/tmp/r0_fh_%s.log" % _rk, "w", buffering=1)
    faulthandler.register(signal.SIGUSR1, file=_fhf, all_threads=True)
    print("[fh-spy] rank=%s armed" % _rk, flush=True)

try:
    typing.override  # 3.12+
except AttributeError:
    import typing_extensions
    typing.override = typing_extensions.override

# MGD_*_TIMING=2 = stock-baseline timing arm: patches install with
# fusion disabled to time the stock paths on the same vehicle op set.
_ENGINE = (os.environ.get("MGD_FUSED", "") == "1"
           or os.environ.get("MGD_DRY", "") == "1"
           or os.environ.get("MGD_AG", "") == "1"
           or os.environ.get("MGD_AG_DRY", "") == "1"
           or os.environ.get("MGD_TP_TIMING", "") == "2"
           or os.environ.get("MGD_AG_TIMING", "") == "2"
           or os.environ.get("MGD_BRS", "") == "1")

# torch-MPI interposition order law (see mgd_prelude.py): preload the real
# MPI RTLD_GLOBAL before anything imports torch.
if _ENGINE:
    import ctypes
    import pydushmem.host as _pHost
    ctypes.CDLL(_pHost.MPI_LIB, mode=ctypes.RTLD_GLOBAL)

# das-fork compat #2: WrappedTorchNorm SP assert bypass (see mgd_prelude.py)
import megatron.core.transformer.torch_norm as _tn

_orig_norm_new = _tn.WrappedTorchNorm.__new__


def _norm_new_sp_ok(cls, config, *args, **kwargs):
    sp = config.sequence_parallel
    config.sequence_parallel = False
    try:
        return _orig_norm_new(cls, config, *args, **kwargs)
    finally:
        config.sequence_parallel = sp


_tn.WrappedTorchNorm.__new__ = _norm_new_sp_ok

# das-fork compat: strip enable_hyper_connection from the local spec
if os.environ.get("MGD_RS_ATTR", "") == "1":
    import stowaway.integrations.mgd_rs_attr as _attr  # noqa: F401

import megatron.core.models.gpt.gpt_layer_specs as _gls
if os.environ.get("MGD_RS_ATTR", "") == "1":
    _attr.install_megatron_bindings()   # layers/mappings already loaded

_orig_local_spec = _gls.get_gpt_layer_local_spec


def _local_spec_no_hc(*args, **kwargs):
    kwargs.pop("enable_hyper_connection", None)
    return _orig_local_spec(*args, **kwargs)


_gls.get_gpt_layer_local_spec = _local_spec_no_hc

# Engine TP-comm patches LAST (import order law: tp before ag so tp's
# train-hook installs inner and its atexit teardown runs last; see
# mgd_ag_patch.install).
if _ENGINE:
    import hcu_megatron.megatron_adaptor  # noqa: F401  (idempotent later)
    if os.environ.get("MGD_FUSED", "") == "1" \
            or os.environ.get("MGD_DRY", "") == "1" \
            or os.environ.get("MGD_TP_TIMING", "") == "2":
        import stowaway.integrations.mgd_tp_patch  # noqa: F401
    if os.environ.get("MGD_AG", "") == "1" \
            or os.environ.get("MGD_AG_DRY", "") == "1" \
            or os.environ.get("MGD_AG_TIMING", "") == "2":
        import stowaway.integrations.mgd_ag_patch  # noqa: F401
    if os.environ.get("MGD_BRS", "") == "1":
        import stowaway.integrations.mgd_brs_patch  # noqa: F401

# MGD_MXOFF=1 (t21 R12l): megatron validate_args asserts
# CUDA_DEVICE_MAX_CONNECTIONS=="1" (arguments.py:1422) but the runtime only
# needs it gone when HW queues are created (post-parse).  Keep =1 through
# the assert, then drop both queue limiters so stock async windows
# (layers.py fire->wgrad->wait) and overlap-grad-reduce get real
# multi-queue overlap instead of the single in-order queue.
if os.environ.get("MGD_MXOFF", "") == "1":
    import torch
    import megatron.training.arguments as _margs
    _orig_pva = _margs.parse_and_validate_args

    def _pva_then_flip(*a, **kw):
        r = _orig_pva(*a, **kw)
        os.environ.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
        os.environ.pop("GPU_MAX_HW_QUEUES", None)
        print("[mx-flip] queue limiters dropped post-parse "
              "(cuda_initialized=%s)" % torch.cuda.is_initialized(),
              flush=True)
        return r

    _margs.parse_and_validate_args = _pva_then_flip

if len(sys.argv) < 2:
    raise SystemExit("usage: mgd_prelude2.py <pretrain_gpt.py path> [args]")
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
