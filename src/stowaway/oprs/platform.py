"""BW1000 platform identity, BW execution routes and static shape support (CPU only).

A Route must describe the execution path that was actually measured, on the device / topology /
TP / dtype / software / timing regime it was measured under. Models are never transferred
between routes: A800 routes are different objects from BW routes, and the legacy operator table
(planner/data/cells.json, old timing protocol, warm-up and knobs unrecorded) is a different route
from the new-protocol measurements even on the same machine.
"""
import os
import re
from typing import Optional, Tuple

from .predictor import OPS, Route
from .registry import PATHS

VENDOR = 'hygon'
DEVICE = 'BW1000'              # platform label carried by every BW route and plan
COMM = 'rccl'
SYMMETRIC = 'dushmem'
DEO = 'deo'                    # Hygon's own overlap method (NVIDIA name: dx)
ASYNC_TP = 'async_tp'          # PyTorch's own fused symm-mem ops (torch.distributed._symmetric_memory)
TRITON_DIST = 'triton_dist'    # the DCU port of Triton-distributed (OPRS_TRITON_DIST_ROOT)
BACKENDS = (ASYNC_TP, TRITON_DIST)        # external overlap backends B on this stack
METHODS = (DEO,) + BACKENDS
DTYPES = ('bfloat16', 'float16')

TIMING = 'b2b16'               # new protocol: group = 16 back-to-back calls, group time / 16, 8 groups per arm
WARMUP = 4

LEGACY_TIMING = 'b2b16_perop_events'   # bench/{ag_gemm,gemm_rs}.py tm(): per-op events inside bursts of 16
LEGACY_CONFIG = 'legacy_env'           # OPUS_* knobs of the historical benches: their then-defaults, unrecorded

PLATFORM_KEYS = ('vendor', 'device', 'arch', 'topology', 'versions', 'comm', 'symmetric')
CONFIG_FAMILY = {'deo': 'deo_default', 'async_tp': 'pytorch_direct', 'triton_dist': 'default_no_autotune'}


def dtk_version() -> str:
    root = os.environ.get('DTKROOT') or os.environ.get('DTK_HOME') or '/opt/dtk'
    try:
        name = os.path.basename(os.path.realpath(root))
    except OSError:
        name = ''
    match = re.match(r'dtk-?(.+)$', name)
    return (match.group(1) if match else name) or 'unknown'


def detect_versions() -> Tuple[Tuple[str, str], ...]:
    """(dtk, hip, torch, triton) of the running interpreter; sorted unique pairs as Route requires."""
    import torch
    import triton
    pairs = [('dtk', dtk_version()), ('hip', str(torch.version.hip)), ('torch', torch.__version__),
             ('triton', triton.__version__)]
    return tuple(sorted({(str(k), str(v)) for k, v in pairs}))


ACTUAL_PATH = {DEO: 'direct_exchange', 'async_tp': 'torch_ops_symm_mem', 'triton_dist': 'triton_dist_dcu'}


def _route(method: str, op: str, *, tp, topology, versions, dtype, timing_mode, warmup, family) -> Route:
    cap = PATHS.get((method, ACTUAL_PATH.get(method, ''), op))
    if cap is None:
        raise ValueError(f'no BW route for ({method}, {op})')
    return Route(method, cap.compute, cap.transport, cap.memory, cap.synchronization, op,
                 cap.actual_path, DEVICE, topology, tp, dtype, versions, timing_mode, warmup, family)


def bw_route(method: str, op: str, *, tp: int, topology: str, versions: Tuple[Tuple[str, str], ...],
             dtype: str = 'bfloat16', timing_mode: str = TIMING, warmup: Optional[int] = WARMUP) -> Route:
    """The execution route of `method` for direction `op` on BW1000 under the new protocol."""
    if method not in METHODS:
        raise ValueError(f'unknown method {method}')
    return _route(method, op, tp=tp, topology=topology, versions=versions, dtype=dtype,
                  timing_mode=timing_mode, warmup=warmup, family=CONFIG_FAMILY[method])


def legacy_deo_route(op: str, *, tp: int, topology: str, versions: Tuple[Tuple[str, str], ...],
                     dtype: str = 'bfloat16') -> Route:
    """DEO route of the historical operator table (planner/data/cells.json): forward-only fused ops
    dx_ag_gemm_op / dx_gemm_rs_op, legacy timing, warm-up unknown."""
    return _route(DEO, op, tp=tp, topology=topology, versions=versions, dtype=dtype,
                  timing_mode=LEGACY_TIMING, warmup=None, family=LEGACY_CONFIG)


def static_support(method: str, op: str, m: int, n: int, k: int, tp: int) -> Tuple[str, str]:
    """Shape-only capability of one BW route.

    supported   inside the kernel's hard constraints AND its validated envelope
    unknown     hard constraints hold but the shape is outside the validated envelope
    unsupported a hard kernel constraint is violated (must not run)

    Whether a route is integrated at all is a separate fact, established only by measurement
    records with a passing correctness check (see data.py).
    """
    if op not in OPS or method not in METHODS:
        return 'unsupported', f'unknown method/op {method}/{op}'
    if tp < 1 or m % tp:
        return 'unsupported', f'M={m} not divisible by TP={tp}'
    if method == DEO:
        if op == 'ag':
            # DXAGGemmContext(M, K): vendor block GEMMs on the staged gather; only M % TP is required.
            return 'supported', 'dx_ag_gemm_op: M % TP only'
        if n % 128 or (m // tp) % 64:
            return 'unknown', (f'outside the validated vehicle envelope '
                               f'(N % 128 == 0 and (M/TP) % 64 == 0): N={n}, M/TP={m // tp}')
        return 'supported', 'dx_gemm_rs_op: inside the vehicle envelope (N % 128, (M/TP) % 64)'
    if method == 'async_tp':
        if ((m // tp) * k * 2) % 16:
            return 'unsupported', f'shard bytes not 16-aligned: (M/TP)*K*2 = {(m // tp) * k * 2}'
        return 'supported', 'torch.ops.symm_mem fused op: M % TP, 16-byte aligned shard'
    # triton_dist (DCU port, M_PER_CHUNK=1024)
    if op == 'ag':
        if m % 128 or n % 128 or k % 16 or (m // tp) % 1024:
            return 'unsupported', (f'ag_gemm_intra_node needs M % 128, N % 128, K % 16, '
                                   f'(M/TP) % 1024: M={m}, N={n}, K={k}, M/TP={m // tp}')
        return 'supported', 'ag_gemm_intra_node: M % 128, N % 128, K % 16, (M/TP) % 1024'
    if m % 256 or n % 256 or k % 256:
        return 'unsupported', f'gemm_rs_intra_node needs M % 256, N % 256, K % 256: M={m}, N={n}, K={k}'
    return 'supported', 'gemm_rs_intra_node: M % 256, N % 256, K % 256'


def heap_bytes(method: str, op: str, m: int, n: int, k: int, tp: int, *, deo_bwd: bool = True) -> int:
    """Symmetric-heap (DUSHMEM) bytes of one context of a BW route (both dtypes are 16-bit).

    async_tp uses torch's own symmetric-memory workspace, not the DUSHMEM heap. DEO row sites hold
    two contexts when the fused backward is on (GEMM+RS forward, AG+dgrad backward). The figures are
    what the allocators request (slot planes + flags, 4 KiB allowance per barrier); DEO planes and
    the triton_dist chunk size are the defaults (OPUS_AG_PLANES=1, M_PER_CHUNK=1024).
    """
    if tp < 1 or m % tp:
        return 0
    if method == DEO:
        slots = m * (k if op == 'ag' else n) * 2
        if op == 'rs' and deo_bwd:
            slots *= 2             # the GEMM+RS context and the AG+dgrad context hold M x N each
        return slots + 16 * tp
    if method == 'async_tp':
        return 0
    if op == 'ag':
        return m * k * 2 + 8 * tp * ((m // tp) // 1024) + 4096
    return m * n * 2 + 4096


def allocations_heap_bytes(allocate, tp: int, *, deo_bwd: bool = True) -> int:
    """Sum of heap_bytes over plan allocations [(method, op, m, n, k), ...] at the plan's TP."""
    return sum(heap_bytes(method, op, m, n, k, tp, deo_bwd=deo_bwd) for method, op, m, n, k in allocate)


def runtime_platform(*, tp: Optional[int] = None, topology: Optional[str] = None, probe_device: bool = True,
                      device_name: Optional[str] = None, arch: Optional[str] = None) -> dict:
    """Platform block as seen by the running process (torch is imported, HIP is not initialised).

    device_name / arch are informational. A GPU process probes its own device (the first HIP call of a
    process costs about 14 s on a BW1000 node); an offline tool sets probe_device=False and may pass
    what the measured samples recorded, so that generating a plan never opens a device that a running
    experiment is using."""
    import torch
    tp = tp or torch.cuda.device_count()
    if probe_device:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        device_name = torch.cuda.get_device_name()
        arch = str(getattr(props, 'gcnArchName', '') or '').split(':')[0]
    return {'vendor': VENDOR, 'device': DEVICE, 'device_name': device_name, 'arch': arch,
            'topology': topology or f'BW{tp}',
            'versions': [list(v) for v in detect_versions()], 'comm': COMM, 'symmetric': SYMMETRIC}


def platform_mismatches(plan_platform: dict, runtime: dict) -> list:
    """Fields on which a plan's platform differs from the running platform (empty list = same)."""
    out = []
    for key in PLATFORM_KEYS:
        want, have = plan_platform.get(key), runtime.get(key)
        if key == 'versions':
            want = [list(v) for v in want] if want is not None else None
            have = [list(v) for v in have] if have is not None else None
        if key == 'arch' and not (want and have):
            continue                      # arch is informational where either side cannot see it
        if want != have:
            out.append(f'{key}: plan={want!r} runtime={have!r}')
    return out
