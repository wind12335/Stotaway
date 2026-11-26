"""Explicit path table and model dispatch for the BW routes (CPU only).

Hygon port of nvidia/planner/original_a800/predictor_registry.py. Detection consumes trusted
integration metadata (method, actual path, direction). It does not infer kernels from imports or
names and does not modify any backend at runtime. Fallback and unknown paths are rejected.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
from typing import Mapping

from .predictor import SERIAL, ContextRequirement, Route, RouteRegressor
from .route_local_residual import RouteLocalResidual
from .route_policy import RoutePolicy


@dataclass(frozen=True)
class Capability:
    compute: str
    transport: str
    memory: str
    synchronization: str
    actual_path: str
    direction: str


class UnsupportedRoute(ValueError):
    pass


# (method, actual_path, direction) -> implementation path on BW1000 / DTK 26.04
PATHS = {
    # DEO: block vendor GEMMs (hipBLASLt via torch.matmul); data pushed by hipMemcpyAsync (the copy
    # engine) into the peer's symmetric-heap VA; tickets = engine 4-byte writes after the data on
    # the same stream; waits = hipStreamWaitValue32 GTE on OWN cells; acks = SM sys-release xchg.
    ('deo', 'direct_exchange', 'ag'): Capability('vendor_gemm', 'hip_engine_memcpy', 'dushmem_symmetric',
                                                 'stream_wait_ticket_ack', 'direct_exchange', 'ag'),
    ('deo', 'direct_exchange', 'rs'): Capability('vendor_gemm', 'hip_engine_memcpy', 'dushmem_symmetric',
                                                 'stream_wait_ticket_ack', 'direct_exchange', 'rs'),
    # Async-TP: torch.ops.symm_mem.{fused_all_gather_matmul,fused_matmul_reduce_scatter} of the
    # torch-HIP build (torch.matmul inside; torch symmetric memory with IPC-imported peer buffers).
    ('async_tp', 'torch_ops_symm_mem', 'ag'): Capability('pytorch_gemm', 'torch_symm_mem', 'torch_symmetric_memory',
                                                         'symm_mem_signal', 'torch_ops_symm_mem', 'ag'),
    ('async_tp', 'torch_ops_symm_mem', 'rs'): Capability('pytorch_gemm', 'torch_symm_mem', 'torch_symmetric_memory',
                                                         'symm_mem_signal', 'torch_ops_symm_mem', 'rs'),
    # Triton-distributed DCU port: triton GEMM kernels; data by copy engine (hipMemcpyAsync);
    # AG flags by host dushmem putmem after an event sync (SIGNAL_MODE=host), consumer spins per
    # chunk; RS gated by SymmBarrier rounds.
    ('triton_dist', 'triton_dist_dcu', 'ag'): Capability('triton', 'hip_engine_memcpy_rma_flags', 'dushmem_symmetric',
                                                         'chunk_flag_spin', 'triton_dist_dcu', 'ag'),
    ('triton_dist', 'triton_dist_dcu', 'rs'): Capability('triton', 'hip_engine_memcpy', 'dushmem_symmetric',
                                                         'round_barrier', 'triton_dist_dcu', 'rs'),
}


def detect(metadata: Mapping[str, object]) -> Capability:
    """Recognize a concrete adapter-reported method/path/direction triple.

    Returned labels describe an implementation path, not a guarantee that a shape, config or
    binary is usable. Fallback and unknown paths are rejected.
    """
    key = tuple(metadata.get(k) for k in ('method', 'actual_path', 'direction'))
    if metadata.get('fallback') is not False:
        raise UnsupportedRoute('fallback status must be explicitly false')
    try:
        return PATHS[key]
    except (KeyError, TypeError):
        raise UnsupportedRoute(f'unrecognized execution path: {key}') from None


def bind(metadata: Mapping[str, object], *, gpu: str, topology: str, tp: int, dtype: str,
         versions: tuple, timing_mode: str, warmup, config_family: str) -> Route:
    capability = detect(metadata)
    return Route(method=str(metadata['method']), compute=capability.compute,
                 transport=capability.transport, memory=capability.memory,
                 synchronization=capability.synchronization, direction=capability.direction,
                 actual_path=capability.actual_path, gpu=gpu, topology=topology, tp=tp,
                 dtype=dtype, versions=versions, timing_mode=timing_mode, warmup=warmup,
                 config_family=config_family)


class PredictorRegistry:
    def __init__(self):
        self._models = {}
        self._generation = 0

    def register(self, model):
        if isinstance(model, RouteRegressor):
            if model.parameters is None:
                raise ValueError('calibrate a route model before registering')
            snapshot = RouteRegressor.from_dict(model.to_dict())
        elif isinstance(model, RoutePolicy):
            if model.model is None:
                raise ValueError('calibrate a route policy before registering')
            snapshot = RoutePolicy.from_dict(model.to_dict())
        elif isinstance(model, RouteLocalResidual):
            if model.model is None:
                raise ValueError('calibrate a local residual model before registering')
            snapshot = RouteLocalResidual.from_dict(model.to_dict())
        else:
            raise TypeError('unsupported route model')
        if model.route in self._models:
            raise ValueError('route already registered')
        # Freeze an internal copy so later caller-side refits cannot silently
        # invalidate predictions already held in an inference cache.
        self._models[snapshot.route] = snapshot
        self._generation += 1

    def predict(self, route: Route, shape, *, capability: ContextRequirement, **features):
        if capability.route != route or capability.shape != shape:
            raise ValueError('capability evidence does not match route and shape')
        if capability.support != 'supported':
            return {'choice': SERIAL, 'status': f'shape capability {capability.support}',
                    'predicted_log_speedup': None}
        model = self._models.get(route)
        if model is None:
            return {'choice': SERIAL, 'status': 'no calibrated model for actual route',
                    'predicted_log_speedup': None}
        return model.choose(route, shape, **features)


class CachedRouteSelector:
    """Process-local inference cache for one registered route model.

    This is not distributed synchronization; callers must coordinate ranks.
    Unknown support returns serial and is cached under its own evidence key.
    """
    def __init__(self, registry: PredictorRegistry, *, capacity=4096):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError('positive capacity required')
        self.registry, self.capacity = registry, capacity
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def predict(self, route, shape, *, capability, gemm_ms=None, comm_ms=None, margin_log=0.):
        if capability.route != route or capability.shape != shape:
            raise ValueError('capability evidence does not match route and shape')
        if not math.isfinite(margin_log) or margin_log < 0:
            raise ValueError('invalid margin')
        for value in (gemm_ms, comm_ms):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('component timings must be finite and positive')
        key = (self.registry._generation, route, shape, capability.support, capability.evidence,
               gemm_ms, comm_ms, margin_log)
        with self._lock:
            if key not in self._cache:
                result = self.registry.predict(route, shape, capability=capability, gemm_ms=gemm_ms,
                                               comm_ms=comm_ms, margin_log=margin_log)
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
                self._cache[key] = dict(result)
            else:
                self._cache.move_to_end(key)
            return dict(self._cache[key])

    def clear(self):
        with self._lock:
            self._cache.clear()
