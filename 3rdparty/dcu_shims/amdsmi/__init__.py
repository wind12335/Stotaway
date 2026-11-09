"""amdsmi shim for DCU (DTK).

vLLM's ROCm platform plugin imports these names from `amdsmi`.
The real amdsmi wheel is ROCm-only and absent on DTK; we bridge to the
ROCm SMI C API that DTK ships instead (/opt/dtk-26.04/.hyhal/rocm_smi).

Only the surface vLLM touches is implemented:
  amdsmi_init / amdsmi_shut_down
  amdsmi_get_processor_handles  -> list of monitor-device indices
  amdsmi_topo_get_link_type(a,b) -> {"hops": int, "type": int} (2 = XGMI)
  amdsmi_get_gpu_asic_info(h)    -> {"market_name": str}
"""
import ctypes
import os

__all__ = [
    "AmdSmiException", "amdsmi_init", "amdsmi_shut_down",
    "amdsmi_get_processor_handles", "amdsmi_topo_get_link_type",
    "amdsmi_get_gpu_asic_info",
]


class AmdSmiException(Exception):
    pass


def _load_rsmi():
    # LD_LIBRARY_PATH first, then explicit DTK locations / our symlink.
    cands = [
        "librocm_smi64.so.2",
        "librocm_smi64.so",
        "/opt/dtk-26.04/.hyhal/rocm_smi/lib/librocm_smi64.so.2.8",
        "/opt/dtk-extra/librocm_smi64.so.2",
        "/opt/dtk-extra/librocm_smi64.so",
        "/opt/dtk-26.04/lib/librocm_smi64.so.2",
    ]
    for p in cands:
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    return None


_lib = _load_rsmi()
_initialized = False


def amdsmi_init(*_a, **_k):
    global _initialized
    if _lib is None:
        raise AmdSmiException("librocm_smi64 not found on this system")
    rc = _lib.rsmi_init(ctypes.c_uint64(0))
    if rc != 0:
        raise AmdSmiException(f"rsmi_init rc={rc}")
    _initialized = True
    return None


def amdsmi_shut_down(*_a, **_k):
    global _initialized
    if _lib is None or not _initialized:
        return None
    rc = _lib.rsmi_shut_down()
    _initialized = False
    if rc != 0:
        raise AmdSmiException(f"rsmi_shut_down rc={rc}")
    return None


def _num_devices() -> int:
    n = ctypes.c_uint32(0)
    rc = _lib.rsmi_num_monitor_devices(ctypes.byref(n))
    if rc != 0:
        raise AmdSmiException(f"rsmi_num_monitor_devices rc={rc}")
    return n.value


def amdsmi_get_processor_handles():
    """Handles == monitor device indices (RSMI works on those directly)."""
    return list(range(_num_devices()))


def amdsmi_topo_get_link_type(src, dst):
    _lib.rsmi_topo_get_link_type.restype = ctypes.c_int
    _lib.rsmi_topo_get_link_type.argtypes = [
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
    ]
    hops = ctypes.c_uint64(0)
    ltype = ctypes.c_int(0)
    rc = _lib.rsmi_topo_get_link_type(src, dst,
                                      ctypes.byref(hops), ctypes.byref(ltype))
    if rc != 0:
        raise AmdSmiException(f"rsmi_topo_get_link_type({src},{dst}) rc={rc}")
    return {"hops": hops.value, "type": ltype.value}


def amdsmi_get_gpu_asic_info(handle):
    _lib.rsmi_dev_name_get.restype = ctypes.c_int
    _lib.rsmi_dev_name_get.argtypes = [
        ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t,
    ]
    buf = ctypes.create_string_buffer(256)
    rc = _lib.rsmi_dev_name_get(handle, buf, 256)
    name = buf.value.decode(errors="replace") if rc == 0 else "unknown"
    return {"market_name": name}
