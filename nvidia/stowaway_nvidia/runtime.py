import ctypes
import os

import nvshmem.core
import torch
import torch.distributed as dist
from cuda.core.experimental import Device


def finalize():
    nvshmem.core.finalize()
    dist.destroy_process_group()


def _assert_single_node(world_size):
    identities = [None] * world_size
    node_id = (os.environ.get("GROUP_RANK"), os.uname().nodename)
    dist.all_gather_object(identities, node_id)
    if any(identity != identities[0] for identity in identities):
        raise ValueError(f"this prototype requires one host, got {identities}")


class CudaOps:
    def __init__(self):
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.runtime = ctypes.CDLL("libcudart.so.12")
        self.driver.cuStreamWaitValue32_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint
        ]
        self.driver.cuStreamWaitValue32_v2.restype = ctypes.c_int
        self.driver.cuStreamWriteValue32_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint
        ]
        self.driver.cuStreamWriteValue32_v2.restype = ctypes.c_int
        self.runtime.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self.runtime.cudaMemcpyAsync.restype = ctypes.c_int

    def wait(self, address, value, stream):
        status = self.driver.cuStreamWaitValue32_v2(
            stream.cuda_stream, address, value, 0
        )
        if status:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {status}")

    def copy(self, destination, source, size, stream):
        status = self.runtime.cudaMemcpyAsync(
            destination, source, size, 4, stream.cuda_stream
        )
        if status:
            raise RuntimeError(f"cudaMemcpyAsync failed: {status}")

    def write(self, address, value, stream):
        status = self.driver.cuStreamWriteValue32_v2(stream.cuda_stream, address, value, 0)
        if status:
            raise RuntimeError(f"cuStreamWriteValue32 failed: {status}")


class IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


class IpcBuffer:
    def __init__(self, cuda, size, cpu_group):
        self.cuda = cuda
        self.cpu_group = cpu_group
        self.rank = dist.get_rank(group=cpu_group)
        self.local = ctypes.c_void_p()
        runtime = cuda.runtime
        runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        runtime.cudaMalloc.restype = ctypes.c_int
        runtime.cudaFree.argtypes = [ctypes.c_void_p]
        runtime.cudaFree.restype = ctypes.c_int
        runtime.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcHandle), ctypes.c_void_p]
        runtime.cudaIpcGetMemHandle.restype = ctypes.c_int
        runtime.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), IpcHandle, ctypes.c_uint]
        runtime.cudaIpcOpenMemHandle.restype = ctypes.c_int
        runtime.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        runtime.cudaIpcCloseMemHandle.restype = ctypes.c_int
        self._check(runtime.cudaMalloc(ctypes.byref(self.local), size), "cudaMalloc")
        handle = IpcHandle()
        self._check(runtime.cudaIpcGetMemHandle(ctypes.byref(handle), self.local), "cudaIpcGetMemHandle")
        handles = [None] * dist.get_world_size(group=cpu_group)
        dist.all_gather_object(handles, bytes(handle), group=cpu_group)
        self.peer = ctypes.c_void_p()
        peer_handle = IpcHandle.from_buffer_copy(handles[1 - self.rank])
        self._check(runtime.cudaIpcOpenMemHandle(ctypes.byref(self.peer), peer_handle, 1), "cudaIpcOpenMemHandle")

    @staticmethod
    def _check(status, name):
        if status:
            raise RuntimeError(f"{name} failed: CUDA error {status}")

    def close(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.cpu_group)
        self._check(self.cuda.runtime.cudaIpcCloseMemHandle(self.peer), "cudaIpcCloseMemHandle")
        dist.barrier(group=self.cpu_group)
        self._check(self.cuda.runtime.cudaFree(self.local), "cudaFree")
        dist.barrier(group=self.cpu_group)


class SymmetricBuffers:
    def __init__(self, rank, world_size, specifications):
        if not dist.is_initialized() or dist.get_world_size() != world_size or dist.get_rank() != rank:
            raise ValueError("symmetric allocations require matching initialized world ranks")
        # NVSHMEM allocations are collective: incompatible allocation sizes or
        # ordering can silently produce mismatched addresses or hang. Exchange
        # the complete allocation layout before any symmetric allocation.
        layout = tuple((tuple(shape), str(dtype)) for shape, dtype in specifications)
        layouts = [None] * world_size
        dist.all_gather_object(layouts, layout)
        if any(candidate != layout for candidate in layouts):
            raise ValueError(f"symmetric allocation layouts differ across ranks: {layouts}")
        self.rank = rank
        self.buffers = []
        library = ctypes.CDLL("libnvshmem_host.so.3")
        library.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.nvshmem_ptr.restype = ctypes.c_void_p
        for shape, dtype in specifications:
            torch.cuda.synchronize()
            local = nvshmem.core.tensor(shape, dtype=dtype)
            torch.cuda.synchronize()
            peers = [local if peer == rank else library.nvshmem_ptr(local.data_ptr(), peer)
                     for peer in range(world_size)]
            if any(pointer is None for pointer in peers):
                raise RuntimeError("NVSHMEM peer pointer unavailable; same-node direct access required")
            self.buffers.append(peers)
            peers[rank].zero_()
        torch.cuda.synchronize()
        dist.barrier()

    def close(self):
        torch.cuda.synchronize()
        dist.barrier()
        for peers in reversed(self.buffers):
            local = peers[self.rank]
            nvshmem.core.free_tensor(local)
            peers.clear()
        self.buffers.clear()
        dist.barrier()


def initialize(*, expected_world_size=2):
    if not dist.is_initialized():
        raise RuntimeError("initialize the torch process group first")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (world_size != expected_world_size or local_size != world_size or rank != local_rank
            or torch.cuda.device_count() < world_size or torch.cuda.current_device() != local_rank):
        raise ValueError(f"this prototype requires exactly {expected_world_size} local GPUs")
    _assert_single_node(world_size)
    unique_id = [nvshmem.core.get_unique_id(empty=rank != 0)]
    dist.broadcast_object_list(unique_id, src=0)
    dist.barrier()
    nvshmem.core.init(device=Device(torch.cuda.current_device()), uid=unique_id[0],
                      rank=rank, nranks=world_size, initializer_method="uid")
    return CudaOps()
