"""One DUSHMEM world per process, shared by every overlap path (DEO and the triton_dist DCU port).

Hygon counterpart of the engine state that the NVIDIA runtime keeps as module globals
(nvidia/stowaway_nvidia/dx8.py, external_tp8.py). On BW every overlap path reaches its peers only
through pydushmem (DUSHMEM = the Hygon symmetric-heap library, the counterpart of the symmetric
memory the NVIDIA kernels use): one MPI communicator, one rank id, one DushmemHost. "One per
process (dushmem global state)": a second host or communicator inside the same process is not
supported, so all backends of a dispatcher must receive THIS object.

The symmetric heap (DUSHMEM_SYMMETRIC_SIZE) is reserved when the host is initialised, i.e. when
this object is built -- before the framework sizes its own memory (vLLM: before LLM(); Megatron: at
train() entry, after the model was built, like the vehicle).

  EngineWorld.standalone()   bench / tests: entry-script prelude; call it BEFORE importing torch
  EngineWorld.over_group(g)  Megatron: torch.distributed is up and `g` is the TP process group; the
                              MPI world is MPI_COMM_WORLD and the TP group is cut out with
                              MPI_Comm_split -- collective over the whole job, so EVERY world rank
                              must call it at the same point (lazy bootstrap from inside a layer
                              deadlocks)
"""
import ctypes
import os

DEFAULT_HEAP = 1 << 33            # DUSHMEM runtime default when DUSHMEM_SYMMETRIC_SIZE is unset (8 GiB)


def configured_heap() -> int:
    return int(os.environ.get('DUSHMEM_SYMMETRIC_SIZE') or DEFAULT_HEAP)


class EngineWorld:
    def __init__(self, mpi, comm, rank, size, host, *, group=None):
        from stowaway.kernels.common_ops import init_dcu
        init_dcu(mpi, comm, rank, size, host=host)           # idempotent: one module state per process
        self.mpi, self.comm, self.rank, self.size, self.host = mpi, comm, rank, size, host
        self.group = group            # torch process group of the same ranks (None for pure-kernel use)

    @classmethod
    def standalone(cls, *, port: int = 29987) -> 'EngineWorld':
        """Bench / test prelude. Must run before `import torch` (torch-MPI interposition order:
        the real MPI library is loaded RTLD_GLOBAL first). One mpirun of `-np R` is expected."""
        import pydushmem.host as pHost
        mpi, comm = pHost.mpi_init()
        rank = pHost.mpi_comm_rank(mpi, comm)
        size = pHost.mpi_comm_size(mpi, comm)
        hip = ctypes.CDLL(os.environ.get('SW_HIP_LIB') or '/opt/dtk-26.04/lib/libamdhip64.so')
        hip.hipSetDevice.restype = ctypes.c_int
        assert hip.hipSetDevice(rank) == 0
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(port))
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(rank)
        # device_id: no 'barrier(): using the device under current context' warning on every barrier
        dist.init_process_group('nccl', rank=rank, world_size=size, device_id=torch.device('cuda', rank))
        probe = torch.ones(1, device='cuda')
        for _ in range(3):                       # RCCL communicator up before anything is timed
            dist.all_reduce(probe)
        torch.cuda.synchronize()
        host = pHost.DushmemHost(preload_bootstrap=False)
        host.init_mpi(comm)
        return cls(mpi, comm, rank, size, host, group=dist.group.WORLD)

    @classmethod
    def over_group(cls, group) -> 'EngineWorld':
        """The engine world over one torch process group (the TP group). All world ranks together."""
        import torch.distributed as dist
        import pydushmem.host as pHost
        mpi, world = pHost.mpi_init()
        granks = dist.get_process_group_ranks(group)
        color, key = min(granks), dist.get_rank()          # keep the torch-rank order inside the group
        # OMPI: MPI_Comm is a pointer -- the OUT parameter is pointer-to-pointer (a POINTER(c_int)
        # truncates the handle and a later call segfaults; bench/s62_split_repro.py). pHost
        # convention: handles travel as integer addresses.
        mpi.MPI_Comm_split.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_void_p)]
        mpi.MPI_Comm_split.restype = ctypes.c_int
        newcomm = ctypes.c_void_p(0)
        rc = mpi.MPI_Comm_split(ctypes.c_void_p(world), color, key, ctypes.byref(newcomm))
        assert rc == 0, f'MPI_Comm_split rc={rc}'
        assert newcomm.value, 'MPI_Comm_split returned NULL'
        comm = newcomm.value
        rank, size = pHost.mpi_comm_rank(mpi, comm), pHost.mpi_comm_size(mpi, comm)
        assert size == group.size() and rank == granks.index(key), \
            f'subgroup mismatch r={rank} n={size} granks={granks}'
        host = pHost.DushmemHost(preload_bootstrap=False)
        host.init_mpi(comm)
        return cls(mpi, comm, rank, size, host, group=group)
