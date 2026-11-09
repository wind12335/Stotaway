# Source me:  source env/env.sh
# Every path below can be overridden by exporting it before sourcing.
SW_ROOT=${SW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export SW_ROOT

# ---- DTK (Hygon DCU toolkit, tested with 26.04) ------------------------------------------------
export DTK_HOME=${DTK_HOME:-/opt/dtk-26.04}
[ -f "$DTK_HOME/env.sh" ] && source "$DTK_HOME/env.sh" >/dev/null 2>&1
export SW_HIP_LIB=${SW_HIP_LIB:-$DTK_HOME/lib/libamdhip64.so}
export DUSHMEM_LIBDIR=${DUSHMEM_LIBDIR:-$DTK_HOME/lib/dushmem}
export HSA_FORCE_FINE_GRAIN_PCIE=1          # required for cross-card fine-grained stores / flags

# ---- python path: our package + pydushmem bindings ---------------------------------------------
export PYTHONPATH=$SW_ROOT/src${PYTHONPATH:+:$PYTHONPATH}

# ---- MPI library used by pydushmem's bootstrap (the one torch ships, so both agree) ------------
if [ -z "${PYDUSHMEM_MPI_LIB:-}" ]; then
    PYDUSHMEM_MPI_LIB=$(python3 -I -c "import glob,os,torch;print((glob.glob(os.path.join(os.path.dirname(torch.__file__),'..','torch.libs','libmpi-*.so*'))+[''])[0])" 2>/dev/null)
    [ -z "$PYDUSHMEM_MPI_LIB" ] && PYDUSHMEM_MPI_LIB=/opt/mpi/lib/libmpi.so
fi
export PYDUSHMEM_MPI_LIB

# ---- symmetric heap per rank (bytes). 8 GB covers every bench and the 27B/70B vehicles ---------
export DUSHMEM_SYMMETRIC_SIZE=${DUSHMEM_SYMMETRIC_SIZE:-8589934592}

# ---- third-party checkouts (see 3rdparty/README.md) --------------------------------------------
export MEGATRON_DAS=${MEGATRON_DAS:-$SW_ROOT/3rdparty/Megatron-LM-das}
export MEGATRON_LM=${MEGATRON_LM:-$MEGATRON_DAS/3rdparty/Megatron-LM}
# extra site-packages for Megatron / vLLM on DCU (amdsmi shim, flash_attn wheel, lightop stub ...)
export SW_PYSITE=${SW_PYSITE:-}
# tokenizer only sets the vocab (training runs on --mock-data); any HF tokenizer dir with vocab >= 248320
export SW_TOKENIZER=${SW_TOKENIZER:-$SW_ROOT/3rdparty/tokenizers/Qwen3.6-27B}
# MoE vehicle (Qwen3-30B-A3B config + tokenizer)
export SW_MOE_MODEL=${SW_MOE_MODEL:-$SW_ROOT/3rdparty/models/Qwen3-30B-A3B}
# vLLM bench model (LLaMA3-70B HF weights)
export SW_VLLM_MODEL=${SW_VLLM_MODEL:-$SW_ROOT/3rdparty/models/LLAMA3-70B}

# ---- mpirun line shared by every 8-card script -------------------------------------------------
export SW_NP=${SW_NP:-8}
SW_MPIRUN="mpirun --allow-run-as-root -np $SW_NP --bind-to none -mca coll ^hcoll \
 -x HSA_FORCE_FINE_GRAIN_PCIE -x PYTHONPATH -x PYDUSHMEM_MPI_LIB -x DUSHMEM_SYMMETRIC_SIZE -x DUSHMEM_LIBDIR -x SW_HIP_LIB"
export SW_MPIRUN
ulimit -c 0
