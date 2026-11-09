"""Stowaway: hide tensor-parallel communication behind the compute that is already there.

Layout
  stowaway.kernels       device code: overlap operators (DX direct push), rings, MoE sync-free path
  stowaway.integrations  monkey-patch hooks for Megatron-LM (training) and vLLM (prefill);
                         every hook is OFF unless its MGD_* / OPUS_* / VLLM_SP switch is set
The planner (predictor) lives outside the package in <repo>/planner (pure python, CPU).
Kernel layer derived from ByteDance triton-distributed (MIT, see LICENSE.triton-dist).
"""
__version__ = "0.1.0"
