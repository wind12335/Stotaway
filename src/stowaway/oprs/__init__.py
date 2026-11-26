"""OPRS on the Hygon BW1000 stack (DUSHMEM symmetric heap, PyTorch-HIP, RCCL).

Port of the NVIDIA-side OPRS framework of this branch (planner
``nvidia/planner/original_a800``, runtime ``nvidia/stowaway_nvidia``) to the
Hygon stack, following docs/BW1000八卡_OPRS迁移与实验交接_20261126.md.
How to run it and what the port changed: docs/BW1000_OPRS移植说明_20261126.md.

OPRS = fix ONE external overlap backend B (async_tp | triton_dist), predict per
call site whether B or DEO (Hygon's own overlap method; NVIDIA calls it DX)
beats the serial path (AG -> GEMM, or GEMM -> RS), and emit a JSON plan with one
choice of {serial, B, deo} per site that a runtime adapter (dispatcher) executes
inside Megatron / vLLM. A method is a candidate at a site only if it was measured
on this machine and its shape is inside the static support of its route.

Layout
  predictor.py, route_local_residual.py, route_policy.py, registry.py   CPU (numpy): per-route models
  platform.py, data.py, workloads.py, arms.py       CPU: BW routes, sample records, site shapes, arm names
  evaluate.py, export_plan.py, auto_plan.py         CPU: family-holdout evaluation, plan export / load
  engine.py, backends/, dispatcher.py, autograd.py  GPU: DUSHMEM engine world, the three methods, dispatch
  megatron_hook.py, vllm_hook.py                    GPU: framework adapters

Entry points
  bench/run_oprs_smoke.sh, bench/run_oprs_matrix.sh, bench/run_oprs_bitwise.sh
  e2e/megatron/run_oprs.sh, e2e/vllm/run_oprs.sh, tools/oprs_plans.py, tools/oprs_tables.py

``import stowaway`` imports none of this; import submodules explicitly. Tests: python3 -m pytest tests (CPU).
"""
