"""Environment check (no GPU work, ~10 s):  source env/env.sh && python3 tools/check_env.py

Checks toolchain versions, the libraries the engine loads (HIP, dushmem, MPI, hipBLASLt shim),
third-party paths, and that every stowaway module imports. Prints OK / WARN / FAIL per line;
exit code 1 if any FAIL.
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
bad = 0


def say(level, msg):
    global bad
    bad += level == "FAIL"
    print(f"[{level:4}] {msg}")


def path(var, what, need=True):
    p = os.environ.get(var, "")
    if p and os.path.exists(p):
        say("OK", f"{var} = {p}")
    else:
        say("FAIL" if need else "WARN", f"{var} = {p or '(unset)'}: {what}")


import torch  # noqa: E402
import triton  # noqa: E402
say("OK", f"python {sys.version.split()[0]}  torch {torch.__version__}  triton {triton.__version__}")
say("OK" if torch.cuda.is_available() else "WARN", f"devices visible: {torch.cuda.device_count()}")
if "SW_ROOT" not in os.environ:
    say("FAIL", "env/env.sh not sourced (SW_ROOT unset)")
path("SW_HIP_LIB", "libamdhip64.so of DTK")
path("DUSHMEM_LIBDIR", "dushmem host library dir (DTK lib/dushmem)")
path("PYDUSHMEM_MPI_LIB", "libmpi used for the engine bootstrap (torch.libs/libmpi-*.so)")
path("MEGATRON_DAS", "Megatron-LM-das checkout (e2e/megatron only)", need=False)
path("SW_TOKENIZER", "HF tokenizer dir for the dense vehicles (e2e/megatron only)", need=False)
path("SW_MOE_MODEL", "Qwen3-30B-A3B config/tokenizer dir (MoE vehicle only)", need=False)
path("SW_VLLM_MODEL", "LLaMA3-70B weights (vLLM bench only)", need=False)
so = os.path.join(ROOT, "src", "csrc", "libmoe_ggemm.so")
say("OK" if os.path.exists(so) else "WARN", f"{so} {'built' if os.path.exists(so) else 'missing: run src/csrc/build.sh (MoE only)'}")
if os.environ.get("HSA_FORCE_FINE_GRAIN_PCIE") != "1":
    say("FAIL", "HSA_FORCE_FINE_GRAIN_PCIE must be 1")
heap = int(os.environ.get("DUSHMEM_SYMMETRIC_SIZE", "0"))
say("OK" if heap >= 4 << 30 else "WARN", f"DUSHMEM_SYMMETRIC_SIZE = {heap / 2**30:.1f} GB")

mods = ["pydushmem.host", "stowaway.kernels.common_ops", "stowaway.kernels.gemm_rs_dx",
        "stowaway.kernels.rs_relay", "stowaway.kernels.gemm_ar_rd", "stowaway.kernels.moe_ggemm",
        "stowaway.kernels.moe_expert_mlp", "stowaway.kernels.moe_sf",
        "stowaway.integrations.opus_moe_gg_patch", "stowaway.integrations.opus_moe_sfl_patch"]
for m in mods:
    try:
        importlib.import_module(m)
        say("OK", f"import {m}")
    except Exception as e:  # noqa: BLE001
        say("FAIL", f"import {m}: {type(e).__name__}: {e}")
if os.environ.get("MEGATRON_DAS") and os.path.isdir(os.environ["MEGATRON_DAS"]):
    for p in (os.environ["MEGATRON_DAS"], os.environ.get("MEGATRON_LM", "")):
        sys.path.insert(0, p)
    for m in ("stowaway.integrations.mgd_tp_patch", "stowaway.integrations.mgd_ag_patch",
              "stowaway.integrations.mgd_brs_patch", "stowaway.integrations.mgd_rs_attr"):
        try:
            importlib.import_module(m)
            say("OK", f"import {m}")
        except Exception as e:  # noqa: BLE001
            say("FAIL", f"import {m}: {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
