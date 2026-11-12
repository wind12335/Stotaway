"""A1 shape table -- every (H, F) pair below comes from a published 24-26
model config (HF config.json / params.json / tech report). The A1 matrix
is therefore exhaustive over the published model space instead of an
arbitrary M/N/K grid.

RS/AR consume row-parallel GEMMs (o_proj / fc2): N = H, K_global = F
(per-rank K = F // R at TP=R).
AG consumes the gate_up column-parallel entry: K = H, N_global = 2*F
(per-rank N = 2*F // R).

Sources (checked 2026-10):
  Qwen3-8B   : HF config.json (local weights, verified)
  Llama3-8B  : public config.json
  Qwen3-32B  : qwenlm.github.io/blog/qwen3 + config
  Llama4-Mav : ai.meta.com/blog/llama-4 + unsloth config (5120/8192e/16384d)
  Gemma3-27B : public config.json
  Mistral-L3 : params.json fetched 2026-10-22 (dim 7168, dense 16384,
               expert 4096, 128e top-4, 61L) -- 675B granular MoE
  MiniMax-M1 : config.json fetched 2026-10-22 (6144, expert 9216, 32e top-2)
  DeepSeek-V3: arxiv 2412.19437 (7168, dense 18432, expert 2048)
  Kimi K2    : github moonshotai/kimi-k2 config (7168, expert 2048, 384e)
  Llama3-70B : HF config.json (local weights)
  Qwen2.5-72B: public config (8192 / 29568)
  Mistral-L  : public config 2407 (12288 / 28672)
  Llama3.1-405B: meta blog + param-count self-check (16384 / 53248)
  GLM-4.5    : 5120 / expert 1536 (160 experts)
  Qwen3-235B : qwen3 report (4096 / expert 1536)
"""

# (key, H, F, kind, models)  -- kind: dense = full FFN, expert = per-expert
# FFN shape class (EP-style expert GEMM, fc2 orientation N=H K=Fe).
DENSE = [
    ("qwen3-8b",        4096, 12288, "Qwen3-8B"),
    ("llama3-8b",       4096, 14336, "Llama3-8B"),
    ("qwen3-32b",       5120, 10240, "Qwen3-32B"),
    ("llama4-mav-d",    5120, 16384, "Llama4-Maverick dense layers"),
    ("gemma3-27b",      5376, 21504, "Gemma3-27B"),
    ("mistral3-d",      7168, 16384, "Mistral-Large-3 dense (first 3)"),
    ("dsv3-d",          7168, 18432, "DeepSeek-V3 dense (first 1)"),
    ("llama3-70b",      8192, 28672, "Llama3-70B"),
    ("qwen25-72b",      8192, 29568, "Qwen2.5-72B"),
    ("mistral-l-2407", 12288, 28672, "Mistral-Large-2407"),
    ("llama31-405b",   16384, 53248, "Llama3.1-405B"),
]

EXPERT = [
    ("dsv3-k2-e",       7168,  2048, "DeepSeek-V3 + Kimi-K2 experts"),
    ("mistral3-e",      7168,  4096, "Mistral-Large-3 experts"),
    ("llama4-mav-e",    5120,  8192, "Llama4-Maverick experts"),
    ("minimax-m1-e",    6144,  9216, "MiniMax-M1 experts"),
    ("glm45-e",         5120,  1536, "GLM-4.5 experts"),
    ("qwen3-235b-e",    4096,  1536, "Qwen3-235B-A22B experts"),
]

# (key, H, F, kind, note) -- kind dense|expert for the runner
SHAPES = [(k, h, f, "dense", m) for k, h, f, m in DENSE] + \
         [(k, h, f, "expert", m) for k, h, f, m in EXPERT]

M_SWEEP = [1024, 2048, 4096, 8192, 16384, 32768]   # seq4096 x mbs 1..8
