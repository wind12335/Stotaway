| Tool | Use |
|---|---|
| `check_env.py` | environment check after `source env/env.sh` (no GPU work) |
| `e2e_summary.py <run dir>` | per-arm / per-kind table of one Megatron run (written to `summary.md` automatically) |
| `dual16_report.py <run dir>` | 16-card job table (both nodes, PLAN vs ALT, RDMA verdict) |
| `vllm_report.py <run dir> <M>` | vLLM arm table (gate consistency, ttft / ttft2 / thru) |
