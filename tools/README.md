| Tool | Use |
|---|---|
| `check_env.py` | environment check after `source env/env.sh` (no GPU work) |
| `e2e_summary.py <run dir>` | per-arm / per-kind table of one Megatron run (written to `summary.md` automatically) |
| `dual16_report.py <run dir>` | 16-card job table (both nodes, PLAN vs ALT, RDMA verdict) |
| `vllm_report.py <run dir> <M>` | vLLM arm table (gate consistency, ttft / ttft2 / thru) |
| `oprs_plans.py <new dir> --tp N --backend B --workload W --data samples.jsonl` | OPRS: plan set of one deployment (predicted / two-way / constant arms / Null + manifest) |
| `oprs_heap.py <plan or plan-set dir> [--deo-bwd ag\|stock]` | OPRS: one symmetric-heap size for every arm of a comparison |
| `oprs_tables.py training\|inference\|operator\|choices\|sites …` | OPRS: end-to-end summaries, `tables/*.csv` of a results root |
| `oprs_tiny_model.py <new dir>` | OPRS: weight-less tiny Llama vehicle for the single-card vLLM plumbing check |

The `oprs_*` tools are documented in `docs/BW1000_OPRS移植说明_20261126.md`.
