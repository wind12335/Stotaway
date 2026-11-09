Micro-probes behind the platform laws the planner uses (results in `results/findings.md`).

| Probe | Law | Ranks |
|---|---|---|
| `perm_bw.py` | permutation-order engine push ~141 GB/s/rank to any peer, fan-out/incast slower (F-perm) | 8 |
| `copy_hostblock_plain.py` | engine copy into ordinary peer memory: value-gated fork releases the host | 2 |
| `copy_hostblock_heap.py` | engine copy into another process's dushmem heap pulls the host back to ~25% of the GPU lag, regardless of fork style / size / count (F50 → stage-2 term) | 2 |
| `graph_tax.py` | graph replay of the DX node count vs eager (why the runtime is not graph-captured) | 1 |

Run: `source env/env.sh && $SW_MPIRUN python3 probes/perm_bw.py` (`-np 2` for the copy probes: `SW_NP=2`).
