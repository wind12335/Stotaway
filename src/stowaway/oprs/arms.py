"""Arm kinds of the end-to-end A/B and the plan file of each inside a plan set (CPU only).

A plan set (tools/oprs_plans.py) is one deployment: one external backend B and the plans of its arms,
  oprs_<B>.json       the predicted plan: per call site serial / DEO / B
  oprs_deo_only.json  two-way plan (serial / DEO): the deployment for when no external backend is runnable
  always_deo.json     constant choice DEO
  always_<B>.json     constant choice B
  null.json           Null: the contexts of oprs_<B>.json are allocated, nothing is replaced
The two frameworks name the same arms with the words of the handoff doc:
  Megatron   serial | null | always_deo | always_<B> | oprs | oprs_deo_only     (e2e/megatron/run_oprs.sh)
  vLLM       stock  | sp_rccl | all_deo | all_<B> | oprs | oprs_deo_only        (e2e/vllm/run_oprs.sh)
In vLLM the SP form with every site on RCCL IS the Null plan (the hook, the heap and the contexts
are there, nothing is replaced), so `sp_rccl` is the Null arm and doubles as the "SP-RCCL" reference.
"""
from .platform import BACKENDS, DEO

BACKEND_ALIASES = {'all_deo': 'always_deo'}
ALIASES = {'sp_rccl': 'null', **BACKEND_ALIASES}
PLAN_FILE = {'null': 'null.json', 'always_deo': 'always_deo.json', 'oprs_deo_only': 'oprs_deo_only.json'}


def plan_file(kind, backend):
    """File name of the plan of arm `kind` inside a plan set whose external backend is `backend`."""
    kind = ALIASES.get(kind, kind)
    if kind == 'oprs':
        return f'oprs_{backend}.json'
    if kind == 'always_deo' or kind.startswith('always_'):
        if kind.removeprefix('always_') not in (DEO,) + BACKENDS:      # a constant arm of an integrated method
            raise ValueError(f'unknown arm kind {kind!r}: always_deo or always_<backend> with a backend in {BACKENDS}')
        return f'{kind}.json'
    if kind in PLAN_FILE:
        return PLAN_FILE[kind]
    raise ValueError(f'unknown arm kind {kind!r}: stock | sp_rccl | all_deo | all_<backend> | oprs | '
                     f'oprs_deo_only | name:<plan.json>')


def resolve_arm(spec, plans, backend):
    """(name, kind, plan file | None) of one vLLM arm spec `name:kind` or `name:<plan.json>`."""
    name, _, kind = spec.partition(':')
    kind = kind or name
    if kind == 'stock':
        return name, 'stock', None
    if kind.endswith('.json'):
        return name, 'plan', kind
    return name, kind, f'{plans.rstrip("/")}/{plan_file(kind, backend)}'
