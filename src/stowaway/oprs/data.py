"""Unified operator-sample records: the legacy table and the new-protocol JSONL (CPU only).

Two data classes exist and are NEVER mixed inside one fit:
  legacy  planner/data/cells.json, the historical BW1000 operator table (99 points, DEO only,
          bench/{ag_gemm,gemm_rs}.py timing protocol, warm-up and environment knobs unrecorded);
          its Route carries timing_mode 'b2b16_perop_events', warmup None, config 'legacy_env'.
  new     raw/operators/*.jsonl written by bench/oprs_matrix.py (per-point records with raw group
          samples, run order, platform, commit); Route timing 'b2b16', warmup 4.

Each record has `kind`: 'operator' = a point of the model-family operator table (training /
holdout material); 'site' = a workload call-site shape (an independent target measurement, never
training material).
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .platform import DEO, legacy_deo_route
from .predictor import Route, Sample

SAMPLE_SCHEMA = 'stowaway-oprs-operator-sample-v1'
LEGACY, NEW = 'legacy', 'b2b16'


@dataclass(frozen=True)
class OperatorRecord:
    family: str                    # model family (operator table) or workload name (site)
    sample: Sample
    op: str                        # route direction: 'ag' | 'rs'
    method: str                    # 'deo' | backend name
    model: str                     # model key (operator) or workload name (site)
    kind: str                      # 'operator' | 'site'
    protocol: str                  # LEGACY | NEW
    origin: str                    # source file + position
    table_op: Optional[str] = None  # 'ag_gu' | 'ag_up' | 'rs'
    site: Optional[str] = None     # 'qkv' | 'proj' | 'fc1' | 'fc2' for kind == 'site'
    anomaly: Optional[str] = None
    correctness_pass: Optional[bool] = None   # None: not recorded (legacy)
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.sample.shape

    @property
    def identity(self) -> tuple:
        base = (self.model, self.op, *self.sample.shape)
        return base + (self.site,) if self.site is not None else base

    @property
    def log_speedup(self) -> float:
        import math
        return math.log(self.sample.stock_ms / self.sample.candidate_ms)


def load_legacy_cells(path, *, tp: int, topology: str, versions: Tuple[Tuple[str, str], ...],
                      dtype: str = 'bfloat16') -> list:
    """The historical operator table as DEO records (b2b medians `stock_b2b` / `dx_b2b`)."""
    path = Path(path)
    cells = json.loads(path.read_text())
    out = []
    for index, cell in enumerate(cells):
        op = 'ag' if cell['op'].startswith('ag') else 'rs'
        sample = Sample(legacy_deo_route(op, tp=tp, topology=topology, versions=versions, dtype=dtype),
                        (cell['M'], cell['N'], cell['K']), float(cell['stock_b2b']), float(cell['dx_b2b']))
        out.append(OperatorRecord(family=cell['model'], sample=sample, op=op, method=DEO, model=cell['model'],
                                  kind='operator', protocol=LEGACY, origin=f'{path}#[{index}]',
                                  table_op=cell['op'],
                                  anomaly=(f"legacy-anomaly:{cell['model']}/{cell['op']}/M{cell['M']}"
                                           if cell.get('anomaly') else None),
                                  raw=dict(cell)))
    return out


def record_from_dict(item: dict, origin: str) -> OperatorRecord:
    """One JSONL line (bench/oprs_matrix.py record) -> OperatorRecord. Raises on malformed rows."""
    if item.get('schema') != SAMPLE_SCHEMA:
        raise ValueError(f'{origin}: unsupported sample schema {item.get("schema")!r}')
    route = Route(**{**item['route'], 'versions': tuple(tuple(v) for v in item['route']['versions'])})
    shape = (item['M'], item['N'], item['K'])
    if route.direction != item['op'] or route.method != item['candidate']:
        raise ValueError(f'{origin}: route does not match op/candidate')
    correctness = item.get('correctness') or {}
    passed = correctness.get('pass') if correctness.get('checked') else None
    return OperatorRecord(family=item['family'],
                          sample=Sample(route, shape, float(item['serial_reference_median_ms']),
                                        float(item['candidate_median_ms'])),
                          op=item['op'], method=item['candidate'], model=item['model'], kind=item['kind'],
                          protocol=NEW, origin=origin, table_op=item.get('table_op'), site=item.get('site'),
                          anomaly=item.get('anomaly'), correctness_pass=passed, raw=dict(item))


def iter_sample_rows(paths: Iterable) -> Iterable[tuple]:
    """(line dict, origin) for every JSONL line; a truncated last line is reported, not parsed."""
    for path in paths:
        with Path(path).open() as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line), f'{path}:{number}'
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{number}: unreadable line ({exc}); a crashed writer '
                                     f'leaves a partial last line -- move it aside by hand') from None


def load_samples(paths: Iterable) -> Tuple[list, list]:
    """(records, failures): new-protocol records with timing, and the error rows the bench wrote
    for points it could not run (kept: failures are data too)."""
    records, failures = [], []
    for item, origin in iter_sample_rows(paths):
        if item.get('status', 'ok') != 'ok':
            failures.append(dict(item, origin=origin))
            continue
        records.append(record_from_dict(item, origin))
    seen = {}
    for record in records:
        key = (record.identity, record.method, record.raw.get('data_class'))
        if key in seen:
            raise ValueError(f'duplicate measurement {key}: {seen[key]} and {record.origin} -- '
                             f'keep runs in separate directories and select with an explicit path list')
        seen[key] = record.origin
    return records, failures


def usable(records: Iterable[OperatorRecord], *, exclude_anomaly: bool = False) -> list:
    """Operator-table records a fit may use: not a failed correctness check; anomalies only when asked."""
    return [r for r in records if r.kind == 'operator' and r.correctness_pass is not False
            and not (exclude_anomaly and r.anomaly)]


def grouped(records: Iterable[OperatorRecord], method: str, op: str, *, exclude_anomaly: bool = False) -> list:
    """GroupedSample rows of one (method, op) route. `op == 'ag'` pools ag_gu and ag_up."""
    from .route_policy import GroupedSample
    return [GroupedSample(r.family, r.sample)
            for r in usable(records, exclude_anomaly=exclude_anomaly) if r.method == method and r.op == op]


def op_of(table_op: str) -> str:
    if table_op.startswith('ag'):
        return 'ag'
    if table_op == 'rs':
        return 'rs'
    raise ValueError(f'unknown operator {table_op}')
