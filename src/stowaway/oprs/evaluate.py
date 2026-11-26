"""Family-holdout evaluation of the BW route models on one aligned sample set (CPU only).

Port of nvidia/planner/original_a800/evaluate_tp8_auto_threeway.py, generalized to any candidate set:
DEO alone = serial/DEO two-way; DEO + one measured backend B = three-way on the points where every
candidate was measured. For every held-out model family ALL base fits, residual supports and route
policies use only the other families.

The label follows the data class and must travel with the number: on the historical table this is
`historical_replay_legacy` (rows not sampled under the new protocol), on new-protocol records it is
`family_holdout_b2b16`. Neither is an in-framework result, and the two are never averaged together.

Policies compared per point (serial = 1.0, ties resolve to serial):
  always_<m>    fixed candidate m
  base_ridge    ridge only (log M, log N, log K) per route, no local residual
  route_policy  RoutePolicy: per-route nested threshold chosen on the training families
  oprs          ridge + family-holdout local residual (the method that produces plans)
  oracle        the candidate that was actually fastest (upper reference, not a predictor)
"""
import argparse
import csv
import math
from pathlib import Path
import statistics

from .export_plan import git_commit, load_records, write_json_exclusive
from .platform import DEO
from .predictor import OPS, SERIAL, RouteRegressor
from .route_local_residual import RouteLocalResidual
from .route_policy import RoutePolicy
from .data import grouped, usable

CLASSIFICATION = {'legacy': 'historical_replay_legacy', 'b2b16': 'family_holdout_b2b16'}
LEARNED = ('base_ridge', 'route_policy', 'oprs')
DESCRIPTION = {'always': 'fixed candidate (serial reference = 1.0)', 'base_ridge': 'ridge on (log M, log N, log K) only',
               'route_policy': 'RoutePolicy: per-route threshold chosen by nested leave-one-family-out on the training families',
               'oprs': 'ridge + family-holdout local residual (plan producer)', 'oracle': 'fastest measured candidate'}


def _pick(scores):
    """Largest score; the first key (serial) wins ties."""
    best = None
    for key, value in scores.items():
        if best is None or value > scores[best]:
            best = key
    return best


def write_csv_exclusive(path, rows: list) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _gmean(values):
    return math.exp(statistics.fmean(math.log(v) for v in values))


def evaluate(records, methods, *, exclude_anomaly=False, with_route_policy=True, sources=(), limitation='') -> dict:
    protocols = sorted({r.protocol for r in records})
    if len(protocols) != 1:
        raise ValueError(f'one data class per evaluation, got {protocols}')
    protocol, methods = protocols[0], tuple(methods)
    total = len(records)
    anomaly_ids = sorted({r.anomaly for r in records if r.anomaly})
    records = usable(records, exclude_anomaly=exclude_anomaly)
    present = {}
    for r in records:
        present.setdefault((r.model, r.op, *r.shape), {})[r.method] = r
    points = sorted(k for k, v in present.items() if all(m in v for m in methods))
    # the holdout unit is the model FAMILY of a point, the same key the training rows are filtered by.
    # (The first port held out by model name but filtered by family: identical keys on the historical
    # table, so nothing measured with it moved, but any data where they differ leaked the held-out
    # family into its own fit.)
    family_of = {}
    for point in points:
        families = {present[point][m].family for m in methods}
        if len(families) != 1:
            raise ValueError(f'point {point}: its records disagree on the model family {sorted(families)}')
        family_of[point] = families.pop()
    learned = [p for p in LEARNED if p != 'route_policy' or with_route_policy]
    rows = []
    for op in OPS:
        op_points = [p for p in points if p[1] == op]
        for held in sorted({family_of[p] for p in op_points}):
            models = {}
            for method in methods:
                train = [r for r in grouped(records, method, op) if r.family != held]
                if len({r.family for r in train}) < 4:
                    raise ValueError(f'{method}/{op}: four training families required to hold out {held}')
                route = train[0].sample.route
                models[method] = {'oprs': RouteLocalResidual(route).fit(train),
                                  'base_ridge': RouteRegressor(route).fit([r.sample for r in train]),
                                  'route_policy': RoutePolicy(route).fit(train) if with_route_policy else None}
            for point in op_points:
                if family_of[point] != held:
                    continue
                shape, actual = point[2:], {SERIAL: 1.0}
                predicted = {policy: {SERIAL: 0.0} for policy in learned}
                enabled = {}
                for method in methods:
                    sample = present[point][method].sample
                    actual[method] = sample.stock_ms / sample.candidate_ms
                    model, route = models[method], models[method]['oprs'].route
                    predicted['base_ridge'][method] = model['base_ridge'].predict(route, shape)
                    predicted['oprs'][method] = model['oprs'].predict(route, shape)['predicted_log_speedup']
                    if with_route_policy:
                        decision = model['route_policy'].predict(route, shape)
                        enabled[method] = decision['choice'] == method
                        predicted['route_policy'][method] = decision['predicted_log_speedup']
                oracle = _pick(actual)
                offered = {m: predicted['route_policy'][m] for m in methods if enabled.get(m)}
                choices = {**{f'always_{m}': m for m in methods}, 'oracle': oracle,
                           'base_ridge': _pick(predicted['base_ridge']), 'oprs': _pick(predicted['oprs'])}
                if with_route_policy:
                    choices['route_policy'] = _pick({SERIAL: 0.0, **offered}) if offered else SERIAL
                row = {'model': point[0], 'op': op, 'table_op': present[point][methods[0]].table_op,
                       'M': shape[0], 'N': shape[1], 'K': shape[2], 'predicted_log_speedup': predicted,
                       'actual_speedup': actual, 'route_policy_enabled': enabled, 'choices': choices,
                       'oracle_choice': oracle, 'selected_speedup': {}, 'correct': {}, 'below_stock': {}, 'regret': {}}
                for policy, choice in choices.items():
                    speedup = actual[choice]
                    row['selected_speedup'][policy] = speedup
                    row['correct'][policy] = choice == oracle
                    row['below_stock'][policy] = speedup < 1.0
                    row['regret'][policy] = math.log(actual[oracle] / speedup)
                rows.append(row)
    summary = {policy: {} for policy in rows[0]['choices']} if rows else {}
    for policy in summary:
        for scope in ('ag', 'rs', 'all'):
            items = [r for r in rows if scope == 'all' or r['op'] == scope]
            summary[policy][scope] = {'count': len(items), 'correct': sum(r['correct'][policy] for r in items),
                                      'selected_gmean_speedup': _gmean([r['selected_speedup'][policy] for r in items]),
                                      'selected_below_stock': sum(r['below_stock'][policy] for r in items),
                                      'regret_gmean_ratio': _gmean([math.exp(r['regret'][policy]) for r in items]),
                                      'oracle_gmean_speedup': _gmean([r['actual_speedup'][r['oracle_choice']] for r in items])} \
                if items else None
            if items:
                summary[policy][scope]['accuracy'] = summary[policy][scope]['correct'] / len(items)
    return {'schema': 'stowaway-oprs-family-holdout-v1', 'classification': CLASSIFICATION[protocol],
            'data_class': protocol, 'methods': methods, 'exclude_anomaly': exclude_anomaly,
            'denominators': {'operator_records': total, 'usable_records': len(records),
                             'points_with_all_candidates': len(points),
                             'points_missing_a_candidate': sorted(str(k) for k in present if k not in points),
                             'anomalies_included': [] if exclude_anomaly else anomaly_ids,
                             'anomalies_excluded': anomaly_ids if exclude_anomaly else []},
            'policies': DESCRIPTION, 'summary': summary, 'rows': rows, 'sources': list(sources),
            'limitation': limitation, **git_commit()}


def choice_rows(result) -> list:
    """One flat row per (point, policy) for tables/predictor_choices.csv."""
    flat = []
    for r in result['rows']:
        for policy, choice in r['choices'].items():
            predicted = r['predicted_log_speedup'].get(policy, {})     # empty for fixed / oracle policies
            flat.append({'data_class': result['data_class'], 'classification': result['classification'],
                         'exclude_anomaly': result['exclude_anomaly'], 'model': r['model'], 'op': r['op'],
                         'table_op': r['table_op'], 'M': r['M'], 'N': r['N'], 'K': r['K'], 'policy': policy,
                         'choice': choice, 'oracle_choice': r['oracle_choice'], 'correct': r['correct'][policy],
                         'selected_speedup': r['selected_speedup'][policy], 'below_stock': r['below_stock'][policy],
                         'regret': r['regret'][policy],
                         **{f'predicted_log_speedup_{m}': predicted.get(m) for m in result['methods']},
                         **{f'actual_speedup_{m}': r['actual_speedup'][m] for m in result['methods']}})
    return flat


def main():
    parser = argparse.ArgumentParser(description='Family-holdout evaluation of BW route models on one data class')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--legacy')
    source.add_argument('--data', nargs='+')
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--topology', required=True)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--methods', default=DEO, help='comma list, e.g. deo,async_tp (all must be measured)')
    parser.add_argument('--exclude-anomaly', action='store_true')
    parser.add_argument('--no-route-policy', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    records, sources, data_class, limitation = load_records(args)
    result = evaluate(records, [m for m in args.methods.split(',') if m], exclude_anomaly=args.exclude_anomaly,
                      with_route_policy=not args.no_route_policy, sources=sources, limitation=limitation)
    write_json_exclusive(args.output, result)
    if args.csv:
        write_csv_exclusive(args.csv, choice_rows(result))
    print(f"{result['classification']} data_class={data_class} methods={result['methods']} "
          f"exclude_anomaly={args.exclude_anomaly} points={result['denominators']['points_with_all_candidates']}")
    for policy, scopes in result['summary'].items():
        print(f"  {policy:<12} " + '  '.join(f"{s}: n={v['count']} acc={v['accuracy']:.3f} gmean={v['selected_gmean_speedup']:.3f} "
                                             f"below={v['selected_below_stock']}" for s, v in scopes.items() if v))


if __name__ == '__main__':
    main()
