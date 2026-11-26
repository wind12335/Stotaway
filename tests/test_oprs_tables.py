"""tools/oprs_tables.py on synthetic logs: the two end-to-end tables, the natural-text gate and the
derived tables."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from oprs_synthetic import ASYNC_TP, DEO, FAMILIES, TRITON_DIST, every_record, records, sample_line, write_samples
from stowaway.oprs.export_plan import main as export_main

LEGACY_TABLE = Path('planner/data/cells.json')
ITER = (' [2026-11-26 21:00:00.000000] iteration        {i}/       8 | consumed samples:            {n} | '
        'elapsed time per iteration (ms): {ms} | throughput per GPU (TFLOP/s/GPU): 36.4 | learning rate: 3.0E-05 | '
        'global batch size:     2 | lm loss: {loss} | loss scale: 1.0 | grad norm: 4.132 | '
        'number of skipped iterations:   0 | number of nan iterations:   0 |')
SITES = ('qkv', 'proj', 'fc1', 'fc2')


@pytest.fixture(scope='module')
def tables(scripts):
    return scripts('oprs_tables', 'tools', 'oprs_tables.py')


def call(module, *argv):
    """Run one tables command on an argv list; return what it printed."""
    out = io.StringIO()
    with redirect_stdout(out):
        try:
            module.main(list(argv))
        except SystemExit as exc:                                       # main() leaves with 0 on success
            if exc.code not in (0, None):
                raise
    return out.getvalue()


# ---- Megatron training ------------------------------------------------------------------------
def training_run(root):
    """A Megatron run as e2e/megatron/run_oprs.sh leaves it: six arms, every exit-code case once."""
    plans = root / 'plans'
    plans.mkdir(parents=True)
    entries = lambda choice: {'entries': [{'site': s, 'choice': choice} for s in SITES]}     # noqa: E731
    for name, choice in (('oprs_async_tp', 'serial'), ('always_deo', 'deo'), ('always_triton_dist', 'triton_dist'),
                         ('always_async_tp', 'async_tp')):
        (plans / f'{name}.json').write_text(json.dumps(entries(choice)))
    driver = ['[e2e] ROOT=/x plans=/p heap=1']

    def arm(name, kind, plan, steps, hits=(), fallbacks=(), code=0, mpirun_rc=None, py_rc=None):
        folder = root / name
        folder.mkdir()
        log = []
        if plan:
            log.append(f"[oprs] OPRS_ADAPTER armed plan={plan} workload=w methods=['deo']")
        for i, ms in enumerate(steps, 1):
            log.append(ITER.format(i=i, n=2 * i, ms=ms, loss='1.182455E+01'))
        if plan:
            log.append('[oprs] OPRS_READY rank=0 tp=1 workload=w plan=abcdef123456 backend=async_tp entries=4 '
                       "deo_bwd=ag contexts=1 {'deo/rs': 1} heap_need_mb=16 heap_mb=1084 load_ms=140.0 prepare_ms=16.5")
            for site, choice, n in hits:
                log.append(f'[oprs] OPRS_HIT rank=0 site={site} op=rs M=4096 N=1024 K=2048 choice={choice} count={n}')
            for reason, n in fallbacks:
                log.append(f'[oprs] OPRS_FALLBACK rank=0 reason={reason} count={n}')
            log.append('[oprs] OPRS_ADAPTER rank=0 row=128 column=0 row_skip=128 column_skip=272 early=0 '
                       "sites={'fc1': 4, 'fc2': 4, 'proj': 4, 'qkv': 4} conflicts=0 untagged=1 group_mismatch=0")
        log.append(' validation loss at iteration 8 on validation set | lm loss value: 1.174535E+01 | lm loss PPL: 1.2E+05 | ')
        log.append(' validation loss at iteration 8 on test set | lm loss value: 1.161879E+01 | lm loss PPL: 1.1E+05 | ')
        log.append(' [Rank 0] (after 8 iterations) memory (MB) | allocated: 15815.44 | max allocated: 16000.50 |')
        if code is not None:
            log.append(f'EXIT {code}')
        (folder / 'out_r0.log').write_text('\n'.join(log) + '\n')
        driver.append(f'[e2e] {name} kind={kind} plan={plan or "none"} model=custom tp=1 MBS=1 NL=4 start 21:00:00')
        tail = f' py_rc={py_rc}' if py_rc is not None else ''
        driver.append(f'[e2e] {name} kind={kind} rc={mpirun_rc if mpirun_rc is not None else code}{tail} '
                      f'iters={len(steps)} med(it3+)=none hit_lines_r0={len(hits)} fallback_lines_r0={len(fallbacks)} '
                      f'end 21:00:53 wall=53s')

    eight = [400.0, 401.0, 397.0, 396.0, 398.0, 399.0, 397.0, 398.0]
    arm('s1', 'serial', None, eight)
    arm('n1', 'null', plans / 'null.json', [406.0] * 8, fallbacks=[('plan', 512)], mpirun_rc=1)    # EXIT line only
    arm('o1', 'oprs', plans / 'oprs_async_tp.json', [410.0, 420.0, 418.0, 417.0, 419.0, 418.0, 417.0, 418.0],
        hits=[('fc2', 'deo', 128)], fallbacks=[('plan', 384)], mpirun_rc=1, py_rc=0)
    arm('d1', 'always_deo', plans / 'always_deo.json', [433.0] * 8,
        hits=[(s, 'deo', 128) for s in SITES], mpirun_rc=1, py_rc=0)
    arm('k1', 'always_async_tp', plans / 'always_async_tp.json', [430.0, 431.0, 436.0], code=None, mpirun_rc=124,
        py_rc='none')                                                                          # killed by timeout
    arm('b1', 'always_triton_dist', plans / 'always_triton_dist.json', [408.0] * 8, mpirun_rc=1, py_rc=0)   # no hit
    (root / 'driver.log').write_text('\n'.join(driver) + '\nE2E DONE\n')


def test_training_table_reads_the_real_exit_code_and_the_hits(tmp_path, tables):
    run_dir = tmp_path / 'custom_tp1_mbs1_1126_2100'
    training_run(run_dir)
    text = call(tables, 'training', str(run_dir))
    assert 'TP=1' in text
    for piece in [
            '| s1 | serial | - | 0 | 8 | 397.5 | 1.182455E+01 | 1.161879E+01 | 16000.5 | - | - | ok |',
            '| n1 | null | null.json | 0 | 8 | 406.0 | 1.182455E+01 | 1.161879E+01 | 16000.5 | - | plan:512 | ok |',
            '| o1 | oprs | oprs_async_tp.json | 0 | 8 | 418.0 | 1.182455E+01 | 1.161879E+01 | 16000.5 | '
            'fc2=deo:128 | plan:384 | ok |',
            '| d1 | always_deo | always_deo.json | 0 | 8 | 433.0 | 1.182455E+01 | 1.161879E+01 | 16000.5 | '
            'fc1=deo:128 fc2=deo:128 proj=deo:128 qkv=deo:128 | - | ok |',
            '| k1 | always_async_tp | always_async_tp.json | 124 | 3 | 436.0 ✗ | 1.182455E+01 | 1.161879E+01 | 16000.5 | '
            '- | - | failed |',
            '| b1 | always_triton_dist | always_triton_dist.json | 0 | 8 | 408.0 ✗ | 1.182455E+01 | 1.161879E+01 | '
            '16000.5 | - | - | NO-HITS |',
            '| null | [0.9791] | 0.9791 | 0.9791..0.9791 |',              # 397.5 / 406.0
            '| oprs | [0.951] | 0.9510 | 0.9510..0.9510 |',               # 397.5 / 418.0
            '| always_deo | [0.918] | 0.9180 | 0.9180..0.9180 |',        # 397.5 / 433.0
            'OPRS 对 always_deo（always_deo 步长 / oprs 步长，> 1 = OPRS 更快）：[1.0359] → 中位 1.0359']:
        assert piece in text
    assert 'always_async_tp |' not in text.split('同批次配对比值')[1]     # its only run was killed: no ratio
    assert 'always_triton_dist |' not in text.split('同批次配对比值')[1]  # NO-HITS: never a slow arm
    rows = list(csv.DictReader((run_dir / 'e2e_training.csv').open()))
    assert [(r['arm'], r['rc'], r['mpirun_rc'], r['ranks_with_hits'], r['plan_on_ranks']) for r in rows] == [
        ('s1', '0', '0', '0/1', '0/1'), ('n1', '0', '1', '0/1', '1/1'), ('o1', '0', '1', '1/1', '1/1'),
        ('d1', '0', '1', '1/1', '1/1'), ('k1', '124', '124', '0/1', '1/1'), ('b1', '0', '1', '0/1', '1/1')]
    assert json.loads(rows[2]['hits']) == {'fc2=deo': 128} and json.loads(rows[1]['fallbacks']) == {'plan': 512}
    assert json.loads(rows[0]['step_ms_all']) == [400.0, 401.0, 397.0, 396.0, 398.0, 399.0, 397.0, 398.0]
    assert rows[0]['val_loss'] == '1.161879E+01' and rows[4]['status'] == 'failed'
    call(tables, 'training', str(run_dir))                               # the summary of one run is rebuilt in place
    assert len(list(csv.DictReader((run_dir / 'e2e_training.csv').open()))) == 6
    (tmp_path / 'empty_tp1').mkdir()
    with pytest.raises(SystemExit, match='no driver.log'):             # not a run directory
        call(tables, 'training', str(tmp_path / 'empty_tp1'))
    (tmp_path / 'empty_tp1' / 'driver.log').write_text('[e2e] ROOT=/x heap=1\n')
    with pytest.raises(SystemExit, match='no arm'):
        call(tables, 'training', str(tmp_path / 'empty_tp1'))
    (tmp_path / 'unnamed').mkdir()
    with pytest.raises(SystemExit, match='give --tp'):                 # the name carries no card count
        call(tables, 'training', str(tmp_path / 'unnamed'))
    (tmp_path / 'unnamed' / 'driver.log').write_text('[e2e] ROOT=/x heap=1\n')
    with pytest.raises(SystemExit, match='no arm'):                     # --tp alone is not enough either
        call(tables, 'training', str(tmp_path / 'unnamed'), '--tp', '1')


def test_after_report_increments_are_summed(tmp_path, tables):
    """The framework evaluates after train() returned; those calls are printed as increments and belong
    to the count of the whole process."""
    run_dir = tmp_path / 'r_tp1'
    training_run(run_dir)
    log = (run_dir / 'o1' / 'out_r0.log').read_text().replace('EXIT 0', (
        '[oprs] OPRS_HIT rank=0 site=fc2 op=rs M=4096 N=1024 K=2048 choice=deo count=32 after_report\n'
        '[oprs] OPRS_FALLBACK rank=0 reason=plan count=96 after_report\nEXIT 0'))
    (run_dir / 'o1' / 'out_r0.log').write_text(log)
    o1 = next(r for r in tables.training_rows(run_dir, 1) if r['arm'] == 'o1')
    assert o1['hits'] == {'fc2=deo': 160} and o1['fallbacks'] == {'plan': 480}


# ---- vLLM inference ---------------------------------------------------------------------------
COLUMNS = ['ts', 'arm', 'kind', 'plan_digest', 'backend', 'M', 'tp', 'cell', 'rep', 'nreq', 'isl', 'wall_s',
           'tok_s', 'prefilled_tokens', 'cached_tokens', 'sp_forwards', 'stock_forwards', 'hits', 'fallbacks',
           'hit_detail', 'load_ms', 'prepare_ms', 'nonserial_entries', 'status', 'note']
# arm: (kind, plan digest, ttft walls, ttft2 walls, thru walls, hits, fallbacks); walls None = stopped after init
SPEC = {'s1': ('stock', '', [0.005, 0.006], [0.009, 0.010], [0.070], 0, 0),
        'r1': ('sp_rccl', '63c452f2af46', [0.009, 0.008], [0.015, 0.014], [0.115], 0, 400),
        'o1': ('oprs', 'cd100a3b4e71', [0.017, 0.016], [0.020, 0.019], [0.165], 300, 100),
        's2': ('stock', '', [0.005, 0.006], [0.009, 0.010], [0.070], 0, 0),
        'r2': ('sp_rccl', '63c452f2af46', [0.008, 0.009], [0.014, 0.015], [0.115], 0, 400),
        'o2': ('oprs', 'cd100a3b4e71', [0.018, 0.016], [0.019, 0.020], [0.156], 300, 100),
        'x1': ('always_deo', 'dce7031f0e8e', None, None, None, None, None)}


def vllm_run(root):
    root.mkdir(parents=True)
    rows = []
    for arm, (kind, plan, ttft, ttft2, thru, hits, fallbacks) in SPEC.items():
        base = dict(ts='2026-11-26 21:14:41', arm=arm, kind=kind, plan_digest=plan, backend='', M=2048, tp=1,
                    nreq='', isl='', tok_s='', prefilled_tokens='', cached_tokens='', sp_forwards=0,
                    stock_forwards=0, hits=0, fallbacks=0, hit_detail='', load_ms='', prepare_ms='',
                    nonserial_entries='', status='', note='')
        final = dict(backend='triton_dist', sp_forwards=25, stock_forwards=36, hits=hits, fallbacks=fallbacks,
                     hit_detail='fc2=deo:100' if hits else '', load_ms=182.5, prepare_ms=16.6,
                     nonserial_entries=3 if hits else 0, status='ok')
        rows.append(dict(base, cell='init', rep=0, wall_s=11.726))
        rows.append(dict(base, cell='warmup', rep=0, nreq=1, isl=2048, wall_s=0.0134, prefilled_tokens=2048,
                         cached_tokens=0))
        for cell, nreq, isl, walls in (('ttft', 1, 2048, ttft), ('ttft2', 1, 4096, ttft2), ('thru', 16, 2048, thru)):
            for rep, wall in enumerate(walls or []):
                rows.append(dict(base, cell=cell, rep=rep, nreq=nreq, isl=isl, wall_s=wall,
                                 prefilled_tokens=nreq * isl, cached_tokens=0,
                                 **({'status': 'ok'} if kind == 'stock' else final)))   # stock: no dispatcher figures
    with (root / 'voprs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    driver = ['[e2e] ROOT=/x model=/tmp/tiny tp=1 M=2048']
    for arm, (kind, *_rest) in SPEC.items():
        driver.append(f'[e2e] {arm} kind={kind} start 21:14:06')
        driver.append(f'[e2e] {arm} kind={kind} rc={0 if arm != "x1" else 1} end 21:14:59 wall=53s')
    driver += ['[e2e] q1 kind=all_deo start 21:15:00', '[e2e] q1 kind=all_deo rc=1 end 21:15:01 wall=1s', 'E2E DONE']
    (root / 'driver.log').write_text('\n'.join(driver) + '\n')


def test_inference_runs_group_rows_by_arm_run(tmp_path, tables):
    root = tmp_path / 'vllm'
    vllm_run(root)
    runs = tables.inference_runs(root, 2048)
    assert [(r['name'], r['base'], r['kind']) for r in runs] == [
        ('s1', 's', 'stock'), ('r1', 'r', 'sp_rccl'), ('o1', 'o', 'oprs'), ('s2', 's', 'stock'),
        ('r2', 'r', 'sp_rccl'), ('o2', 'o', 'oprs'), ('x1', 'x', 'always_deo'), ('q1', 'q1', 'all_deo')]
    assert set(runs[2]['cells']) == {'init', 'ttft', 'ttft2', 'thru'}          # the warm-up row is not a sample
    assert runs[0]['med']['ttft'] == pytest.approx(0.0055) and runs[2]['med']['thru'] == 0.165
    assert (runs[2]['last']['hits'], runs[2]['last']['hit_detail']) == ('300', 'fc2=deo:100')
    assert [r['status'] for r in runs] == ['ok'] * 6 + ['incomplete: stopped after init', 'FAILED rc=1 (no row written)']
    assert tables.inference_runs(tmp_path / 'nothing', 2048) == []


def test_inference_table_pairs_runs_by_round(tmp_path, tables):
    root = tmp_path / 'vllm'
    vllm_run(root)
    text = call(tables, 'inference', str(root), '2048')
    for piece in [
            '| s1 | stock | - | 11.726 | 0.005 | 0.009 | 0.070 | 2048 / 4096 / 32768 | 0 / 0 | 0 | 0 | - / - | ok |',
            '| r1 | sp_rccl | 63c452f2af46 | 11.726 | 0.009 | 0.014 | 0.115 | 2048 / 4096 / 32768 | 25 / 36 | 0 | 400 | 182.5 / 16.6 | ok |',
            '| o1 | oprs | cd100a3b4e71 | 11.726 | 0.017 | 0.019 | 0.165 | 2048 / 4096 / 32768 | 25 / 36 | 300 fc2=deo:100 | 100 | 182.5 / 16.6 | ok |',
            '| x1 | always_deo | dce7031f0e8e | 11.726 | — | — | — | - / - / - | 0 / 0 | 0 | 0 | - / - | incomplete: stopped after init |',
            '| q1 | all_deo | - | — | — | — | — | - / - / - | - / - | - | - | - / - | FAILED rc=1 (no row written) |',
            '| r（sp_rccl） | ttft | 0.647，0.647 → 0.647 [0.647..0.647] | — |',       # 0.0055 / 0.0085, twice
            '| o（oprs） | ttft | 0.333，0.324 → 0.328 [0.324..0.333] | 0.515，0.500 → 0.508 [0.500..0.515] |',
            '| s（stock） | thru | — | 1.643，1.643 → 1.643 [1.643..1.643] |']:                    # 0.070 / 0.115
        assert piece in text
    assert '## 自然文本输出检查' not in text and '数值一致性未测' in text      # no gate files here
    assert len(list(csv.DictReader((root / 'e2e_inference.csv').open()))) == 6 * 6 + 1      # warm-up rows are not samples
    with pytest.raises(SystemExit, match='no row'):
        call(tables, 'inference', str(tmp_path), '2048')


def write_gates(root):
    """gatetext files of the gate: stock, an arm equal to it and one that diverges on its first prompt.
    `top5[i]` = {generated token: logprob}; `prompt_lp` = logprob of each prompt token (None first)."""
    def prompt(ids, top, lp):
        return {'ids': ids, 'top5': top, 'prompt_lp': lp}
    stock = [prompt([10, 11, 12, 13], [{10: -1.0}, {11: -0.5}, {12: -2.0}, {13: -3.0}], [None, -1.0, -0.5, -2.0]),
             prompt([20, 21, 22, 23], [{20: -3.0}, {21: -1.0}, {22: -0.1}, {23: -2.5}], [None, -3.0, -1.0, -0.1])]
    other = [prompt([10, 11, 99, 98], [{10: -1.5}, {11: -0.25}, {99: -5.0}, {98: -6.0}], [None, -1.0, -0.4, -2.0]),
             prompt([20, 21, 22, 23], [{20: -3.0}, {21: -1.01}, {22: -0.1}, {23: -2.5}], [None, -3.0, -1.0, -0.1])]
    root.mkdir(parents=True, exist_ok=True)
    for name, kind, data in (('stock', 'stock', stock), ('r1', 'sp_rccl', stock), ('o1', 'oprs', other)):
        (root / f'gatetext_{name}_M2048.json').write_text(json.dumps(
            {'arm': name, 'kind': kind, 'M': 2048, 'np': 2, 'osl': 4, 'prompts': data}))
    return sorted(root.glob('gatetext_*.json'))


def test_natural_text_gate_table(tmp_path, tables):
    text = call_gate(tables, write_gates(tmp_path / 'gate'))
    assert '| r1 | 8/8 | 无 / 无 | 0.0000 | 0.0000 | 0.0000 / 0.00000 |' in text
    assert '| o1 | 6/8 | 2 / 无 | 0.5000 | 0.5000 | 0.1000 / 0.01667 |' in text
    lone = tmp_path / 'lone'
    lone.mkdir()
    (lone / 'gatetext_o1_M2048.json').write_text(json.dumps(
        {'arm': 'o1', 'kind': 'oprs', 'M': 2048, 'np': 1, 'osl': 4, 'prompts': []}))
    assert '没有 stock' in call_gate(tables, sorted(lone.glob('*.json')))


def call_gate(tables, paths):
    out = io.StringIO()
    with redirect_stdout(out):
        tables.gate_table(paths)
    return out.getvalue()


def test_gate_only_directory(tmp_path, tables):
    root = tmp_path / 'vllm_gate'
    write_gates(root)
    rows = []
    for arm, kind, hits in (('stock', 'stock', 0), ('o1', 'oprs', 72)):
        base = dict(ts='t', arm=arm, kind=kind, plan_digest='abc', backend='triton_dist', M=2048, tp=1, nreq=1,
                    isl=2048, tok_s='', cached_tokens=0, sp_forwards=6, stock_forwards=191, hits=hits,
                    fallbacks=24, hit_detail='fc2=deo:24' if hits else '', load_ms=186.5, prepare_ms=47.2,
                    nonserial_entries=3 if hits else '', status='ok', note='')
        rows.append(dict(base, cell='init', rep=0, wall_s=9.0, nreq='', isl='', prefilled_tokens=''))
        for rep, wall in enumerate((0.28, 0.27)):
            rows.append(dict(base, cell='gate_text', rep=rep, wall_s=wall, prefilled_tokens=2048))
    with (root / 'voprs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    text = call(tables, 'inference', str(root), '2048')
    assert '只跑了 VS_GATE=1' in text
    assert '| o1 | oprs | abc | 0.275 | 2048 | 6 / 191 | 72 fc2=deo:24 | 24 | 186.5 / 47.2 | ok |' in text
    assert '| o1 | 6/8 | 2 / 无 | 0.5000 | 0.5000 | 0.1000 / 0.01667 |' in text


# ---- derived tables ----------------------------------------------------------------------------
def test_operator_samples_table_keeps_every_row(tmp_path, tables):
    root = tmp_path / 'res'
    lines = [sample_line(r) for r in every_record(8)[:12]]
    timing = ('serial_samples_ms', 'candidate_samples_ms', 'serial_reference_median_ms', 'candidate_median_ms',
              'median_ms', 'speedup', 'lower_observed', 'upper_observed', 'serial_parts_ms', 'serial_part_samples_ms')
    unsupported = {k: v for k, v in dict(sample_line(every_record(8)[0]), candidate=ASYNC_TP, status='unsupported',
                                         support={'status': 'unsupported', 'evidence': 'needs M % 16'}).items()
                   if k not in timing}                                   # what the bench writes: no timing at all
    crashed = {k: v for k, v in dict(sample_line(every_record(8)[1]), candidate=ASYNC_TP, status='error',
                                      error='boom').items() if k not in timing}   # a crash leaves no timing either
    lines += [unsupported, crashed]
    folder = root / 'raw' / 'operators' / 'batch_a'
    folder.mkdir(parents=True)
    path = folder / 'samples_tp8_bfloat16.jsonl'
    path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n{"kind": "oper')   # a crashed writer's tail
    assert len(tables.operator_rows([path], 8)) == 14                    # the partial last line is skipped, warned
    assert tables.operator_rows([path], 1) == []                         # another card count
    text = call(tables, 'operator', str(root), '--tp', '8')
    assert '14 rows from 1 file(s)' in text                                   # every row kept, with its status
    assert '  operator  async_tp     unsupported  1' in text and '  operator  async_tp     error        1' in text
    table = list(csv.DictReader((root / 'tables' / 'operator_samples.csv').open()))
    bad = next(r for r in table if r['status'] == 'error')
    assert (bad['error'], bad['speedup'], bad['candidate_median_ms']) == ('boom', '', '')     # never a 0
    assert next(r for r in table if r['status'] == 'unsupported')['support_status'] == 'unsupported'
    good = next(r for r in table if r['candidate'] == DEO)
    assert (good['batch'], good['status'], good['correctness_pass'], good['support_status']) == (
        'batch_a', 'ok', 'True', 'supported') and float(good['speedup']) > 1.0
    assert (good['serial_reference_median_ms'], good['serial_gemm_only_ms'], good['topology']) == ('1.0', '0.6', 'BW8')
    with pytest.raises(SystemExit, match='move it aside'):
        call(tables, 'operator', str(root), '--tp', '8', '--data', str(path))
    with pytest.raises(SystemExit, match='no sample file'):
        call(tables, 'operator', str(tmp_path / 'nowhere'), '--tp', '8')


def test_choices_table_keeps_the_classes_apart(tmp_path, tables):
    if not LEGACY_TABLE.exists():                                       # pragma: no cover
        pytest.skip('planner/data/cells.json not available')
    root = tmp_path / 'res'
    data = write_samples(every_record(8), tmp_path / 'samples_tp8_bfloat16.jsonl')
    text = call(tables, 'choices', str(root), '--tp', '8', '--data', data, '--legacy', str(LEGACY_TABLE))
    assert '| family_holdout_b2b16 | deo | oprs | all | 99 | 1.000 |' in text
    assert '| deo+async_tp | oprs | all | 99 | 1.000 |' in text and '| deo+triton_dist | oprs | all | 99 | 1.000 |' in text
    assert '| historical_replay_legacy | deo | oprs | all | 99 | 0.848 |' in text       # 84 of 99, labelled
    flat = list(csv.DictReader((root / 'tables' / 'predictor_choices.csv').open()))
    summary = list(csv.DictReader((root / 'tables' / 'predictor_summary.csv').open()))
    assert {r['data_class'] for r in summary} == {'b2b16', 'legacy'}
    assert {r['classification'] for r in flat} == {'family_holdout_b2b16', 'historical_replay_legacy'}
    assert {r['candidates'] for r in summary} == {'deo', 'deo+async_tp', 'deo+triton_dist'}
    assert all(r['candidates'] == 'deo' and r['classification'] == 'historical_replay_legacy'
               for r in summary if r['data_class'] == 'legacy')                       # the old table has DEO only
    # one flat row per (point, policy, candidate set): 5 policies for a two-way, 6 for a three-way set
    assert len(flat) == 99 * 5 * 2 + 99 * 6 * 2 and {r['policy'] for r in flat} == {
        'always_deo', f'always_{ASYNC_TP}', 'always_triton_dist', 'base_ridge', 'route_policy', 'oprs', 'oracle'}
    held = json.loads((root / 'tables' / 'predictor_family_holdout.json').read_text())
    assert set(held) == {'legacy_deo', 'b2b16_deo', 'b2b16_deo+async_tp', 'b2b16_deo+triton_dist'}
    assert held['legacy_deo']['denominators']['operator_records'] == 99
    with pytest.raises(SystemExit, match='move it aside'):
        call(tables, 'choices', str(root), '--tp', '8', '--data', data)
    small = write_samples(records(DEO, 'rs', families=FAMILIES[:4]), tmp_path / 'small.jsonl')
    with pytest.raises(SystemExit, match='nothing could be evaluated'):             # too few families to hold out
        call(tables, 'choices', str(tmp_path / 'res2'), '--tp', '8', '--data', small)


def test_sites_table(tmp_path, tables):
    root = tmp_path / 'res'
    data = write_samples(every_record(8), tmp_path / 'samples_tp8_bfloat16.jsonl')
    out = io.StringIO()
    with redirect_stdout(out):
        export_main(['--workload', 'bw27b_train', '--tp', '8', '--backend', TRITON_DIST, '--data', data,
                     '--output', str(tmp_path / 'oprs_triton_dist.json')])
    assert 'out-of-range qkv deo: NK' in out.getvalue()
    (tmp_path / 'manifest.json').write_text(json.dumps({'plans': {'oprs': {'path': 'oprs_triton_dist.json'}}}))
    call(tables, 'sites', str(root), '--plans', str(tmp_path), '--data', data)
    rows = list(csv.DictReader((root / 'tables' / 'target_sites.csv').open()))
    assert [(r['site'], r['origin'], r['table_op'], r['op'], r['M'], r['choice']) for r in rows] == [
        ('qkv', 'qkv', 'ag_up', 'ag', '32768', DEO), ('proj', 'proj', 'rs', 'rs', '32768', DEO),
        ('fc1', 'fc1', 'ag_gu', 'ag', '32768', DEO), ('fc2', 'fc2', 'rs', 'rs', '32768', DEO)]
    assert all(r['on_operator_table'] == 'False' for r in rows)                 # off-table targets, all four
    assert rows[0]['out_of_range'] == f'deo:NK;{TRITON_DIST}:NK' and rows[1]['out_of_range'] == 'deo:NK'
    assert rows[0]['support_triton_dist'] == 'supported' and rows[1]['support_triton_dist'] == 'unsupported'
    assert float(rows[0]['predicted_log_speedup_deo']) > float(rows[0]['predicted_log_speedup_triton_dist']) > 0
    with pytest.raises(SystemExit, match='move it aside'):
        call(tables, 'sites', str(root), '--plans', str(tmp_path))
