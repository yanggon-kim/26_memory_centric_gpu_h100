#!/usr/bin/env python3
"""CPU validation fixtures for scripts/amd/cubw/analyze.py (no GPU). Run: python3 test_analyze.py

Synthetic campaign roots exercise: failure/status separation, nested trials in the bootstrap, missing
clock readings, monitor coverage gaps, overlapping competitor windows (contrasted with the old
summarize_competitor.py unweighted mean), not-run steps, and gate/release logic.
"""
import json
from pathlib import Path
import random
import statistics as st
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

UUID = 'GPU-test0001'
NS = 1_000_000_000


def run_json(path, ms, start_s, digest='d0', status='complete', identical=True, mask='unset', n_trials=10):
    trials = []
    t = start_s
    for i in range(n_trials):
        trials.append({'cuda_event_ms': ms[i % len(ms)] * 63, 'wall_ms': ms[i % len(ms)] * 63 + 1, 'decode_iterations': 63,
                       'prefill_wall_ms': 300.0, 'output_sha256': digest,
                       'wall_start_unix_ns': int(t * NS), 'wall_end_unix_ns': int((t + 1.0) * NS)})
        t += 2.0
    d = {'status': status, 'hardware': {'uuid': UUID}, 'environment': {'HSA_CU_MASK': mask},
         'script_sha256': {'controlled_vllm_experiments.py': 'abc'},
         'cases': [{'batch_size': 1, 'context_length': 4096, 'output_digests_identical': identical, 'trials': trials}]}
    path.mkdir(parents=True, exist_ok=True)
    (path / 'run.json').write_text(json.dumps(d))
    return t


class Root:
    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix='cubw_fixture_'))
        self.steps, self.journal, self.monitor = [], [], []
        (self.dir / 'journal').mkdir()
        (self.dir / 'gpus.tsv').write_text(f'0\t{UUID}\t-\n')
        self.t = 1000.0

    def vllm(self, exp, n, treatment='absent', block=0, ms=(10.0,), digest='d0', status='complete', identical=True,
             rc=0, competitor_windows=None, notrun=None):
        sid = f'{exp}:q__n{n}__{treatment}:a{block}'
        step = {'id': sid, 'lane': 'L0', 'stage': 's', 'experiment': exp, 'kind': 'vllm', 'cell': f'q__n{n}__{treatment}',
                'acquisition': block, 'block': block, 'model': 'qwen32b', 'case': 'qwen32b_graph_b1_c4096',
                'decoder_cus': n, 'treatment': treatment, 'mode': 'timing'}
        self.steps.append(step)
        if notrun:
            self.journal.append({'step': sid, 'disposition': 'not_run', 'reason': notrun})
            return
        adir = self.dir / 'runs' / sid.replace(':', '_') / 'attempt1'
        mask = 'unset' if n == 304 else f'0:0-{n - 1}'
        start = self.t
        self.t = run_json(adir / 'run', ms, start, digest, status, identical, mask) + 5
        rec = {'step': sid, 'attempt': 1, 'attempt_dir': str(adir), 'kind': 'vllm', 'gpu_uuid': UUID,
               'command_completed': rc == 0 and status == 'complete', 'returncode': rc,
               'decoder_mask': None if n == 304 else f'0-{n - 1}', 'residual_gpu_pids': []}
        if status != 'complete':
            rec['failure_class'] = 'execution'
        if competitor_windows is not None:
            lines = [json.dumps({'event': 'started'})]
            for a, b, gb in competitor_windows(start):
                lines.append(json.dumps({'t0_unix_ns': int(a * NS), 't1_unix_ns': int(b * NS), 't0_mono_ns': int(a * NS) - 7,
                                         't1_mono_ns': int(b * NS) - 7, 'gb_per_s': gb, 'tflop_per_s': 0.0, 'busy_fraction': 1.0}))
            lines.append(json.dumps({'event': 'stopped'}))
            (adir / 'competitor.jsonl').write_text('\n'.join(lines) + '\n')
            rec['competitor'] = {'mode': 'read', 'duty_requested': 1.0, 'hsa_cu_mask': '0:232-303', 'stopped_event': True,
                                 'stop': {'gone': True, 'stopped_by': 'SIGTERM'}}
        self.journal.append(rec)
        return start

    def clocks(self, start_s, end_s, gfx=2100, mem=1300, step=0.1, gap=None, none_at=None):
        t = start_s
        while t < end_s:
            if not (gap and gap[0] <= t < gap[1]):
                g = None if (none_at and none_at[0] <= t < none_at[1]) else gfx
                self.monitor.append({'t_unix_ns': int(t * NS), 'uuid': UUID, 'gfx_mhz': g, 'mem_mhz': mem})
            t += step

    def write(self):
        plan = {'cohort': 'mi300x', 'kind': 'fixture', 'levels': [304, 232, 200, 152, 96], 'e3_levels': [232, 200, 152, 96],
                'lanes': {'L0': {'gpu_index': 0}}, 'steps': self.steps}
        (self.dir / 'launch_plan.json').write_text(json.dumps(plan))
        (self.dir / 'journal' / 'L0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.journal))
        (self.dir / 'node_monitor.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.monitor))
        return self.dir


def check(name, cond, detail=''):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        check.failed += 1


check.failed = 0


def test_status_separation():
    r = Root()
    r.vllm('e2', 304, block=0)
    r.vllm('e2', 152, block=0, identical=False)        # command completed, outputs differ -> correctness failure
    r.vllm('e2', 232, block=0, status='failed', rc=1)   # execution failure
    r.vllm('e2', 200, block=0, notrun='deadline')       # not run
    root = r.write()
    plan, acqs, *_ = A.build(root)
    by = {a['n']: a for a in acqs}
    check('rc0 + differing outputs is a correctness failure, not success',
          by[152]['command_completed'] is True and by[152]['correctness'] is False and by[152]['disposition'] == 'failed')
    check('execution failure kept separate from correctness', by[232]['disposition'] == 'failed' and by[232]['correctness'] is not True)
    check('not_run steps surface with their reason', by[200]['disposition'] == 'not_run' and by[200].get('not_run_reason') == 'deadline')
    d, why = A.disposition([by[152]], True, True, 1)
    check('cell with an output mismatch is failed', d == 'failed', why)


def test_nesting():
    rng = random.Random(1)
    # three blocks; large trial noise but identical block means -> the CI must come from 3 units, not 30 trials
    num = {b: [12 + rng.gauss(0, 1.0) for _ in range(10)] for b in range(3)}
    den = {b: [10 + rng.gauss(0, 1.0) for _ in range(10)] for b in range(3)}
    r = A.ratio_ci(num, den, blocked=True)
    check('bootstrap unit count is blocks, not trials', r['units'] == 3, str(r['units']))
    flat_num = {i: [x] for i, x in enumerate(v for vals in num.values() for v in vals)}
    flat_den = {i: [x] for i, x in enumerate(v for vals in den.values() for v in vals)}
    rf = A.ratio_ci(flat_num, flat_den, blocked=False)
    check('treating trials as independent units would claim 30 units', rf['units'] == 30)
    check('point estimate is ratio of block means', abs(r['ratio'] - st.mean(map(st.mean, num.values())) / st.mean(map(st.mean, den.values()))) < 1e-12)


def test_missing_clocks_and_coverage():
    r = Root()
    s0 = r.vllm('e2', 304, block=0)
    s1 = r.vllm('e2', 152, block=0)
    r.clocks(s0 - 1, r.t)
    root = r.write()
    _, acqs, *_ = A.build(root)
    check('complete monitor coverage passes pooled clock gate', A.clock_ok(acqs), str(A.clock_summary(acqs)))
    r = Root()
    s0 = r.vllm('e2', 304, block=0)
    s1 = r.vllm('e2', 152, block=0)
    r.clocks(s0 - 1, r.t, none_at=(s1 + 0.2, s1 + 0.6))
    _, acqs, *_ = A.build(r.write())
    check('a missing (None) clock reading fails the gate', not A.clock_ok(acqs))
    r = Root()
    s0 = r.vllm('e2', 304, block=0)
    s1 = r.vllm('e2', 152, block=0)
    r.clocks(s0 - 1, r.t, gap=(s1 + 2.0, s1 + 3.0))   # trial 1 window has no samples at all
    _, acqs, *_ = A.build(r.write())
    check('a monitor gap (coverage < 0.5) fails the gate', not A.clock_ok(acqs),
          f"coverage_min={min(a['clock']['coverage_min'] for a in acqs):.2f}")
    r = Root()
    s0 = r.vllm('e2', 304, block=0)
    s1 = r.vllm('e2', 152, block=0)
    r.clocks(s0 - 1, s1 - 1, gfx=2100)
    r.clocks(s1 - 1, r.t, gfx=1900)                    # each cell is flat, the pool spans 2100/1900 = 1.105
    _, acqs, *_ = A.build(r.write())
    check('individual cell passes do not make a pooled pass', all(a['clock_cell_ok'] for a in acqs) and not A.clock_ok(acqs))


def test_overlapping_windows():
    # each trial [s, s+1]; competitor windows overlap: [s-0.9, s+0.9] 100 GB/s and [s+0.8, s+1.05] 300 GB/s
    r = Root()
    r.vllm('e3', 232, 'mem100', block=0, competitor_windows=lambda s: [(s + 2 * i - 0.9, s + 2 * i + 0.9, 100) for i in range(10)]
           + [(s + 2 * i + 0.8, s + 2 * i + 1.05, 300) for i in range(10)])
    _, acqs, *_ = A.build(r.write())
    c = acqs[0]['competitor']
    # clipped: 0.9 s of the trial at 100 GB/s, then [s+0.9, s+1.0] at 300 GB/s -> 120 GB/s weighted
    check('duration-weighted join with clipping', abs(c['gb_per_s_during_trials'] - 120) < 1e-6, f"{c['gb_per_s_during_trials']:.3f}")
    check('trial coverage reported', abs(c['trial_coverage'] - 1.0) < 1e-6, f"{c['trial_coverage']:.3f}")
    # old reader: unweighted mean over every window overlapping any trial, with duplicates per trial
    wins, _ = A.competitor_windows(Path(json.loads((r.dir / 'journal/L0.jsonl').read_text().splitlines()[0])['attempt_dir']) / 'competitor.jsonl')
    raw = [json.loads(l) for l in (Path(json.loads((r.dir / 'journal/L0.jsonl').read_text().splitlines()[0])['attempt_dir'])
                                   / 'competitor.jsonl').read_text().splitlines() if '"gb_per_s"' in l]
    run = json.loads((Path(json.loads((r.dir / 'journal/L0.jsonl').read_text().splitlines()[0])['attempt_dir']) / 'run/run.json').read_text())
    trials = run['cases'][0]['trials']
    hit = [w['gb_per_s'] for w in raw for t in trials if w['t0_unix_ns'] < t['wall_end_unix_ns'] and w['t1_unix_ns'] > t['wall_start_unix_ns']]
    check('old unweighted reader differs on overlapping windows (documented defect)', abs(st.mean(hit) - 120) > 1, f'old={st.mean(hit):.1f}')
    check('realtime-monotonic offset constant', c['realtime_step_ns'] == 0)


def test_gates():
    r = Root()
    for b in range(3):
        r.vllm('e1_e2_pilot', 304, block=b)
        r.vllm('e1_e2_pilot', 152, block=b, digest='d1' if b == 2 else 'd0')
    r.clocks(999, r.t)
    plan, acqs, nat, ptab, comp_only, *_ = A.build(r.write())
    g = A.gates_from(plan, acqs, ptab, comp_only)
    check('a single differing digest fails outputs_ok', g['outputs_ok'] is False and g['outputs_detail']['distinct_digests'] == 2)
    check('no placement probes -> placement_ok false -> nothing released', not g['placement_ok'] and not g['release_e2'])


def test_v11_gates():
    """v1.1: profiled runs are outside outputs_ok; leak is judged against placement-proven native co-runners."""
    def acq(cell, n, profiled=False, comp=None, lane='L0', disp='complete'):
        return {'model': 'qwen32b', 'case': 'qwen32b_graph_b1_c4096', 'mode': 'timing', 'generated_tokens': None,
                'profiled': profiled, 'disposition': disp, 'output_sha256': 'd0', 'experiment': 'e1_leak', 'cell': cell,
                'n': n, 'lane': lane, 'competitor': comp, 'lifecycle_ok': True}
    def probe(tf, mask='232-303'):
        return {'mode': 'fma', 'mask': f'0:{mask}', 'tflop_per_s_during_trials': tf}
    def ref(tf, corun=None, lane='L0', mask='232-303'):
        return {'tflop_per_s': tf, 'gb_per_s': 0.0, 'windows': 30, 'events': ['started', 'stopped'], 'gone': True,
                'residual_gpu_pids': [], 'orphans': [], 'lane': lane, 'probe_mask': mask, 'probe_mode': 'fma',
                'corunner': corun, 'corunner_gone': True if corun else None}
    plan = {'levels': [304, 232]}
    acqs = [acq('x__n232__absent', 232), acq('x__n304__absent__pmc', 304, profiled=True, disp='failed'),
            acq('x__n232__arith100', 232, comp=probe(10.5))]
    comp_only = {'alone': ref(17.4), 'co_fma': ref(10.0, {'treatment': 'arith100', 'mask': '0-231'}),
                 'co_read': ref(11.0, {'treatment': 'mem100', 'mask': '0-231'})}
    g = A.gates_from(plan, acqs, {}, comp_only)
    check('failed profiled run is outside outputs_ok', g['outputs_ok'] is True and g['outputs_detail']['runs'] == 2,
          str(g['outputs_detail']))
    check('v1 leak gate still computed and failing (0.60 < 0.85)', g['leak_ok'] is False)
    r = g['leak_v11']['x__n232__arith100']
    check('v1.1 leak relative to the gentlest co-runner', abs(r['relative'] - 10.5 / 11.0) < 1e-9 and g['leak_ok_v11'] is True,
          str(r))
    comp_only['co_read'] = ref(16.0, {'treatment': 'mem100', 'mask': '0-231'})
    g = A.gates_from(plan, acqs, {}, comp_only)
    check('decoder far below a gentle co-runner fails leak_ok_v11', g['leak_ok_v11'] is False and not g['release_e3_minimal'])
    g = A.gates_from(plan, acqs + [acq('x__n200__absent', 200, disp='not_run')], {}, comp_only)
    check('v1.2: unattempted (not_run) steps are outside outputs_ok', g['outputs_ok'] is True and g['outputs_detail']['runs'] == 2,
          str(g['outputs_detail']))
    g = A.gates_from(plan, acqs + [acq('x__n200__absent', 200, disp='failed')], {}, comp_only)
    check('v1.2: an attempted failed run still fails outputs_ok', g['outputs_ok'] is False)
    comp_only['co_read']['orphans'] = [{'pid': 1, 'gone': False}]
    g = A.gates_from(plan, acqs, {}, comp_only)
    check('an orphan that survived the sweep fails lifecycle_ok', g['lifecycle_ok'] is False)


if __name__ == '__main__':
    for f in (test_status_separation, test_nesting, test_missing_clocks_and_coverage, test_overlapping_windows, test_gates,
              test_v11_gates):
        print(f'-- {f.__name__}')
        f()
    print(f'{check.failed} failures')
    sys.exit(1 if check.failed else 0)
