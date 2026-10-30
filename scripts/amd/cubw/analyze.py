#!/usr/bin/env python3
"""cubw_v1 reader: gates, E2 noninferiority, E3 T(n,z), E4 service curves, E5 coverage, E8 guardrails.

Versioned replacement for the old summaries for this follow-up (old readers and summaries are left
unchanged). Separate fields are kept for command completion, numerical correctness, mask/placement,
clock comparability, isolation, source identity, sample counts and the final disposition
(qualified | observational_clock_confounded | unavailable | failed | not_run).

Statistics: an acquisition's value is the mean of its trials' HIP-event ms per decode step. Ratios use
blocks (one GPU lane = one block holding one acquisition of every cell): 10,000 bootstrap draws (seed
20261018) resample blocks, then trials inside each selected acquisition. Unblocked cells fall back to
resampling acquisitions per cell independently. Trials are never treated as independent processes.

Clock gate: pooled in-window gfx and memory clock samples over every trial window of the compared
cells; max/min <= 1.05 for each, no missing readings, and per-trial sample coverage >= 0.5 of the
monitor's own median cadence. Individual cell passes are reported but never substitute for the pool.

Competitor joins: the owned competitor's ~0.25 s windows are clipped to be non-overlapping, intersected
with each decoder trial window, and averaged weighted by intersected duration; coverage is reported.
Uniform throughput inside a window is assumed (labelled as an attribution estimate). The achieved
competitor rate is a response as well as an intervention; it is not subtracted from decoder bandwidth.

v1.1 (after the MI300X pilot, job 481167; the v1 gates are still computed and reported unchanged):
- outputs_ok counts only unprofiled timing runs (the v1 reader also counted rocprofv3 runs, contrary to
  its own declared definition);
- leak_ok_v11: the v1 probe ratio (probe TFLOP/s during decoder trials / probe alone) is compared with
  the same probe beside a placement-proven native co-runner (svc_bench fma and read) on the decoder
  mask. A ratio well below 1 that a placement-proven co-runner also produces is inter-process
  contention, not CU leakage; leak_ok_v11 requires every decoder probe ratio >= 0.85 x the gentlest
  (highest) native co-runner ratio on the same probe mask and GPU;
- lifecycle also requires every owned process found by the per-attempt tag sweep to be gone.

v1.2 (after MI300X auto job 483498): outputs_ok ignores steps never attempted (disposition not_run). An
auto plan's gate runs after its pilot stage, when every production step is still unattempted; v1.1
counted those 146 steps as incomplete runs, so outputs_ok and every release failed although all 18
attempted pilot runs completed with one digest. Attempted runs that fail still fail the gate.

  analyze.py report --root <root> [--out <dir>]     tables + summary.md
  analyze.py gate   --root <pilot-root> --out gates.json
"""
import argparse
import bisect
import collections
import csv
import json
import math
from pathlib import Path
import random
import statistics as st

DRAWS, BOOT_SEED = 10000, 20261018
CLOCK_ENVELOPE, MIN_COVERAGE, NONINF = 1.05, 0.5, 1.05
LEAK_MIN = 0.85


# ---------------------------------------------------------------- loading

def load_plan(root):
    return json.loads((root / 'launch_plan.json').read_text())


def load_journal(root):
    """step id -> list of attempt records (ordered); plus not_run records."""
    attempts, notrun = collections.defaultdict(list), {}
    for f in sorted((root / 'journal').glob('*.jsonl')):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('disposition') == 'not_run':
                notrun[r['step']] = r['reason']
            elif 'attempt' in r:
                attempts[r['step']].append(r)
    return attempts, notrun


def load_monitor(path):
    samples = collections.defaultdict(list)
    if not path.exists():
        return samples
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        samples[d['uuid']].append((d['t_unix_ns'], d.get('gfx_mhz'), d.get('mem_mhz'), d.get('power_mw')))
    for v in samples.values():
        v.sort()
    return samples


def cadence_s(samples, uuid):
    rows = samples.get(uuid, [])
    if len(rows) < 3:
        return None
    dts = [(b[0] - a[0]) / 1e9 for a, b in zip(rows, rows[1:])]
    return st.median(dts)


def window(samples, uuid, a, b):
    rows = samples.get(uuid, [])
    return rows[bisect.bisect_left(rows, (a,)):bisect.bisect_right(rows, (b, float('inf')))]


def competitor_windows(path):
    wins, events = [], []
    if not path.exists():
        return wins, events
    for line in path.read_text().splitlines():
        if not line.startswith('{'):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        (events if 'event' in d else wins).append(d)
    for w in wins:  # offset measured before clipping
        if 't0_mono_ns' in w:
            w['rt_offset_ns'] = w['t0_unix_ns'] - w['t0_mono_ns']
    wins.sort(key=lambda w: w['t0_unix_ns'])
    end = None
    for w in wins:  # clip to non-overlapping intervals (deduplication)
        if end is not None and w['t0_unix_ns'] < end:
            w['t0_unix_ns'] = end
        end = max(end or 0, w['t1_unix_ns'])
    return [w for w in wins if w['t1_unix_ns'] > w['t0_unix_ns']], events


def overlap_rates(wins, a, b):
    """Duration-weighted competitor rates over [a, b] and the covered fraction."""
    cov, gb, tf, busy = 0, 0.0, 0.0, 0.0
    for w in wins:
        lo, hi = max(a, w['t0_unix_ns']), min(b, w['t1_unix_ns'])
        if hi > lo:
            d = hi - lo
            cov += d; gb += w['gb_per_s'] * d; tf += w['tflop_per_s'] * d; busy += w.get('busy_fraction', 0) * d
    if not cov:
        return {'coverage': 0.0}
    return {'coverage': cov / (b - a), 'gb_per_s': gb / cov, 'tflop_per_s': tf / cov, 'busy_fraction': busy / cov}


def realtime_step_ns(wins):
    """Spread of (realtime - monotonic) across windows: a realtime step would break the join."""
    offs = [w['rt_offset_ns'] for w in wins if 'rt_offset_ns' in w]
    return (max(offs) - min(offs)) if offs else None


# ---------------------------------------------------------------- acquisitions

def final_attempt(atts):
    return atts[-1] if atts else None


def acquisition(step, atts, samples, plan):
    """One decoder acquisition with every gate field kept separate."""
    rec = final_attempt(atts)
    a = {'step': step['id'], 'experiment': step['experiment'], 'cell': step['cell'], 'model': step.get('model'),
         'case': step.get('case'), 'n': step.get('decoder_cus'), 'treatment': step.get('treatment'),
         'block': step.get('block'), 'lane': step['lane'], 'acquisition': step.get('acquisition'),
         'attempts': len(atts), 'profiled': bool(step.get('profile')), 'mode': step.get('mode', 'timing'),
         'generated_tokens': step.get('generated_tokens')}
    if rec is None:
        a['disposition'] = 'not_run'
        return a
    a['gpu_uuid'] = rec.get('gpu_uuid')
    a['command_completed'] = bool(rec.get('command_completed'))
    a['failure_class'] = rec.get('failure_class')
    run_path = Path(rec['attempt_dir']) / 'run' / 'run.json'
    if not run_path.exists():
        a['disposition'] = 'failed'
        return a
    d = json.loads(run_path.read_text())
    a['run_status'] = d['status']
    a['script_sha256'] = d.get('script_sha256', {}).get('controlled_vllm_experiments.py')
    env_mask = d.get('environment', {}).get('HSA_CU_MASK', 'unset')
    want = rec.get('decoder_mask')
    a['mask_recorded'] = env_mask
    a['mask_ok'] = (env_mask in ('unset', '') and not want) or (want and env_mask.endswith(':' + want))
    if d['status'] != 'complete' or not d.get('cases'):
        a['correctness'] = False if rec.get('failure_class') == 'correctness' else None
        a['disposition'] = 'failed'
        return a
    case = d['cases'][0]
    a['correctness'] = bool(case.get('output_digests_identical'))
    a['output_sha256'] = case['trials'][0]['output_sha256'] if case['trials'] else None
    a['context'] = case['context_length']; a['batch'] = case['batch_size']
    trials = case['trials']
    a['trials'] = len(trials)
    a['ms_per_step'] = [t['cuda_event_ms'] / t['decode_iterations'] for t in trials]
    a['wall_ms_per_step'] = [t['wall_ms'] / t['decode_iterations'] for t in trials]
    a['prefill_wall_ms'] = [t['prefill_wall_ms'] for t in trials]
    a['windows'] = [(t['wall_start_unix_ns'], t['wall_end_unix_ns']) for t in trials]
    a['mean_ms_per_step'] = st.mean(a['ms_per_step'])
    if a['mode'] == 'energy':
        tel = [t.get('telemetry', {}) for t in trials]
        a['energy'] = {'joules_per_token': [x.get('joules_per_token') for x in tel],
                       'counter_j': [x.get('counter_j') for x in tel], 'window_s': [x.get('window_s') for x in tel],
                       'agreement': [x.get('counter_power_relative_difference') for x in tel]}
    # clocks per trial
    uuid = d['hardware']['uuid']
    cad = cadence_s(samples, uuid)
    gfx, mem, cover = [], [], []
    for lo, hi in a['windows']:
        w = window(samples, uuid, lo, hi)
        gfx += [s[1] for s in w]; mem += [s[2] for s in w]
        cover.append(len(w) * cad / ((hi - lo) / 1e9) if cad else 0.0)
    a['clock'] = {'gfx': gfx, 'mem': mem, 'coverage_min': min(cover) if cover else 0.0, 'cadence_s': cad}
    a['clock_cell_ok'] = clock_ok([a])
    # isolation and competitor
    comp = rec.get('competitor')
    a['isolation_ok'] = rec.get('residual_gpu_pids') == []
    a['orphans'] = rec.get('orphans')
    if comp:
        wins, events = competitor_windows(Path(rec['attempt_dir']) / 'competitor.jsonl')
        joined = [overlap_rates(wins, lo, hi) for lo, hi in a['windows']]
        tot = sum(((hi - lo) for lo, hi in a['windows']))
        cov = sum(j['coverage'] * (hi - lo) for j, (lo, hi) in zip(joined, a['windows']))
        def wavg(key):
            num = sum(j.get(key, 0) * j['coverage'] * (hi - lo) for j, (lo, hi) in zip(joined, a['windows']))
            return num / cov if cov else None
        a['competitor'] = {'mode': comp['mode'], 'duty_requested': comp['duty_requested'], 'mask': comp['hsa_cu_mask'],
                           'trial_coverage': cov / tot if tot else 0, 'gb_per_s_during_trials': wavg('gb_per_s'),
                           'tflop_per_s_during_trials': wavg('tflop_per_s'), 'busy_fraction': wavg('busy_fraction'),
                           'started': 'start_failed' not in comp, 'stopped_event': comp.get('stopped_event'),
                           'gone': comp.get('stop', {}).get('gone'), 'stopped_by': comp.get('stop', {}).get('stopped_by'),
                           'realtime_step_ns': realtime_step_ns(wins),
                           'attribution': 'uniform throughput within each ~0.25 s window (estimate)'}
        a['lifecycle_ok'] = bool(a['competitor']['started'] and a['competitor']['stopped_event'] and a['competitor']['gone']
                                 and a['isolation_ok'] and a['competitor']['trial_coverage'] >= 0.95
                                 and all(o.get('gone') for o in rec.get('orphans') or []))
    a['counts_ok'] = a['trials'] == 10 or a['mode'] == 'energy' or a['profiled']
    a['disposition'] = 'complete' if (a['command_completed'] and a['correctness']) else 'failed'
    return a


def clock_ok(acqs):
    g = [x for a in acqs for x in a.get('clock', {}).get('gfx', [])]
    m = [x for a in acqs for x in a.get('clock', {}).get('mem', [])]
    if not g or not m or any(x is None or x <= 0 for x in g + m):
        return False
    if any(a.get('clock', {}).get('coverage_min', 0) < MIN_COVERAGE for a in acqs):
        return False
    return max(g) / min(g) <= CLOCK_ENVELOPE and max(m) / min(m) <= CLOCK_ENVELOPE


def clock_summary(acqs):
    g = [x for a in acqs for x in a.get('clock', {}).get('gfx', []) if x]
    m = [x for a in acqs for x in a.get('clock', {}).get('mem', []) if x]
    return {'pooled_ok': clock_ok(acqs), 'gfx_min': min(g, default=None), 'gfx_max': max(g, default=None),
            'gfx_ratio': max(g) / min(g) if g else None, 'mem_min': min(m, default=None), 'mem_max': max(m, default=None),
            'samples': len(g), 'coverage_min': min((a['clock']['coverage_min'] for a in acqs if 'clock' in a), default=None)}


# ---------------------------------------------------------------- statistics

def ratio_ci(num, den, blocked=True):
    """num, den: {block: [trial values]} (or lists of trial lists when unblocked). Returns point, lo, hi, blocks."""
    rng = random.Random(BOOT_SEED)
    if blocked:
        keys = sorted(set(num) & set(den))
        if not keys:
            return None
        point = st.mean(st.mean(num[k]) for k in keys) / st.mean(st.mean(den[k]) for k in keys)
        draws = []
        for _ in range(DRAWS):
            pick = [rng.choice(keys) for _ in keys]
            nn = st.mean(st.mean(rng.choice(num[k]) for _ in num[k]) for k in pick)
            dd = st.mean(st.mean(rng.choice(den[k]) for _ in den[k]) for k in pick)
            draws.append(nn / dd)
        n_units = len(keys)
    else:
        nl, dl = list(num.values()), list(den.values())
        if not nl or not dl:
            return None
        point = st.mean(st.mean(x) for x in nl) / st.mean(st.mean(x) for x in dl)
        draws = []
        for _ in range(DRAWS):
            nn = st.mean(st.mean(rng.choice(a) for _ in a) for a in (rng.choice(nl) for _ in nl))
            dd = st.mean(st.mean(rng.choice(a) for _ in a) for a in (rng.choice(dl) for _ in dl))
            draws.append(nn / dd)
        n_units = min(len(nl), len(dl))
    draws.sort()
    return {'ratio': point, 'ci95': [draws[int(.025 * DRAWS)], draws[int(.975 * DRAWS) - 1]], 'units': n_units,
            'paired_by_block': blocked}


def by_block(acqs):
    out = {}
    for a in acqs:
        key = a['block'] if a['block'] is not None else a['lane']
        out[key] = a['ms_per_step']
    return out


def good(a):
    return a.get('disposition') == 'complete' and a.get('mask_ok') and a.get('counts_ok') and not a.get('profiled')


def disposition(acqs, gates_ok, clock_pass, min_units=3):
    if not acqs:
        return 'not_run', 'no acquisitions'
    if any(a.get('correctness') is False for a in acqs):
        return 'failed', 'output/correctness failure present'
    ok = [a for a in acqs if good(a)]
    if len(ok) < min_units:
        return 'unavailable', f'{len(ok)} complete acquisitions < {min_units}'
    if not gates_ok:
        return 'unavailable', 'placement/isolation/lifecycle gate failed'
    if not clock_pass:
        return 'observational_clock_confounded', 'pooled clock envelope failed'
    return 'qualified', ''


# ---------------------------------------------------------------- native / placement / E4

def native_rows(plan, attempts, samples, gpus):
    rows = []
    for s in plan['steps']:
        if s['kind'] != 'native':
            continue
        rec = final_attempt(attempts.get(s['id'], []))
        r = {'step': s['id'], 'experiment': s['experiment'], 'name': s['cell'], 'level': s.get('level'),
             'acquisition': s.get('acquisition'), 'lane': s['lane'], 'expect_failure': s.get('expect_failure', False)}
        if rec is None:
            r['disposition'] = 'not_run'
            rows.append(r); continue
        r.update(returncode=rec.get('returncode'), timed_out=rec.get('timed_out'), check_passed=rec.get('check_passed'))
        out = Path(rec['attempt_dir']) / 'out.json'
        lines = out.read_text().strip().splitlines() if out.exists() else []
        try:
            res = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:
            res = None
        r['result'] = res
        uuid = gpus.get(s['lane'])
        if uuid and rec.get('t_start'):
            w = window(samples, uuid, int(rec['t_start'] * 1e9), int(rec['t_end'] * 1e9))
            g = [x[1] for x in w if x[1]]
            r['gfx_mhz_range'] = [min(g), max(g)] if g else None
        r['disposition'] = 'complete' if rec.get('returncode') == 0 and res and res.get('check_passed') else \
            ('recorded_expected_failure' if s.get('expect_failure') else 'failed')
        rows.append(r)
    return rows


def placement_table(rows):
    out = {}
    for r in rows:
        if r['experiment'] != 'e1_placement':
            continue
        res = r.get('result') or {}
        per = res.get('per_xcc', {})
        balanced = bool(per) and max(per.values()) - min(per.values()) <= 1 and len(per) == 8
        out[r['name']] = {'returncode': r.get('returncode'), 'timed_out': r.get('timed_out'),
                          'distinct_cus': res.get('distinct_cus'), 'expected_cus': res.get('expected_cus'),
                          'per_xcc': per, 'xcc_balanced': balanced, 'check_passed': res.get('check_passed'),
                          'hsa_cu_mask': res.get('hsa_cu_mask'), 'expect_failure_probe': r['expect_failure']}
    return out


def placement_ok(table):
    """Every lane that ran probes proved every decoder level and the competitor partition."""
    lanes = sorted({k.split('_', 1)[0] for k in table if k.startswith('L') and '_' in k})
    if not lanes:
        return False
    for lane in lanes:
        need = [f'{lane}_n{n}' for n in (304, 232, 200, 152, 96)] + [f'{lane}_competitor_232-303']
        if not all(k in table and table[k]['check_passed'] and table[k]['xcc_balanced'] for k in need):
            return False
    return True


def metric(res):
    for k in ('useful_gb_per_s', 'useful_tflop_per_s', 'ns_per_dependent_load'):
        if res and k in res:
            return k, res[k]
    return None, None


def e4_table(rows):
    cells = collections.defaultdict(list)
    for r in rows:
        if r['experiment'] not in ('e4', 'e4_pilot') or r['disposition'] != 'complete':
            continue
        k, v = metric(r['result'])
        name = r['name'].split('_', 1)[1] if r['name'].startswith('n') and '_' in r['name'] else r['name']
        cells[(r['experiment'], str(r['level']), name, k)].append(v)
    out = {}
    for (exp, level, name, k), vals in sorted(cells.items()):
        e = {'metric': k, 'acquisitions': len(vals), 'mean': st.mean(vals), 'min': min(vals), 'max': max(vals)}
        ref = cells.get((exp, '304', name, k))
        if ref and level != '304':
            e['ratio_to_304'] = st.mean(vals) / st.mean(ref)
        out.setdefault(exp, {}).setdefault(name, {})[level] = e
    return out


# ---------------------------------------------------------------- E5 profile coverage

def profile_summary(step, rec):
    d = Path(rec['attempt_dir']) / 'rocprof'
    out = {'step': step['id'], 'profile': step['profile'], 'command_completed': rec.get('command_completed')}
    kt = next(d.rglob('*kernel_trace.csv'), None) if d.exists() else None
    mk = next(d.rglob('*marker_api_trace.csv'), None) if d.exists() else None
    cc = next(d.rglob('*counter_collection.csv'), None) if d.exists() else None
    ranges = []
    if mk:
        for r in csv.DictReader(open(mk)):
            if r['Function'] == 'decode':
                ranges.append((int(r['Start_Timestamp']), int(r['End_Timestamp'])))
    ranges.sort()
    out['decode_ranges'] = len(ranges)

    def inside(t):
        i = bisect.bisect_right(ranges, (t, float('inf'))) - 1
        return i >= 0 and ranges[i][0] <= t <= ranges[i][1]
    run = Path(rec['attempt_dir']) / 'run' / 'run.json'
    ticks = 63
    if run.exists():
        rd = json.loads(run.read_text())
        if rd.get('cases') and rd['cases'][0]['trials']:
            ticks = rd['cases'][0]['trials'][0]['decode_iterations']
            out['profiled_ms_per_step'] = st.mean(t['cuda_event_ms'] / t['decode_iterations'] for t in rd['cases'][0]['trials'])
            out['output_sha256'] = rd['cases'][0]['trials'][0]['output_sha256']
    if kt:
        rows = [r for r in csv.DictReader(open(kt))]
        dec = [r for r in rows if inside(int(r['Start_Timestamp']))]
        out['dispatches_total'] = len(rows)
        out['dispatches_in_decode'] = len(dec)
        if ranges and dec:
            out['dispatches_per_tick'] = len(dec) / (len(ranges) * ticks)
            iv = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in dec)
            busy, cur = 0, None
            for lo, hi in iv:
                if cur and lo <= cur[1]:
                    cur[1] = max(cur[1], hi)
                else:
                    if cur:
                        busy += cur[1] - cur[0]
                    cur = [lo, hi]
            busy += cur[1] - cur[0]
            span = sum(hi - lo for lo, hi in ranges)
            out['busy_union_fraction_of_decode_ranges'] = busy / span
            names = collections.Counter(r['Kernel_Name'][:80] for r in dec)
            out['top_kernels'] = names.most_common(8)
    if cc:
        rows = [r for r in csv.DictReader(open(cc))]
        dec = [r for r in rows if inside(int(r['Start_Timestamp']))]
        sums = collections.defaultdict(float)
        for r in dec:
            sums[r['Counter_Name']] += float(r['Counter_Value'])
        out['counter_rows_total'] = len(rows)
        out['counter_rows_in_decode'] = len(dec)
        if ranges:
            out['counter_per_tick'] = {k: v / (len(ranges) * ticks) for k, v in sums.items()}
        if kt:
            out['counter_dispatch_coverage'] = len({r['Dispatch_Id'] for r in dec}) / max(out.get('dispatches_in_decode', 0), 1)
        out['byte_domain'] = 'FETCH_SIZE/WRITE_SIZE: KB observed at TCC before Infinity Cache/MALL (pre-MALL D), not physical HBM D_H'
    return out


# ---------------------------------------------------------------- report

def build(root):
    plan = load_plan(root)
    attempts, notrun = load_journal(root)
    samples = load_monitor(root / 'node_monitor.jsonl')
    gpus = {}
    if (root / 'gpus.tsv').exists():
        idx = {int(l.split('\t')[0]): l.split('\t')[1] for l in (root / 'gpus.tsv').read_text().splitlines()}
        gpus = {lane: idx.get(v['gpu_index']) for lane, v in plan['lanes'].items()}
    acqs = [acquisition(s, attempts.get(s['id'], []), samples, plan) for s in plan['steps'] if s['kind'] == 'vllm']
    for a in acqs:
        if a['step'] in notrun:
            a['not_run_reason'] = notrun[a['step']]
    nat = native_rows(plan, attempts, samples, gpus)
    ptab = placement_table(nat)
    comp_only = {}
    for s in plan['steps']:
        if s['kind'] == 'competitor_only':
            rec = final_attempt(attempts.get(s['id'], []))
            if rec:
                wins, ev = competitor_windows(Path(rec['attempt_dir']) / 'competitor.jsonl')
                co = rec.get('corunner')
                if co and co.get('started_unix') and co.get('stopped_unix'):  # probe windows inside the co-run only
                    lo, hi = co['started_unix'] * 1e9, co['stopped_unix'] * 1e9
                    wins = [w for w in wins if w['t0_unix_ns'] >= lo and w['t1_unix_ns'] <= hi]
                steady = wins[1:-1] or wins
                comp_only[s['cell']] = {'tflop_per_s': st.mean(w['tflop_per_s'] for w in steady) if steady else None,
                                        'gb_per_s': st.mean(w['gb_per_s'] for w in steady) if steady else None,
                                        'windows': len(wins), 'events': [e['event'] for e in ev],
                                        'gone': rec['competitor'].get('stop', {}).get('gone'),
                                        'residual_gpu_pids': rec.get('residual_gpu_pids'),
                                        'orphans': rec.get('orphans'), 'lane': rec.get('lane'),
                                        'probe_mask': rec['competitor']['hsa_cu_mask'].split(':', 1)[1],
                                        'probe_mode': rec['competitor']['mode'],
                                        'corunner': s.get('corunner'),
                                        'corunner_gone': (co or {}).get('stop', {}).get('gone')}
    return plan, acqs, nat, ptab, comp_only, attempts, samples


def gates_from(plan, acqs, ptab, comp_only):
    g = {'placement_ok': placement_ok(ptab)}
    comp = [a for a in acqs if a.get('competitor')]
    alone_ok = all(v['gone'] and v['residual_gpu_pids'] == [] and 'stopped' in v['events']
                   and all(o.get('gone') for o in v.get('orphans') or []) and v.get('corunner_gone', True) is not False
                   for v in comp_only.values())
    g['lifecycle_ok'] = bool(comp) and all(a.get('lifecycle_ok') for a in comp) and alone_ok
    q = [a for a in acqs if a.get('model') == 'qwen32b' and a['case'] == 'qwen32b_graph_b1_c4096' and a['mode'] == 'timing'
         and not a.get('generated_tokens') and not a.get('profiled') and a.get('disposition') != 'not_run']
    digests = {a.get('output_sha256') for a in q if a.get('disposition') == 'complete'}
    g['outputs_ok'] = bool(q) and all(a.get('disposition') == 'complete' for a in q) and len(digests) == 1
    g['outputs_detail'] = {'runs': len(q), 'complete': sum(a.get('disposition') == 'complete' for a in q),
                           'distinct_digests': len(digests)}
    leak = {}
    for a in comp:
        if a['competitor']['mode'] != 'fma':
            continue
        mask = a['competitor']['mask'].split(':', 1)[1]
        ref = comp_only.get(f'alone_fma_{mask}')
        if ref and ref['tflop_per_s'] and a['competitor']['tflop_per_s_during_trials']:
            leak[a['cell']] = a['competitor']['tflop_per_s_during_trials'] / ref['tflop_per_s']
    g['leak_ratios'] = leak
    g['leak_ok'] = bool(leak) and all(v >= LEAK_MIN for v in leak.values())
    # v1.1: decoder probe ratio relative to the gentlest placement-proven native co-runner on the same mask/GPU
    v11 = {}
    for a in comp:
        if a['competitor']['mode'] != 'fma' or not a['competitor']['tflop_per_s_during_trials']:
            continue
        mask = a['competitor']['mask'].split(':', 1)[1]
        def ref(corun):
            rows = [v for v in comp_only.values() if v['probe_mask'] == mask and v['probe_mode'] == 'fma'
                    and bool(v['corunner']) == corun and v['tflop_per_s']]
            same = [v for v in rows if v['lane'] == a['lane']]
            return same or rows
        alone, coruns = ref(False), ref(True)
        if alone and coruns:
            base = st.mean(v['tflop_per_s'] for v in alone)
            native = {v['corunner']['treatment']: v['tflop_per_s'] / base for v in coruns}
            dec = a['competitor']['tflop_per_s_during_trials'] / base
            v11[a['cell']] = {'decoder_ratio': dec, 'native_ratios': native, 'relative': dec / max(native.values())}
    g['leak_v11'] = v11
    g['leak_ok_v11'] = bool(v11) and all(v['relative'] >= LEAK_MIN for v in v11.values())
    e2 = [a for a in q if a['experiment'] in ('e1_e2_pilot', 'e2') and good(a)]
    g['clock'] = clock_summary(e2)
    g['clock_ok'] = g['clock']['pooled_ok']
    g['release_e2'] = g['placement_ok'] and g['outputs_ok']
    g['release_e3_minimal'] = g['placement_ok'] and g['lifecycle_ok'] and g['outputs_ok'] and g['leak_ok_v11']
    g['release_e3_full'] = g['release_e3_minimal'] and g['clock_ok']
    g['release_e5e8'] = g['outputs_ok']
    return g


def report(args):
    root = args.root
    out = args.out or (root / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    plan, acqs, nat, ptab, comp_only, attempts, samples = build(root)
    gates = gates_from(plan, acqs, ptab, comp_only)
    struct_ok = gates['placement_ok']
    tables = {'version': 'cubw_v1', 'scope': __doc__.strip(), 'cohort': plan['cohort'], 'plan_kind': plan['kind'],
              'plan_sha256': plan.get('plan_sha256'), 'gates': gates, 'placement': ptab, 'competitor_alone': comp_only}
    # E2 per model (production 'e2'; pilot rows are labelled pilot and never promoted)
    e2 = {}
    for exp in ('e2', 'e1_e2_pilot'):
        for model in sorted({a['model'] for a in acqs if a['experiment'] == exp}):
            for gpu_tag in sorted({a['cell'].endswith('__gpu7') for a in acqs if a['experiment'] == exp and a['model'] == model}):
                sel = [a for a in acqs if a['experiment'] == exp and a['model'] == model and a['cell'].endswith('__gpu7') == gpu_tag]
                base = [a for a in sel if a['n'] == 304 and good(a)]
                key = f"{exp}:{model}{':gpu7' if gpu_tag else ''}"
                levels = {}
                for n in plan['levels']:
                    cell = [a for a in sel if a['n'] == n]
                    okc = [a for a in cell if good(a)]
                    entry = {'acquisitions': len(cell), 'complete': len(okc),
                             'mean_ms_per_step': st.mean(a['mean_ms_per_step'] for a in okc) if okc else None,
                             'clock_cells': [a.get('clock_cell_ok') for a in okc], 'mask_recorded': sorted({a.get('mask_recorded') for a in cell if a.get('mask_recorded')})}
                    if okc and base and n != 304:
                        r = ratio_ci(by_block(okc), by_block(base), blocked=True)
                        entry['R'] = r
                        clk = clock_summary(okc + base)
                        entry['pooled_clock_vs_304'] = clk
                        disp, why = disposition(cell, struct_ok, clk['pooled_ok'], 1 if exp != 'e2' else 3)
                        if exp != 'e2':
                            disp, why = 'pilot', 'one pilot acquisition; never promoted to a production comparison'
                        entry['disposition'], entry['reason'] = disp, why
                        entry['noninferior_5pct'] = bool(disp == 'qualified' and r and r['ci95'][1] <= NONINF)
                    levels[n] = entry
                q = [n for n in plan['levels'] if n != 304 and levels[n].get('noninferior_5pct')]
                e2[key] = {'levels': levels, 'smallest_noninferior_qualified': min(q) if q else None,
                           'criterion': 'upper 95% CI of R(n) <= 1.05 latency regression (not throughput, not equality)'}
    tables['e2'] = e2
    # E3: T(n, z) per model, ratios to absent at fixed n, memory vs arithmetic, cross-n contrasts
    e3 = {}
    for exp in ('e3', 'e1_e3_pilot', 'e1_leak'):
        for model in sorted({a['model'] for a in acqs if a['experiment'] == exp}):
            sel = [a for a in acqs if a['experiment'] == exp and a['model'] == model]
            cells = collections.defaultdict(list)
            for a in sel:
                cells[(a['n'], a['treatment'])].append(a)
            rows = {}
            for (n, t), cs in sorted(cells.items()):
                okc = [a for a in cs if good(a)]
                e = {'acquisitions': len(cs), 'complete': len(okc),
                     'T_ms_per_step': st.mean(a['mean_ms_per_step'] for a in okc) if okc else None,
                     'competitor': [a.get('competitor') for a in okc if a.get('competitor')],
                     'lifecycle_ok': all(a.get('lifecycle_ok', True) for a in okc),
                     'not_run_reasons': sorted({a['not_run_reason'] for a in cs if a.get('not_run_reason')})}
                absent = [a for a in cells.get((n, 'absent'), []) if good(a)]
                if okc and absent and t != 'absent':
                    e['ratio_to_absent_same_n'] = ratio_ci(by_block(okc), by_block(absent), blocked=exp == 'e3')
                    clk = clock_summary(okc + absent)
                    e['pooled_clock'] = clk
                    if exp == 'e3':
                        d, why = disposition(cs, struct_ok and gates['lifecycle_ok'], clk['pooled_ok'])
                    else:
                        d, why = 'pilot', 'pilot'
                    e['disposition'], e['reason'] = d, why
                rows[f'n{n}:{t}'] = e
            contrasts = {}
            def cc(a_key, b_key, label):
                A = [a for a in cells.get(a_key, []) if good(a)]
                B = [a for a in cells.get(b_key, []) if good(a)]
                if A and B:
                    r = ratio_ci(by_block(A), by_block(B), blocked=exp == 'e3')
                    clk = clock_summary(A + B)
                    contrasts[label] = {'ratio': r, 'numerator': f'n={a_key[0]} {a_key[1]}', 'denominator': f'n={b_key[0]} {b_key[1]}',
                                        'pooled_clock_ok': clk['pooled_ok'],
                                        'numerator_faster': bool(r and r['ci95'][1] < 1.0)}
            for n in plan['e3_levels']:
                cc((n, 'mem100'), (n, 'arith100'), f'n{n}: memory100 / arithmetic100')
                cc((n, 'mem50'), (n, 'arith100'), f'n{n}: memory50 / arithmetic100')
            for lo in (200, 152, 96):
                for t in ('mem50', 'mem100', 'mem75'):
                    cc((lo, 'absent'), (232, t), f'n{lo} absent / n232 {t} (shared-service balance, not stock-GPU win)')
            e3[f'{exp}:{model}'] = {'cells': rows, 'contrasts': contrasts,
                                    'product': 'T(n, z): z is the declared treatment; not T(n, B_available)'}
    tables['e3'] = e3
    tables['e4'] = e4_table(nat)
    # E5 / timeline
    prof = []
    for s in plan['steps']:
        if s['kind'] == 'vllm' and s.get('profile'):
            rec = final_attempt(attempts.get(s['id'], []))
            if rec:
                p = profile_summary(s, rec)
                ordinary = [a for a in acqs if a['case'] == s['case'] and a['n'] == s['decoder_cus'] and good(a)
                            and a['experiment'] in ('e1_timeline', 'e2', 'e1_e2_pilot') and a['mode'] == 'timing'
                            and not a.get('generated_tokens')]
                if ordinary and p.get('profiled_ms_per_step'):
                    ref = st.mean(a['mean_ms_per_step'] for a in ordinary)
                    p['ordinary_ms_per_step'] = ref
                    p['instrumentation_slowdown'] = p['profiled_ms_per_step'] / ref
                    p['output_matches_ordinary'] = p.get('output_sha256') in {a.get('output_sha256') for a in ordinary}
                prof.append(p)
            else:
                prof.append({'step': s['id'], 'disposition': 'not_run'})
    tables['e5_profiles'] = prof
    # E8 guardrails and energy
    e8 = {}
    for a in acqs:
        if a['experiment'] not in ('e8', 'e8_pilot'):
            continue
        key = f"{a['experiment']}:{a['cell']}"
        e = e8.setdefault(key, {'acquisitions': 0, 'complete': 0, 'ms_per_step': [], 'prefill_ms': [], 'joules_per_token': [],
                                'mean_power_w': []})
        e['acquisitions'] += 1
        if good(a) or (a.get('disposition') == 'complete' and a['mode'] == 'energy'):
            e['complete'] += 1
            e['ms_per_step'].append(a['mean_ms_per_step'])
            e['prefill_ms'].append(st.median(a['prefill_wall_ms']))
            if a.get('energy'):
                jt = [x for x in a['energy']['joules_per_token'] if isinstance(x, (int, float))]
                if jt:
                    e['joules_per_token'].append(st.mean(jt))
                pw = [c / w for c, w in zip(a['energy']['counter_j'], a['energy']['window_s']) if c and w]
                if pw:
                    e['mean_power_w'].append(st.mean(pw))
    tables['e8'] = e8
    tables['acquisitions'] = [{k: v for k, v in a.items() if k not in ('clock', 'windows', 'wall_ms_per_step')} for a in acqs]
    tables['native'] = nat
    (out / 'cubw_tables.json').write_text(json.dumps(tables, indent=1, default=str) + '\n')
    (out / 'summary.md').write_text(summary_md(tables))
    print((out / 'summary.md').read_text())
    return tables


def fmt(r):
    if not r:
        return 'n/a'
    return f"{r['ratio']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] ({r['units']} {'blocks' if r['paired_by_block'] else 'acq'})"


def summary_md(t):
    L = [f"# cubw_v1 {t['plan_kind']} — {t['cohort']}", '', '## Gates', '']
    g = t['gates']
    for k in ('placement_ok', 'lifecycle_ok', 'outputs_ok', 'leak_ok', 'leak_ok_v11', 'clock_ok', 'release_e2',
              'release_e3_minimal', 'release_e3_full', 'release_e5e8'):
        L.append(f'- {k}: **{g.get(k)}**')
    L.append(f"- outputs: {g.get('outputs_detail')}")
    L.append(f"- leak ratios (probe during decode / probe alone): {g.get('leak_ratios')}")
    L.append(f"- leak v1.1 (relative to placement-proven native co-runners): {g.get('leak_v11')}")
    L.append(f"- pooled clocks (Qwen32B E2): {g.get('clock')}")
    L += ['', '## Placement', '', '| probe | distinct CUs | expected | per-XCD | balanced | rc |', '|---|---:|---:|---|---|---:|']
    for k, v in t['placement'].items():
        L.append(f"| {k} | {v['distinct_cus']} | {v['expected_cus']} | {v['per_xcc']} | {v['xcc_balanced']} | {v['returncode']} |")
    L += ['', '## E2 latency ratio R(n) = T(n)/T(304)', '']
    for key, m in t['e2'].items():
        L.append(f'### {key}')
        L.append('| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |')
        L.append('|---:|---:|---:|---|---|---|')
        for n, e in m['levels'].items():
            clk = e.get('pooled_clock_vs_304') or {}
            L.append(f"| {n} | {e['complete']}/{e['acquisitions']} | {e['mean_ms_per_step'] and round(e['mean_ms_per_step'], 4)} | "
                     f"{fmt(e.get('R'))} | {clk.get('pooled_ok')} (gfx x{clk.get('gfx_ratio') and round(clk['gfx_ratio'], 3)}) | "
                     f"{e.get('disposition', '')} {e.get('reason', '')} |")
        L.append(f"Smallest qualified noninferior n: {m['smallest_noninferior_qualified']}")
        L.append('')
    L += ['## E3 T(n, z)', '']
    for key, m in t['e3'].items():
        L.append(f'### {key}')
        L.append('| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |')
        L.append('|---|---:|---:|---|---:|---:|---:|---|')
        for c, e in m['cells'].items():
            comp = e['competitor'][0] if e['competitor'] else {}
            L.append(f"| {c} | {e['complete']}/{e['acquisitions']} | {e['T_ms_per_step'] and round(e['T_ms_per_step'], 4)} | "
                     f"{fmt(e.get('ratio_to_absent_same_n'))} | {comp.get('gb_per_s_during_trials') and round(comp['gb_per_s_during_trials'])} | "
                     f"{comp.get('tflop_per_s_during_trials') and round(comp['tflop_per_s_during_trials'], 1)} | "
                     f"{comp.get('trial_coverage') and round(comp['trial_coverage'], 3)} | {e.get('disposition', '')} |")
        for label, c in m['contrasts'].items():
            L.append(f"- {label}: {fmt(c['ratio'])}; numerator faster: {c['numerator_faster']}; pooled clock ok: {c['pooled_clock_ok']}")
        L.append('')
    L += ['## E4 service (mean over acquisitions; ratio to 304)', '']
    for exp, names in t['e4'].items():
        L.append(f'### {exp}')
        for name, lv in names.items():
            cells = ', '.join(f"{lvl}: {e['mean']:.1f}" + (f" ({e['ratio_to_304']:.3f})" if 'ratio_to_304' in e else '')
                              for lvl, e in lv.items())
            L.append(f"- {name} [{next(iter(lv.values()))['metric']}]: {cells}")
    L += ['', '## E5 / timeline', '']
    for p in t['e5_profiles']:
        L.append(f"- {p['step']}: completed={p.get('command_completed')} slowdown={p.get('instrumentation_slowdown')} "
                 f"outputs_match={p.get('output_matches_ordinary')} decode_ranges={p.get('decode_ranges')} "
                 f"dispatches/tick={p.get('dispatches_per_tick')} counter/tick={p.get('counter_per_tick')} "
                 f"counter coverage={p.get('counter_dispatch_coverage')}")
    L += ['', '## E8 guardrails', '']
    for k, e in t['e8'].items():
        L.append(f"- {k}: {e['complete']}/{e['acquisitions']} ms/step={[round(x, 4) for x in e['ms_per_step']]} "
                 f"prefill_ms={[round(x, 1) for x in e['prefill_ms']]} J/token={[round(x, 4) for x in e['joules_per_token']]} "
                 f"W={[round(x, 1) for x in e['mean_power_w']]}")
    return '\n'.join(L) + '\n'


def gate(args):
    plan, acqs, nat, ptab, comp_only, *_ = build(args.root)
    g = gates_from(plan, acqs, ptab, comp_only)
    g['source_root'] = str(args.root)
    args.out.write_text(json.dumps(g, indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in g.items() if k.endswith('_ok') or k.startswith('release')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('report'); r.add_argument('--root', type=Path, required=True); r.add_argument('--out', type=Path)
    g = sub.add_parser('gate'); g.add_argument('--root', type=Path, required=True); g.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report(a) if a.cmd == 'report' else gate(a)
