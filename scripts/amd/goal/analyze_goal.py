#!/usr/bin/env python3
"""goal_v1 reader (docs/amd/goal_based_followup_20261105.md; execution plan goal_followup_execution_plan_20261105.md).
CPU only. Builds on scripts/amd/cubw/analyze.py (acquisitions, clocks, placement, competitor windows, leak tests,
block-paired bootstrap) and adds the G1-G7 tables, the goal gates and goal_status.json.

  analyze_goal.py gate   --root <run root> --out <gates.json>
  analyze_goal.py report --root <run root> [--out <dir>]      -> goal_tables.json, goal_summary.md, goal_status.json

Pre-registered rules implemented here (frozen in the execution plan before data):
  release_g1  placement_ok
  release_g5  placement_ok and lifecycle_ok and isolation, where isolation = leak_ok_v4 (tuned decoder) if leak v4 is
              sensitive on every GPU/level; on gfx942 only, otherwise leak_ok_v2 (tuned decoder); else false
  G1 qualified noninferiority: upper 95% CI of R <= 1.05 AND clock_ok (pooled max/min <= 1.05 over the cell and its
              stock reference) AND placement_ok AND the cell's output digest equals the stock digest; else observational
  G1 clock sensitivity (not a substitute): per-trial log(ms/step) regressed on log(median in-window gfx MHz) with cell
              fixed effects over all G1 trials of the cohort; elasticity and clock-adjusted ratios reported as sensitivity
  G4 physical-HBM candidate: streaming reads (read1g_x4, read4g_x1) ratio within [0.9, 1.15] in one unit AND the
              MALL-resident read (read128m_x16) ratio <= 0.25 in that unit AND the L2-resident read (read8m_x256)
              ratio <= 0.25; a counter with streaming ~1 but MALL-resident ~1 is classified pre-MALL (request side)
"""
import argparse
import collections
import csv
import bisect
import json
import math
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cubw'))
import analyze as A  # noqa: E402

UNITS = ('bytes', 'KiB', 'req32', 'req64', 'req128')


def isolation(plan, g):
    v4 = g.get('leak_v4') or {}
    sens = bool(v4) and all(x.get('sensitive') for x in v4.values())
    if sens:
        return {'route': 'leak_v4', 'ok': bool(g.get('leak_ok_v4'))}
    if plan.get('arch') == 'gfx942' and g.get('leak_v2'):
        return {'route': 'leak_v2 (v4 insensitive somewhere)', 'ok': bool(g.get('leak_ok_v2'))}
    return {'route': 'none sensitive', 'ok': False}


def goal_gates(plan, acqs, ptab, comp_only):
    g = A.gates_from(plan, acqs, ptab, comp_only)
    iso = isolation(plan, g)
    g['goal_isolation'] = iso
    g['release_g1'] = bool(g['placement_ok'])
    g['release_g5'] = bool(g['placement_ok'] and g['lifecycle_ok'] and iso['ok'])
    return g


def gate(a):
    plan, acqs, nat, ptab, comp_only, *_ = A.build(a.root)
    g = goal_gates(plan, acqs, ptab, comp_only)
    g['source_root'] = str(a.root)
    a.out.write_text(json.dumps(g, indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in g.items() if k.endswith('_ok') or k.startswith('release') or k == 'goal_isolation'},
                     default=str))


# ------------------------------------------------------------------ helpers

def grid_of(step):
    for e in step.get('extra_env') or []:
        if e.startswith('CUBW_CU_COUNT_OVERRIDE='):
            return int(e.split('=')[1])
    return None


def blocks(acqs, key='ms_per_step'):
    return {a['block']: a[key] for a in acqs if a.get('disposition') == 'complete' and a.get(key)}


def ci(num, den):
    r = A.ratio_ci(num, den) if num and den else None
    return r


def clocks_of(acqs):
    g = [x for a in acqs for x in a.get('clock', {}).get('gfx', [])]
    m = [x for a in acqs for x in a.get('clock', {}).get('mem', [])]
    if not g:
        return None
    return {'gfx_median': st.median(g), 'gfx_min': min(g), 'gfx_max': max(g), 'mem_min': min(m), 'mem_max': max(m),
            'pooled_ok': A.clock_ok(acqs)}


def trial_gfx(a, samples):
    uuid = a.get('gpu_uuid')
    out = []
    for lo, hi in a.get('windows', []):
        w = A.window(samples, uuid, lo, hi)
        out.append(st.median(s[1] for s in w) if w else None)
    return out


# ------------------------------------------------------------------ G1

def g1(plan, acqs, steps, samples, gates):
    full = plan['full_cus']
    rows = [a for a in acqs if a['experiment'] == 'g1']
    for a in rows:
        a['grid'] = grid_of(steps[a['step']])
    stock = [a for a in rows if a['n'] == full and a['grid'] is None]
    first = [a for a in stock if a['cell'].endswith('__first')]
    last = [a for a in stock if a['cell'].endswith('__last')]
    den = {}
    for b in {a['block'] for a in stock if a.get('disposition') == 'complete'}:
        den[b] = [x for a in stock if a['block'] == b and a.get('disposition') == 'complete' for x in a['ms_per_step']]
    ref_digest = collections.Counter(a.get('output_sha256') for a in stock if a.get('disposition') == 'complete').most_common(1)
    ref_digest = ref_digest[0][0] if ref_digest else None
    out = {'stock_reference': 'mean of the stock full-device first and last acquisitions in the same block',
           'drift_last_over_first': ci(blocks(last), blocks(first)), 'cells': [], 'tuned_vs_stock_full': []}
    groups = collections.defaultdict(list)
    for a in rows:
        if not (a['n'] == full and a['grid'] is None):
            groups[(a['n'], a['grid'])].append(a)
    for (n, g), sel in sorted(groups.items(), key=lambda kv: (-kv[0][0], -(kv[0][1] or 10 ** 6))):
        r = ci(blocks(sel), den)
        clk = clocks_of(sel + stock)
        digests = sorted({str(a.get('output_sha256')) for a in sel if a.get('disposition') == 'complete'})
        same = digests == [str(ref_digest)]
        row = {'mask': n, 'grid': g if g is not None else 'stock', 'acquisitions': len(sel),
               'complete': sum(a.get('disposition') == 'complete' for a in sel),
               'ratio': r['ratio'] if r else None, 'ci95': r['ci95'] if r else None, 'blocks': r['units'] if r else 0,
               'mean_ms_per_step': st.mean(x for a in sel if a.get('ms_per_step') for x in a['ms_per_step']) if any(a.get('ms_per_step') for a in sel) else None,
               'clock': clk, 'outputs_match_stock': same, 'digests': digests,
               'b_cu_peak_gbps': None}
        qualified = bool(r and r['ci95'][1] <= 1.05 and clk and clk['pooled_ok'] and gates.get('placement_ok') and same)
        row['noninferior_5pct_qualified'] = qualified
        row['disposition'] = 'qualified' if qualified else ('observational_clock_confounded' if r else 'unavailable')
        out['cells'].append(row)
        if g is not None:
            out['tuned_vs_stock_full'].append({'mask': n, 'grid': g, 'ratio': row['ratio'], 'ci95': row['ci95'],
                                               'blocks': row['blocks'], 'clock': clk})
    # clock sensitivity: log T = cell effect + e * log f
    pts = []
    for a in rows:
        if a.get('disposition') != 'complete':
            continue
        key = 'stock_full' if (a['n'] == full and a['grid'] is None) else f"{a['n']}/{a['grid']}"
        for t, f in zip(a['ms_per_step'], trial_gfx(a, samples)):
            if f:
                pts.append((key, math.log(t), math.log(f)))
    if len(pts) > 10:
        means = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for k, y, x in pts:
            m = means[k]; m[0] += y; m[1] += x; m[2] += 1
        sxy = sum((x - means[k][1] / means[k][2]) * (y - means[k][0] / means[k][2]) for k, y, x in pts)
        sxx = sum((x - means[k][1] / means[k][2]) ** 2 for k, y, x in pts)
        e = sxy / sxx if sxx else None
        out['clock_sensitivity'] = {'elasticity_dlogT_dlogf': e, 'trials': len(pts), 'within_cell_gfx_log_var': sxx / len(pts),
                                    'note': 'pre-registered sensitivity only; not a qualification route'}
        if e is not None:
            fmed = {k: math.exp(means[k][1] / means[k][2]) for k in means}
            for row in out['cells']:
                k = f"{row['mask']}/{row['grid'] if row['grid'] != 'stock' else None}"
                if k in fmed and 'stock_full' in fmed and row['ratio']:
                    row['clock_adjusted_ratio_sensitivity'] = row['ratio'] * (fmed[k] / fmed['stock_full']) ** e
    return out


# ------------------------------------------------------------------ G2 / G3 isolated decode + energy

def conditions_rows(plan, acqs, steps, exp, case_of):
    cond = {(n, g): lab for lab, n, g in plan.get('conditions', [])}
    res = collections.defaultdict(dict)
    for a in acqs:
        if a['experiment'] != exp:
            continue
        lab = cond.get((a['n'], grid_of(steps[a['step']])))
        if lab:
            res[case_of(a)].setdefault(lab, []).append(a)
    return res


def g2_isolated(plan, acqs, steps):
    def case_of(a):
        return 'qwen32b_prefill' if a.get('generated_tokens') == 2 else a['case']
    res = conditions_rows(plan, acqs, steps, 'g2', case_of)
    out = {}
    for case, cells in sorted(res.items()):
        key = 'prefill_wall_ms' if case == 'qwen32b_prefill' else 'ms_per_step'
        base = blocks(cells.get('A', []), key)
        dA = {a.get('output_sha256') for a in cells.get('A', []) if a.get('disposition') == 'complete'}
        row = {}
        for lab, sel in sorted(cells.items()):
            r = ci(blocks(sel, key), base) if lab != 'A' else None
            d = {a.get('output_sha256') for a in sel if a.get('disposition') == 'complete'}
            row[lab] = {'metric': key, 'complete': sum(a.get('disposition') == 'complete' for a in sel), 'acquisitions': len(sel),
                        'mean': st.mean(x for a in sel if a.get(key) for x in a[key]) if any(a.get(key) for a in sel) else None,
                        'ratio_vs_A': r, 'outputs_match_A': d <= dA if d else None, 'clock': clocks_of(sel)}
        out[case] = row
    return out


def g3_energy(plan, acqs, steps):
    res = conditions_rows(plan, acqs, steps, 'g3_energy', lambda a: a['case'])
    out = {}
    for case, cells in res.items():
        row = {}
        def per_block(sel, f):
            d = {}
            for a in sel:
                if a.get('disposition') != 'complete' or not a.get('energy'):
                    continue
                vals = [f(a, i) for i in range(len(a['ms_per_step']))]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    d[a['block']] = vals
            return d
        jpt = lambda a, i: a['energy']['joules_per_token'][i]
        cj = lambda a, i: a['energy']['counter_j'][i]
        ws = lambda a, i: a['energy']['window_s'][i]
        pw = lambda a, i: (a['energy']['counter_j'][i] / a['energy']['window_s'][i]
                           if isinstance(a['energy']['counter_j'][i], (int, float)) and a['energy']['window_s'][i] else None)
        base = {k: per_block(cells.get('A', []), f) for k, f in (('jpt', jpt), ('p', pw), ('t', ws))}
        for lab, sel in sorted(cells.items()):
            cur = {k: per_block(sel, f) for k, f in (('jpt', jpt), ('p', pw), ('t', ws))}
            row[lab] = {'complete': sum(a.get('disposition') == 'complete' for a in sel),
                        'joules_per_token_mean': st.mean(x for v in cur['jpt'].values() for x in v) if cur['jpt'] else 'NA',
                        'avg_power_w_mean': st.mean(x for v in cur['p'].values() for x in v) if cur['p'] else 'NA',
                        'window_s_mean': st.mean(x for v in cur['t'].values() for x in v) if cur['t'] else 'NA',
                        'agreement_max': max((x for a in sel if a.get('energy') for x in a['energy']['agreement']
                                              if isinstance(x, (int, float))), default='NA'),
                        'jpt_ratio_vs_A': ci(cur['jpt'], base['jpt']) if lab != 'A' else None,
                        'power_ratio_vs_A': ci(cur['p'], base['p']) if lab != 'A' else None,
                        'time_ratio_vs_A': ci(cur['t'], base['t']) if lab != 'A' else None,
                        'scope': 'decode-only window, board energy accumulator; primary J/token NA when sensor checks fail'}
        out[case] = row
    return out


# ------------------------------------------------------------------ serving

def serve_rows(plan, attempts, samples):
    cond = {(n, g): lab for lab, n, g in plan.get('conditions', [])}
    rows = []
    for s in plan['steps']:
        if s['kind'] != 'serve':
            continue
        rec = A.final_attempt(attempts.get(s['id'], []))
        r = {'step': s['id'], 'experiment': s['experiment'], 'block': s.get('block'), 'lane': s['lane'], 'mode': s['mode'],
             'n': s['decoder_cus'], 'grid': grid_of(s), 'label': cond.get((s['decoder_cus'], grid_of(s))),
             'concurrency': int(s['serve_args'][1])}
        if not rec:
            r['disposition'] = 'not_run'
            rows.append(r)
            continue
        p = Path(rec['attempt_dir']) / 'run' / 'run.json'
        d = json.loads(p.read_text()) if p.exists() else {}
        r['disposition'] = 'complete' if d.get('status') == 'complete' and rec.get('command_completed') else 'failed'
        r['mask_recorded'] = d.get('environment', {}).get('HSA_CU_MASK')
        if d.get('trials'):
            T = d['trials']
            r['trials'] = len(T)
            r['tok_s'] = [t['output_tokens_per_s'] for t in T]
            r['ttft_p50'] = [t['ttft_ms']['p50'] for t in T]; r['ttft_p99'] = [t['ttft_ms']['p99'] for t in T]
            r['itl_p50'] = [t['itl_ms']['p50'] for t in T]; r['itl_p99'] = [t['itl_ms']['p99'] for t in T]
            r['e2e_p99'] = [t['e2e_ms']['p99'] for t in T]
            r['output_agreement'] = d.get('output_agreement')
            r['digests_trial0'] = [q['output_sha256'] for q in T[0]['per_request']]
            if s['mode'] == 'energy':
                r['j_per_tok'] = [t.get('joules_per_output_token') for t in T]
                r['power_w'] = [t['telemetry']['counter_j'] / t['makespan_s'] if isinstance(t.get('telemetry', {}).get('counter_j'), (int, float)) else None for t in T]
            gfx = []
            for t in T:
                w = A.window(samples, d['hardware']['uuid'], t['wall_start_unix_ns'], t['wall_end_unix_ns'])
                gfx += [x[1] for x in w]
            r['gfx_min'], r['gfx_max'] = (min(gfx), max(gfx)) if gfx else (None, None)
        rows.append(r)
    return rows


def serve_tables(rows):
    out = {}
    for exp in ('g2_serve', 'g3_serve_energy'):
        for conc in sorted({r['concurrency'] for r in rows if r['experiment'] == exp}):
            sel = [r for r in rows if r['experiment'] == exp and r['concurrency'] == conc and r.get('disposition') == 'complete']
            by = collections.defaultdict(list)
            for r in sel:
                by[r['label']].append(r)
            metrics = ['tok_s', 'ttft_p50', 'ttft_p99', 'itl_p50', 'itl_p99', 'e2e_p99'] + (['j_per_tok', 'power_w'] if exp == 'g3_serve_energy' else [])
            tab = {}
            for lab, rs in sorted(by.items()):
                tab[lab] = {'acquisitions': len(rs)}
                for m in metrics:
                    vals = {r['block']: [x for x in r.get(m, []) if isinstance(x, (int, float))] for r in rs}
                    vals = {k: v for k, v in vals.items() if v}
                    base = {r['block']: [x for x in r.get(m, []) if isinstance(x, (int, float))] for r in by.get('A', [])}
                    base = {k: v for k, v in base.items() if v}
                    tab[lab][m] = {'mean': st.mean(x for v in vals.values() for x in v) if vals else None,
                                   'ratio_vs_A': ci(vals, base) if lab != 'A' else None}
                ref = {r['block']: r['digests_trial0'] for r in by.get('A', []) if r.get('digests_trial0')}
                agree = [sum(x == y for x, y in zip(r['digests_trial0'], ref[r['block']])) / len(r['digests_trial0'])
                         for r in rs if r.get('digests_trial0') and r['block'] in ref]
                tab[lab]['request_output_agreement_with_A'] = st.mean(agree) if agree else None
                tab[lab]['within_cell_agreement'] = [r['output_agreement'] for r in rs]
                g = [x for r in rs for x in (r.get('gfx_min'), r.get('gfx_max')) if x]
                tab[lab]['gfx_range'] = [min(g), max(g)] if g else None
            out[f'{exp}:c{conc}'] = tab
    return out


# ------------------------------------------------------------------ G4 / G6 / G7 / routes

def counter_audit(plan, attempts):
    s = next((s for s in plan['steps'] if s['experiment'] == 'g4_counters'), None)
    rec = A.final_attempt(attempts.get(s['id'], [])) if s else None
    f = Path(rec['attempt_dir']) / 'counter_audit.json' if rec else None
    if not f or not f.exists():
        return {'status': 'unavailable', 'reason': 'no audit output', 'returncode': rec.get('returncode') if rec else None}
    d = json.loads(f.read_text())
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    fails = collections.Counter()
    for r in d['runs']:
        if r.get('ratio'):
            for u in UNITS:
                per[r['counter']][(r['load'], u)].append(r['ratio'][u])
        else:
            fails[r['counter']] += 1
    table, classes = {}, {}
    for c, m in per.items():
        med = {f'{l}|{u}': st.median(v) for (l, u), v in m.items()}
        spread = {f'{l}|{u}': (max(v) - min(v)) / st.median(v) if len(v) > 1 and st.median(v) else 0 for (l, u), v in m.items()}
        table[c] = {'median_ratio': med, 'rep_spread': spread, 'failed_runs': fails[c]}
        cls = 'unclassified'
        for u in UNITS:
            sr = [med.get(f'{l}|{u}') for l in ('read1g_x4', 'read4g_x1')]
            mall, l2 = med.get(f'read128m_x16|{u}'), med.get(f'read8m_x256|{u}')
            if all(x is not None and 0.9 <= x <= 1.15 for x in sr):
                if mall is not None and l2 is not None and mall <= 0.25 and l2 <= 0.25:
                    cls = f'physical-HBM read candidate ({u})'
                elif mall is not None and mall > 0.5:
                    cls = f'pre-MALL request-side read counter ({u})'
                break
        if cls == 'unclassified':
            for u in UNITS:
                w = [med.get(f'{l}|{u}') for l in ('write4g',)]
                if all(x is not None and 0.9 <= x <= 1.15 for x in w):
                    cls = f'write-traffic counter, streaming-consistent ({u}); MALL behaviour of writes not separable here'
                    break
        classes[c] = cls
    return {'status': 'measured', 'inventory': {k: v[:3000] if isinstance(v, str) else v for k, v in d['inventory'].items()},
            'candidates_found': d['candidates_found'], 'counters_tested': d['counters_tested'],
            'never_produced_values': {c: n for c, n in fails.items() if c not in per}, 'table': table, 'classification': classes,
            'physical_hbm_read_counter': sorted(c for c, v in classes.items() if v.startswith('physical'))}


def shapes(plan, attempts):
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for s in plan['steps']:
        if s['experiment'] != 'g6_shapes':
            continue
        rec = A.final_attempt(attempts.get(s['id'], []))
        f = Path(rec['attempt_dir']) / 'e6.json' if rec else None
        if not f or not f.exists():
            continue
        d = json.loads(f.read_text())
        for name, row in d['shapes'].items():
            for vn, v in row['variants'].items():
                if 'median_us' in v:
                    out[s['level']][name][vn].append(v['median_us'])
    tab = {}
    for n, sh in sorted(out.items(), reverse=True):
        tab[n] = {}
        for name, vs in sh.items():
            means = {vn: st.mean(v) for vn, v in vs.items()}
            best = min((v, vn) for vn, v in means.items() if vn.startswith('wvsplitk'))
            tab[n][name] = {'mean_us': means, 'acquisitions': {vn: len(v) for vn, v in vs.items()}, 'best_wvsplitk': best[1],
                            'best_us': best[0]}
    return tab


def natives(plan, attempts, exp):
    out = {}
    for s in plan['steps']:
        if s['experiment'] != exp:
            continue
        rec = A.final_attempt(attempts.get(s['id'], []))
        if not rec:
            continue
        lines = (Path(rec['attempt_dir']) / 'out.json').read_text().strip().splitlines()
        try:
            res = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:
            res = None
        if not res:
            continue
        if exp == 'g6_dispatch' or s['experiment'] == 'e6_dispatch':
            out[s['cell']] = A.dispatch_summary(res)
        else:
            k, v = A.metric(res)
            out[s['cell']] = {'level': s.get('level'), 'metric': k, 'value': v}
    return out


def routes(plan, attempts):
    """Kernel routes from kernel-trace timelines: dominant kernels inside prefill and decode ranges, with grid sizes."""
    cond = {(n, g): lab for lab, n, g in plan.get('conditions', [])}
    out = {}
    for s in plan['steps']:
        if s['experiment'] != 'g2_route':
            continue
        rec = A.final_attempt(attempts.get(s['id'], []))
        if not rec:
            continue
        d = Path(rec['attempt_dir']) / 'rocprof'
        try:
            mk = next(d.rglob('*marker_api_trace.csv')); kt = next(d.rglob('*kernel_trace.csv'))
        except StopIteration:
            out[s['cell']] = {'status': 'no trace files'}
            continue
        ranges = []
        for r in csv.DictReader(mk.open()):
            if r['Function'] in ('prefill', 'decode'):
                ranges.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Function']))
        ranges.sort()
        starts = [x[0] for x in ranges]
        agg = collections.defaultdict(lambda: collections.defaultdict(lambda: {'n': 0, 'ns': 0, 'grids': collections.Counter()}))
        for r in csv.DictReader(kt.open()):
            t0 = int(r['Start_Timestamp'])
            i = bisect.bisect_right(starts, t0) - 1
            if i < 0 or t0 > ranges[i][1]:
                continue
            k = r['Kernel_Name'].split('(')[0][:100]
            v = agg[ranges[i][2]][k]
            v['n'] += 1; v['ns'] += int(r['End_Timestamp']) - t0
            v['grids'][f"{r['Grid_Size_X']}x{r['Workgroup_Size_X']}"] += 1
        phase = {}
        for ph, ks in agg.items():
            tot = sum(v['ns'] for v in ks.values()) or 1
            top = sorted(ks.items(), key=lambda kv: -kv[1]['ns'])[:8]
            phase[ph] = [{'kernel': k, 'share': v['ns'] / tot, 'calls': v['n'], 'grids': dict(v['grids'].most_common(3))} for k, v in top]
            phase[ph + '_skinny_share'] = sum(v['ns'] for k, v in ks.items() if 'wvSplitK' in k or 'LLMM1' in k) / tot
        out[s['cell']] = {'case': s['case'], 'tokens': s.get('generated_tokens'), 'label': cond.get((s['decoder_cus'], grid_of(s))),
                          'phases': phase}
    # does the override reach a kernel? compare skinny-kernel grid sizes between A and C per case
    for case in {v['case'] for v in out.values() if 'case' in v}:
        for tok in {v['tokens'] for v in out.values() if v.get('case') == case}:
            pair = {v['label']: v for v in out.values() if v.get('case') == case and v['tokens'] == tok}
            if 'A' in pair and 'C' in pair:
                def sk(v):
                    return {k['kernel']: list(k['grids']) for ph in ('prefill', 'decode') for k in v['phases'].get(ph, [])
                            if 'wvSplitK' in k['kernel']}
                a_, c_ = sk(pair['A']), sk(pair['C'])
                pair['C']['override_reaches_kernel'] = bool(c_) and a_ != c_
    return out


def g7_logical(plan):
    """Logical per-token decode demand of Qwen32B B1/C4096 (BF16): weights + KV read + KV write, from the model config."""
    cfg = json.loads((Path(plan['config_source']) / 'configs/qwen32b_graph_b1_c4096.json').read_text())
    mc = json.loads((Path(cfg['model_path']) / 'config.json').read_text())
    h, L, i, v = mc['hidden_size'], mc['num_hidden_layers'], mc['intermediate_size'], mc['vocab_size']
    nh, kv = mc['num_attention_heads'], mc['num_key_value_heads']
    hd = h // nh
    per_layer = h * (h + 2 * kv * hd) + h * h + 3 * h * i
    weights = 2 * (L * per_layer + v * h)  # embeddings gathered (one row) not counted; lm_head streamed
    ctx = 4096
    kv_read = 2 * L * 2 * kv * hd * ctx
    kv_write = 2 * L * 2 * kv * hd
    return {'weights_bytes_per_token': weights, 'kv_read_bytes_per_token_at_4096': kv_read, 'kv_write_bytes_per_token': kv_write,
            'compulsory_read_bytes_per_token': weights + kv_read, 'scope': 'logical demand, BF16; activations and norms '
            'excluded (< 1 MB); no reuse within a token is assumed or excluded'}


def g7_counters(plan, attempts):
    out = {}
    for s in plan['steps']:
        if s['experiment'] != 'g7_counters':
            continue
        rec = A.final_attempt(attempts.get(s['id'], []))
        row = {'counter': s['profile']['counters'][0], 'n': s['decoder_cus'], 'grid': grid_of(s),
               'returncode': rec.get('returncode') if rec else None}
        if rec:
            d = Path(rec['attempt_dir']) / 'rocprof'
            mk = next(d.rglob('*marker_api_trace.csv'), None); cc = next(d.rglob('*counter_collection.csv'), None)
            run = Path(rec['attempt_dir']) / 'run' / 'run.json'
            if mk and cc and run.exists():
                rj = json.loads(run.read_text())
                dec = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in csv.DictReader(mk.open()) if r['Function'] == 'decode')
                per_range = [0.0] * len(dec)
                rows = 0
                for r in csv.DictReader(cc.open()):
                    t = int(r.get('Start_Timestamp') or r.get('Begin_Timestamp') or 0)
                    i = next((i for i, (lo, hi) in enumerate(dec) if lo <= t <= hi), None)
                    if i is not None:
                        per_range[i] += float(r['Counter_Value']); rows += 1
                # every decode marker range (warmups and trials alike) generates the same token count, so each range
                # is one sample of per-tick traffic; summing ranges and dividing by one trial's ticks overstated it
                its = {t['decode_iterations'] for c in rj.get('cases', []) for t in c['trials']}
                ticks = its.pop() if len(its) == 1 else None
                ptk = [v / ticks for v in per_range] if ticks else []
                row.update(decode_total=sum(per_range), dispatch_rows=rows, decode_ranges=len(dec), ticks_per_range=ticks,
                           per_tick_by_range=ptk, per_tick=st.median(ptk) if ptk else None,
                           note='kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over '
                                'decode marker ranges (warmups and trials) of range total / decode iterations')
        out[s['cell']] = row
    return out


def g5_surface(plan, acqs, steps):
    sel = [a for a in acqs if a['experiment'] == 'g5']
    out = {}
    for n in sorted({a['n'] for a in sel}, reverse=True):
        cells = {a['treatment']: [] for a in sel if a['n'] == n}
        for a in sel:
            if a['n'] == n:
                cells[a['treatment']].append(a)
        base = blocks(cells.get('absent', []))
        row = {}
        for t, xs in cells.items():
            comp = [a['competitor'] for a in xs if a.get('competitor')]
            row[t] = {'complete': sum(a.get('disposition') == 'complete' for a in xs),
                      'ratio_vs_absent': ci(blocks(xs), base) if t != 'absent' else None,
                      'competitor_gb_s': st.mean(c['gb_per_s_during_trials'] for c in comp if c.get('gb_per_s_during_trials')) if any(c.get('gb_per_s_during_trials') for c in comp) else None,
                      'competitor_tflop_s': st.mean(c['tflop_per_s_during_trials'] for c in comp if c.get('tflop_per_s_during_trials')) if any(c.get('tflop_per_s_during_trials') for c in comp) else None,
                      'coverage_min': min((c['trial_coverage'] for c in comp), default=None),
                      'lifecycle_ok': all(a.get('lifecycle_ok') for a in xs if a.get('competitor')) if comp else None,
                      'clock': clocks_of(xs)}
        out[n] = row
    for n in out:
        if 'mem100' in out[n] and 'arith100' in out[n]:
            ab = blocks([a for a in sel if a['n'] == n and a['treatment'] == 'arith100'])
            out[n]['mem100_over_arith100'] = ci(blocks([a for a in sel if a['n'] == n and a['treatment'] == 'mem100']), ab)
    return out


def report(a):
    root = a.root
    out = a.out or root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    plan, acqs, nat, ptab, comp_only, attempts, samples = A.build(root)
    steps = {s['id']: s for s in plan['steps']}
    gates = goal_gates(plan, acqs, ptab, comp_only)
    srows = serve_rows(plan, attempts, samples)
    t = {'version': 'goal_v1', 'cohort': plan['cohort'], 'plan_kind': plan['kind'], 'plan_sha256': plan.get('plan_sha256'),
         'gates': gates, 'placement': ptab,
         'g1': g1(plan, acqs, steps, samples, gates), 'g2_isolated': g2_isolated(plan, acqs, steps),
         'g2_routes': routes(plan, attempts), 'serving': serve_tables(srows), 'serve_rows': srows,
         'g3_energy': g3_energy(plan, acqs, steps), 'g4_counters': counter_audit(plan, attempts),
         'g5': g5_surface(plan, acqs, steps), 'g6_shapes': shapes(plan, attempts),
         'g6_dispatch': natives(plan, attempts, 'e6_dispatch'), 'g6_service': natives(plan, attempts, 'g6_service'),
         'g7_logical': g7_logical(plan), 'g7_counters': g7_counters(plan, attempts),
         'acquisitions': [{k: v for k, v in x.items() if k not in ('clock', 'windows')} for x in acqs]}
    isa = next((s for s in plan['steps'] if s['experiment'] == 'g6_isa'), None)
    rec = A.final_attempt(attempts.get(isa['id'], [])) if isa else None
    f = Path(rec['attempt_dir']) / 'isa' / 'isa_meta.json' if rec else None
    t['g6_isa'] = json.loads(f.read_text()) if f and f.exists() else {'status': 'unavailable'}
    (out / 'goal_tables.json').write_text(json.dumps(t, indent=1, default=str) + '\n')
    (out / 'goal_status.json').write_text(json.dumps(status(t), indent=1, default=str) + '\n')
    (out / 'goal_summary.md').write_text(summary_md(t))
    print(f"goal tables written to {out}")


def status(t):
    g = t['gates']
    g1c = t['g1']['cells']
    return {'cohort': t['cohort'], 'plan_kind': t['plan_kind'],
            'G1': {'cells': len(g1c), 'qualified': sum(c['noninferior_5pct_qualified'] for c in g1c),
                   'clock_ok_any': any((c.get('clock') or {}).get('pooled_ok') for c in g1c),
                   'controlled_clocks': 'unavailable: not authorized'},
            'G2': {'isolated_cases': len(t['g2_isolated']), 'serving_tables': list(t['serving']),
                   'routes_traced': len(t['g2_routes'])},
            'G3': {'decode_energy_cases': len(t['g3_energy']), 'serving_energy': [k for k in t['serving'] if 'energy' in k]},
            'G4': {'status': t['g4_counters'].get('status'), 'physical_hbm_read_counter': t['g4_counters'].get('physical_hbm_read_counter')},
            'G5': {'released': g.get('release_g5'), 'isolation': g.get('goal_isolation'), 'levels': list(t['g5'])},
            'G6': {'shape_masks': list(t['g6_shapes']), 'dispatch_cells': len(t['g6_dispatch']), 'isa': t['g6_isa'].get('kernel_count')},
            'G7': {'counter_cells': len(t['g7_counters'])}}


def fmt(r):
    if not r:
        return 'n/a'
    return f"{r['ratio']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] ({r['units']} blocks)"


def summary_md(t):
    L = [f"# goal_v1 — {t['cohort']} ({t['plan_kind']})", '', '## Gates', '']
    for k in ('placement_ok', 'lifecycle_ok', 'outputs_ok', 'leak_ok_v4', 'leak_ok_v2', 'clock_ok', 'release_g1', 'release_g5'):
        L.append(f"- {k}: **{t['gates'].get(k)}**")
    L.append(f"- isolation route: {t['gates'].get('goal_isolation')}")
    L += ['', '## G1 restriction curve (Qwen32B graph B1/C4096; R vs stock full device, same block)', '',
          f"- drift stock last/first: {fmt(t['g1'].get('drift_last_over_first'))}",
          f"- clock sensitivity: {t['g1'].get('clock_sensitivity')}", '',
          '| mask | grid | R [95% CI] | gfx MHz min-max | clock_ok | outputs = stock | disposition | clock-adjusted (sensitivity) |',
          '|---:|---|---|---|---|---|---|---|']
    for c in t['g1']['cells']:
        clk = c.get('clock') or {}
        L.append(f"| {c['mask']} | {c['grid']} | {fmt({'ratio': c['ratio'], 'ci95': c['ci95'], 'units': c['blocks']}) if c['ratio'] else 'n/a'} | "
                 f"{clk.get('gfx_min')}-{clk.get('gfx_max')} | {clk.get('pooled_ok')} | {c['outputs_match_stock']} | {c['disposition']} | "
                 f"{c.get('clock_adjusted_ratio_sensitivity', 'n/a') if not isinstance(c.get('clock_adjusted_ratio_sensitivity'), float) else round(c['clock_adjusted_ratio_sensitivity'], 4)} |")
    L += ['', '## G2 isolated phases (ratio vs A = full mask, stock kernel)', '', '| case | B full/reduced grid | C reduced/grid | D reduced/stock | Cb boundary |', '|---|---|---|---|---|']
    for case, row in t['g2_isolated'].items():
        L.append(f"| {case} | " + ' | '.join(fmt(row.get(l, {}).get('ratio_vs_A')) for l in ('B', 'C', 'D', 'Cb')) + ' |')
    L += ['', '## G2 kernel routes (does the CU-count override reach a kernel?)', '']
    for cellname, v in t['g2_routes'].items():
        if 'override_reaches_kernel' in v:
            L.append(f"- {v['case']} tokens={v['tokens']}: override reaches a kernel: **{v['override_reaches_kernel']}**; "
                     f"skinny share decode {v['phases'].get('decode_skinny_share')}, prefill {v['phases'].get('prefill_skinny_share')}")
    L += ['', '## Mixed serving (ratio vs A)', '']
    for key, tab in t['serving'].items():
        L += [f'### {key}', '', '| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |', '|---|---|---|---|---|---|---|---|---|']
        for lab, r in tab.items():
            cells = []
            for m in ('tok_s', 'ttft_p50', 'ttft_p99', 'itl_p50', 'itl_p99', 'e2e_p99', 'j_per_tok'):
                v = r.get(m)
                cells.append('—' if not v else (f"{v['mean']:.3g}" if lab == 'A' or not v.get('ratio_vs_A') else f"{v['ratio_vs_A']['ratio']:.3f} [{v['ratio_vs_A']['ci95'][0]:.3f}, {v['ratio_vs_A']['ci95'][1]:.3f}]"))
            L.append(f"| {lab} | " + ' | '.join(cells) + f" | {r.get('request_output_agreement_with_A')} |")
        L.append('')
    L += ['## G3 decode energy (board accumulator, decode window)', '']
    for case, row in t['g3_energy'].items():
        for lab, r in row.items():
            L.append(f"- {case} {lab}: J/token {r['joules_per_token_mean']}, P {r['avg_power_w_mean']} W, window {r['window_s_mean']} s; "
                     f"J/tok ratio {fmt(r['jpt_ratio_vs_A'])}, P ratio {fmt(r['power_ratio_vs_A'])}, T ratio {fmt(r['time_ratio_vs_A'])}")
    g4 = t['g4_counters']
    L += ['', '## G4 counter audit', '', f"- status: {g4.get('status')}; physical-HBM read counter: {g4.get('physical_hbm_read_counter')}"]
    for c, cls in (g4.get('classification') or {}).items():
        L.append(f"- {c}: {cls}")
    L += ['', '## G5 tuned decoder x memory service (ratio vs absent, same block)', '']
    for n, row in t['g5'].items():
        L.append(f"- n={n}: " + '; '.join(f"{k} {fmt(v.get('ratio_vs_absent')) if isinstance(v, dict) and 'ratio_vs_absent' in v else fmt(v)}"
                                         + (f" ({v['competitor_gb_s']:.0f} GB/s)" if isinstance(v, dict) and v.get('competitor_gb_s') else '')
                                         for k, v in row.items()))
    L += ['', '## G6 kernel-shape sweep (best wvSplitK grid per mask)', '']
    for n, sh in t['g6_shapes'].items():
        L.append(f"- mask {n}: " + '; '.join(f"{k} {v['best_wvsplitk']} {v['best_us']:.1f} us" for k, v in sh.items()))
    L += ['', '## G7', '', f"- logical: {t['g7_logical']}"]
    for k, v in t['g7_counters'].items():
        L.append(f"- {k}: {v}")
    L += ['', 'Observational unless a G1 row says qualified. AMD acknowledgement: "This work was supported in part by Advanced '
          'Micro Devices, Inc. under the AMD University Program\'s AI & HPC Cluster".', '']
    return '\n'.join(L)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate'); g.add_argument('--root', type=Path, required=True); g.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('report'); r.add_argument('--root', type=Path, required=True); r.add_argument('--out', type=Path)
    a = p.parse_args()
    gate(a) if a.cmd == 'gate' else report(a)
