#!/usr/bin/env python3
"""Chunked-prefill campaign analysis (rules: docs/amd/cp_campaign_design_20261118.md). CPU only; run on a devel node.

  cp_analyze.py s1 --root ROOT [--out DIR]     experiment 1 headroom, profile attribution, experiment 3 step level,
                                               SLO and capacity pilot, experiment 4, KV link
  cp_analyze.py serve --root ROOT [--slo slo.json] [--out DIR]   experiment 2 / 3 serving tables
  cp_analyze.py disagg --root ROOT --slo slo.json [--out DIR]    experiment 5 tables
"""
import argparse
import collections
import csv
import glob
import json
import math
from pathlib import Path
import random
import statistics as st

BOOT, SEED = 10000, 20261118
HEAD = 1.05
GRAPH_MAX = 512


def jl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def journal(root):
    out = {}
    for f in sorted((root / 'journal').glob('*.jsonl')):
        for r in jl(f):
            if 'step' in r and 'attempt' in r:
                out.setdefault(r['step'], []).append(r)
    return out


def final(recs):
    ok = [r for r in recs if r.get('command_completed')]
    return ok[-1] if ok else (recs[-1] if recs else None)


def run_dir(rec):
    return Path(rec['attempt_dir']) / 'run'


def boot_ci(xs, stat=st.median, n=BOOT, seed=SEED):
    if len(xs) < 2:
        return (None, None)
    r = random.Random(seed)
    v = sorted(stat([xs[r.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return (v[int(0.025 * n)], v[int(0.975 * n) - 1])


def graph_path(kind, tokens, sizes):
    if tokens > (max(sizes) if sizes else GRAPH_MAX):
        return 'eager', tokens
    pad = min(s for s in sizes if s >= tokens)
    return ('full' if kind == 'decode' else 'piecewise'), pad


# ---------------------------------------------------------------- experiment 1
def e1_tables(run):
    """run = attempt run dir of cp_replay.py (no --prefix). Returns per-(S,N,C) rows and headroom per (S,N)."""
    meta = json.loads((run / 'run.json').read_text())
    sizes = meta.get('engine_info', {}).get('cudagraph_capture_sizes') or []
    rows = [r for r in jl(run / 'steps.jsonl') if r.get('role') and r.get('rep', -1) >= 0]
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[(r['S'], r['N'], r['C'], r['rep'])][r['role']] = r
    cells = collections.defaultdict(list)
    anchors = collections.defaultdict(list)
    for (S, N, Cn, rep), d in pairs.items():
        if N == 0:
            if 'mix' in d:
                anchors[Cn].append(d['mix']['exec_s'])
            continue
        if 'mix' in d and 'ref' in d:
            cells[(S, N, Cn)].append((d['mix']['exec_s'] / d['ref']['exec_s'], d['mix'], d['ref']))
    out = []
    for (S, N, Cn), v in sorted(cells.items()):
        ratios = [x[0] for x in v]
        lo, hi = boot_ci(ratios)
        mix = v[0][1]
        gp, pad = graph_path(mix['kind'], mix['total_tokens'], sizes)
        out.append({'S': S, 'N': N, 'C': Cn, 'pairs': len(v), 'ratio_median': st.median(ratios), 'ci95': [lo, hi],
                    't_mix_ms': 1e3 * st.median(x[1]['exec_s'] for x in v), 't_ref_ms': 1e3 * st.median(x[2]['exec_s'] for x in v),
                    'wall_mix_ms': 1e3 * st.median(x[1]['wall_s'] for x in v), 'sched_mix_ms': 1e3 * st.median(x[1]['sched_s'] for x in v),
                    'tokens': mix['total_tokens'], 'graph_path': gp, 'padded_tokens': pad,
                    'ctx_drift_tokens': max(x[2]['decode_ctx_max'] for x in v) - min(x[2]['decode_ctx_max'] for x in v),
                    'within': hi is not None and hi <= HEAD})
    head = {}
    for S, N in sorted({(r['S'], r['N']) for r in out}):
        rs = sorted((r for r in out if r['S'] == S and r['N'] == N and r['C'] > 0), key=lambda r: r['C'])
        h, first_fail, nonmono = 0, None, []
        for r in rs:
            if r['within'] and first_fail is None:
                h = r['C']
            elif not r['within'] and first_fail is None:
                first_fail = r['C']
            elif r['within'] and first_fail is not None:
                nonmono.append(r['C'])
        ref = st.median([r['t_ref_ms'] for r in out if r['S'] == S and r['N'] == N])
        head[f'S{S}_N{N}'] = {'S': S, 'N': N, 'H': h, 'boundary': [h, first_fail] if first_fail else [h, None],
                              'open': first_fail is None, 'nonmonotonic_pass_after_fail': nonmono, 't_ref_ms': ref}
    anc = {Cn: 1e3 * st.median(v) for Cn, v in sorted(anchors.items())}
    return {'cells': out, 'headroom': head, 'anchors_ms': anc, 'capture_sizes': sizes,
            'composition_mismatches': meta.get('composition_mismatches'), 'status': meta.get('status')}


# ---------------------------------------------------------------- experiment 3 step level
def e3_tables(run):
    rows = jl(run / 'steps.jsonl')
    pairs = collections.defaultdict(dict)
    checks = []
    warm = collections.defaultdict(list)
    for r in rows:
        if r.get('event') == 'prefix_check' and r['rep'] >= 0:
            checks.append(r)
        if r.get('event') == 'warm_build' and r['rep'] >= 0:
            warm[(r['S'], r['N'], r['P'])].append(r['s'])
        if r.get('role') and r.get('rep', -1) >= 0:
            pairs[(r['S'], r['N'], r['P'], r['treatment'], r['rep'])][r['role']] = r
    cells = collections.defaultdict(list)
    for (S, N, P, t, rep), d in pairs.items():
        if 'mix' in d and 'ref' in d:
            cells[(S, N, P, t)].append((d['mix']['exec_s'], d['ref']['exec_s'], d['mix']))
    out = []
    for (S, N, P, t), v in sorted(cells.items()):
        ratios = [m / r for m, r, _ in v]
        out.append({'S': S, 'N': N, 'P': P, 'treatment': t, 'pairs': len(v), 't_mix_ms': 1e3 * st.median(x[0] for x in v),
                    't_ref_ms': 1e3 * st.median(x[1] for x in v), 'ratio_median': st.median(ratios), 'ci95': boot_ci(ratios),
                    'executed_tokens': st.median(x[2]['prefill'][0][2] for x in v if x[2].get('prefill')),
                    'warm_build_ms': 1e3 * st.median(warm[(S, N, P)]) if warm.get((S, N, P)) else None})
    ok = [c['ok'] for c in checks]
    return {'cells': out, 'checks': {'n': len(ok), 'ok': sum(ok),
                                     'warm_cached_exact': sum(1 for c in checks if c['treatment'] == 'warm' and c['cached_at_admit'] == c['P']),
                                     'warm_n': sum(1 for c in checks if c['treatment'] == 'warm')}}


# ---------------------------------------------------------------- serving traces (experiments 2, 3; pilot)
def p95(xs):
    if not xs:
        return None
    xs = sorted(xs)
    k = 0.95 * (len(xs) - 1)
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = q * (len(xs) - 1)
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def trace_summary(t, slo=None):
    reqs = [r for r in t['requests'] if r['window'] == 'measured']
    done = [r for r in reqs if r['complete']]
    span = (max(r['arrival_s'] for r in reqs) - min(r['arrival_s'] for r in reqs)) if len(reqs) > 1 else None
    out = {'requests': len(reqs), 'complete': len(done), 'incomplete': len(reqs) - len(done),
           'ttft_p50_s': pct([r['ttft_s'] for r in done], .5), 'ttft_p90_s': pct([r['ttft_s'] for r in done], .9),
           'ttft_p99_s': pct([r['ttft_s'] for r in done], .99),
           'itl_p50_ms': pct([x for r in done for x in r['itl_ms']], .5), 'itl_p99_ms': pct([x for r in done for x in r['itl_ms']], .99),
           'req_itl_p95_median_ms': st.median([p95(r['itl_ms']) for r in done if r['itl_ms']]) if done else None,
           # exploratory (not the pre-registered compliance rule): the longest stall each request saw
           'req_itl_max_median_ms': st.median([max(r['itl_ms']) for r in done if r['itl_ms']]) if done else None,
           'req_itl_max_p99_ms': pct([max(r['itl_ms']) for r in done if r['itl_ms']], .99),
           'itl_mean_ms': st.mean([x for r in done for x in r['itl_ms']]) if done else None,
           'window_s': span}
    first = min(r['arrival_s'] for r in t['requests'])
    lastdone = max((r.get('done_s') or 0) for r in t['requests'])
    out['output_tokens_per_s'] = sum(r['tokens'] for r in t['requests']) / max(1e-9, lastdone - first)
    steps = t.get('step_records') or []
    if steps:
        by = collections.Counter(s['kind'] for s in steps)
        tt = collections.Counter()
        for s in steps:
            tt[s['kind']] += s.get('exec_s') or 0
        tot = sum(tt.values()) or 1
        out['step_fraction_count'] = {k: v / len(steps) for k, v in by.items()}
        out['step_fraction_time'] = {k: v / tot for k, v in tt.items()}
        pf = sum(c[2] for s in steps for c in s['prefill'])
        dec = sum(s['n_decode'] for s in steps)
        out['effective_p_to_d'] = pf / dec if dec else None
        out['preemptions'] = sum(s.get('preempted', 0) for s in steps)
        out['max_waiting'] = max(s['n_waiting'] for s in steps)
        out['recomputed_tokens'] = sum(c[2] for s in steps for c in s['prefill'] if c[5])
    tel = t.get('telemetry')
    if isinstance(tel, dict) and isinstance(tel.get('counter_j'), (int, float)):
        tok = sum(r['tokens'] for r in t['requests'])
        out['joules_per_output_token'] = tel['counter_j'] / tok if tok else None
        out['energy_window_s'] = tel.get('window_s')
    if slo and span:
        ok = [r for r in done if r['ttft_s'] <= slo['ttft_s'] and (not r['itl_ms'] or p95(r['itl_ms']) <= slo['itl_ms'])]
        out['attainment'] = len(ok) / len(reqs)
        out['goodput_rps'] = len(ok) / span
        out['goodput_tok_s'] = sum(r['tokens'] for r in ok) / span
        # lc amendment 3: stall-aware secondary SLO, the primary SLO plus a cap on each request's worst inter-token gap
        for lim in slo.get('stall_ms_grid', ()):
            okm = [r for r in ok if not r['itl_ms'] or max(r['itl_ms']) <= lim]
            out[f'attainment_stall{int(lim)}'] = len(okm) / len(reqs)
            out[f'goodput_stall{int(lim)}_rps'] = len(okm) / span
    return out


def round2(x):
    if x <= 0:
        return x
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e) * 10 ** e


def slo_from_pilot(t):
    s = trace_summary(t)
    return {'ttft_s': round2(5 * s['ttft_p90_s']), 'itl_ms': round2(5 * s['req_itl_p95_median_ms']),
            'pilot': {'ttft_p90_s': s['ttft_p90_s'], 'req_itl_p95_median_ms': s['req_itl_p95_median_ms'], 'requests': s['requests'],
                      'complete': s['complete']}}


def serve_run(run, slo=None):
    meta = json.loads((run / 'run.json').read_text())
    out = []
    for tr in meta.get('traces', []):
        t = json.loads((run / tr['file']).read_text())
        sm = trace_summary(t, slo)
        out.append({'tag': tr['tag'], 'workload': t['workload'], 'arrival': t['arrival'], 'rate': t['rate'], 'seed': t['seed'],
                    'trace_sha256': t['trace_sha256'], 'content_sha256': t['content_sha256'], 'p_to_d_offered': t['demand']['p_to_d'],
                    **sm, 'warm': t.get('warm'), 'prefix_len': t.get('prefix_len'), 'prefix_mode': t.get('prefix_mode')})
    return meta, out


def capacity(rows, att=0.9):
    ok = [r['rate'] for r in rows if r.get('attainment', 0) >= att and r['incomplete'] == 0]
    return max(ok) if ok else None


# ---------------------------------------------------------------- experiment 4
def competitor_gbs(adir):
    f = adir / 'competitor.jsonl'
    if not f.exists():
        return None
    xs = [json.loads(x) for x in f.read_text().splitlines() if x.startswith('{"t0')]
    xs = [x for x in xs if not x.get('final')]
    return st.median(x['gb_per_s'] for x in xs) if xs else None


def e4_tables(root, plan, jr):
    rows = []
    for s in plan['steps']:
        if s['experiment'] != 'e4' or s['kind'] != 'engine':
            continue
        rec = final(jr.get(s['id'], []))
        if not rec or not rec.get('command_completed'):
            rows.append({'id': s['id'], 'status': 'missing' if not rec else rec.get('failure_class')})
            continue
        t = e1_tables(run_dir(rec))
        gbs = competitor_gbs(Path(rec['attempt_dir']))
        for c in t['cells']:
            rows.append({'id': s['id'], 'n': s['cus'], 'treatment': s['treatment'], 'block': s.get('block'), 'S': c['S'], 'N': c['N'],
                         'C': c['C'], 't_mix_ms': c['t_mix_ms'], 't_ref_ms': c['t_ref_ms'], 'competitor_gb_s': gbs})
    resp = []
    key = lambda r: (r['n'], r['block'], r['S'], r['N'], r['C'])  # noqa: E731
    base = {key(r): r for r in rows if r.get('treatment') == 'absent'}
    for r in rows:
        if 't_mix_ms' not in r or key(r) not in base:
            continue
        b = base[key(r)]
        resp.append(dict(r, mix_vs_absent=r['t_mix_ms'] / b['t_mix_ms'], ref_vs_absent=r['t_ref_ms'] / b['t_ref_ms']))
    agg = collections.defaultdict(list)
    for r in resp:
        agg[(r['n'], r['treatment'], r['S'], r['C'])].append(r)
    summ = []
    for (n, tr, S, Cn), v in sorted(agg.items()):
        mv = [x['mix_vs_absent'] for x in v]
        summ.append({'n': n, 'treatment': tr, 'S': S, 'C': Cn, 'blocks': len(v), 'ratio_median': st.median(mv),
                     'ci95': boot_ci(mv) if len(mv) > 1 else [None, None],
                     'competitor_gb_s': st.median([x['competitor_gb_s'] for x in v if x['competitor_gb_s']] or [0]) or None})
    return {'rows': rows, 'summary': summ}


# ---------------------------------------------------------------- profile attribution
def classify(name):
    n = name.lower()
    if any(k in n for k in ('attention', 'attn', 'paged', 'fmha', 'flash', 'mha', '_fwd_kernel')):
        return 'attention'
    if any(k in n for k in ('wvsplitk', 'gemm', 'cijk', 'matmul', 'mm_', 'skinny', 'hipblaslt')):
        return 'linear'
    return 'other'


def profile_tables(adir):
    kt = glob.glob(str(adir / 'rocprof' / '**' / '*kernel_trace.csv'), recursive=True)
    mt = glob.glob(str(adir / 'rocprof' / '**' / '*marker_api_trace.csv'), recursive=True)
    if not kt or not mt:
        return {'status': 'unavailable', 'kernel_files': len(kt), 'marker_files': len(mt)}
    ranges = []
    for f in mt:
        for r in csv.DictReader(open(f)):
            nm = r.get('Function') or r.get('Kind_Name') or ''
            if nm.startswith('cp|') and '_r-1_' not in nm:
                ranges.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), nm[3:]))
    ranges.sort()
    agg = collections.defaultdict(lambda: collections.Counter())
    names = collections.defaultdict(collections.Counter)
    for f in kt:
        for r in csv.DictReader(open(f)):
            s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
            for a, b, tag in ranges:
                if a <= s <= b:
                    cl = classify(r['Kernel_Name'])
                    agg[tag][cl] += (e - s) / 1e6
                    names[cl][r['Kernel_Name'][:80]] += 1
                    break
    by = collections.defaultdict(list)
    for tag, c in agg.items():
        parts = tag.split('_')  # S{S}_N{N}_C{C}_r{rep}_{role}
        key = '_'.join(p for p in parts if not p.startswith('r') or p in ('ref',))
        by[key].append(dict(c))
    out = {k: {cl: st.median(x.get(cl, 0) for x in v) for cl in ('linear', 'attention', 'other')} for k, v in by.items()}
    return {'status': 'ok', 'ms_per_step_by_class': out, 'top_kernels': {k: v.most_common(8) for k, v in names.items()}}


# ---------------------------------------------------------------- stage 1 report
def s1(a):
    root = Path(a.root)
    out = Path(a.out or root / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / 'launch_plan.json').read_text())
    jr = journal(root)
    res = {'plan': plan['plan'], 'cohort': plan['cohort'], 'steps': {}, 'e1': {}, 'e1p': {}, 'e3s': {}, 'pilot': {}, 'link': {}}
    for s in plan['steps']:
        rec = final(jr.get(s['id'], []))
        res['steps'][s['id']] = 'missing' if not rec else ('complete' if rec.get('command_completed') else rec.get('failure_class'))
        if not rec or not rec.get('command_completed'):
            continue
        run = run_dir(rec)
        if s['experiment'] == 'e1':
            res['e1'][s['id']] = dict(e1_tables(run), model=s.get('model'), alloc=s.get('alloc'), cus=s['cus'], grid=s['grid'])
        elif s['experiment'] == 'e1p':
            res['e1p'][s['id']] = dict(profile_tables(Path(rec['attempt_dir'])), model=s.get('model'), alloc=s.get('alloc'))
        elif s['experiment'] == 'e3s':
            res['e3s'][s['id']] = dict(e3_tables(run), model=s.get('model'), alloc=s.get('alloc'))
        elif s['experiment'] == 'pilot':
            meta, rows = serve_run(run)
            res['pilot'][s['id']] = {'model': s.get('model'), 'alloc': s.get('alloc'), 'rows': rows}
        elif s['experiment'] == 'e5link':
            res['link'][s['id']] = json.loads((run / 'run.json').read_text())
    # SLOs and capacity (frozen rules, design section 4)
    slo = {}
    for sid, p in res['pilot'].items():
        if sid.startswith('pilot_slo_'):
            m, wl = sid[len('pilot_slo_'):].split('_', 1)
            rec = final(jr[sid])
            meta = json.loads((run_dir(rec) / 'run.json').read_text())
            t = json.loads((run_dir(rec) / meta['traces'][0]['file']).read_text())
            slo[f'{m}_{wl}'] = slo_from_pilot(t)
    caps = {}
    for sid, p in res['pilot'].items():
        if sid.startswith('pilot_cap_'):
            m, wl, alloc = sid[len('pilot_cap_'):].split('_', 2)
            lim = slo.get(f'{m}_{wl}')
            if not lim:
                continue
            rec = final(jr[sid])
            _, rows = serve_run(run_dir(rec), lim)
            caps[f'{m}_{wl}_{alloc}'] = {'rows': rows, 'lambda_star': capacity(rows)}
    res['slo'], res['capacity'] = slo, caps
    res['e4'] = e4_tables(root, plan, jr)
    (out / 's1_tables.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(f"s1 tables -> {out / 's1_tables.json'}")
    return res


# ---------------------------------------------------------------- experiment 5
def disagg_trace(t, slo, kv_per_tok, stall=()):
    reqs = t['requests']
    nw = int(len(reqs) * 0.10)
    order = sorted(reqs, key=lambda r: r['arrival_s'])
    meas = order[nw:]
    done = [r for r in meas if r['tokens'] == r['max_tokens'] and r['token_s']]
    span = meas[-1]['arrival_s'] - meas[0]['arrival_s'] if len(meas) > 1 else None
    def itl(r):
        ts = r['token_s']
        return [1e3 * (b - a) for a, b in zip(ts, ts[1:])]
    ttft = [r['token_s'][0] - r['arrival_s'] for r in done]
    # lc amendment 7 (optional): prompts >= slo['long_prompt_min'] use slo['ttft_long_s'] (mixed-tenant workload)
    lim_ttft = lambda r: (slo['ttft_long_s'] if 'long_prompt_min' in slo and r['prompt_len'] >= slo['long_prompt_min']  # noqa: E731
                          else slo['ttft_s'])
    ok = [r for r in done if r['token_s'][0] - r['arrival_s'] <= lim_ttft(r) and (len(r['token_s']) < 2 or p95(itl(r)) <= slo['itl_ms'])]
    out = {'requests': len(meas), 'complete': len(done), 'attainment': len(ok) / len(meas) if meas else None,
           'goodput_rps': len(ok) / span if span else None, 'goodput_tok_s': sum(r['tokens'] for r in ok) / span if span else None,
           'ttft_p50_s': pct(ttft, .5), 'ttft_p99_s': pct(ttft, .99),
           'itl_p50_ms': pct([x for r in done for x in itl(r)], .5), 'itl_p99_ms': pct([x for r in done for x in itl(r)], .99)}
    if stall:  # lc amendment 5: the stall SLO (primary SLO and each request's maximum ITL <= cap), additive
        out['incomplete'] = len(meas) - len(done)
        mx = [max(itl(r)) for r in done if len(r['token_s']) > 1]
        out['req_itl_max_p50_ms'], out['req_itl_max_p99_ms'] = pct(mx, .5), pct(mx, .99)
        for lim in stall:
            okm = [r for r in ok if len(r['token_s']) < 2 or max(itl(r)) <= lim]
            out[f'attainment_stall{int(lim)}'] = len(okm) / len(meas) if meas else None
            out[f'goodput_stall{int(lim)}_rps'] = len(okm) / span if span else None
    if slo.get('itl_tight_ms'):  # lc amendment 7: tight-ITL SLO, same TTFT, p95 ITL <= itl_tight_ms
        okt = [r for r in done if r['token_s'][0] - r['arrival_s'] <= lim_ttft(r) and
               (len(r['token_s']) < 2 or p95(itl(r)) <= slo['itl_tight_ms'])]
        k = f"itl{int(slo['itl_tight_ms'])}"
        out[f'attainment_{k}'] = len(okt) / len(meas) if meas else None
        out[f'goodput_{k}_rps'] = len(okt) / span if span else None
    if 'long_prompt_min' in slo:  # lc amendment 7: short (chat) requests alone
        sh = [r for r in done if r['prompt_len'] < slo['long_prompt_min']]
        out['short_ttft_p50_s'] = pct([r['token_s'][0] - r['arrival_s'] for r in sh], .5)
        out['short_ttft_p99_s'] = pct([r['token_s'][0] - r['arrival_s'] for r in sh], .99)
        out['short_req_itl_max_p99_ms'] = pct([max(itl(r)) for r in sh if len(r['token_s']) > 1], .99)
        out['short_itl_p99_ms'] = pct([x for r in sh for x in itl(r)], .99)
    pre = [r for r in done if r.get('prefill_s') is not None]
    if pre:
        out['prefill_response_p50_s'] = pct([r['prefill_s'] for r in pre], .5)
        out['decode_first_token_after_prefill_p50_s'] = pct([r['token_s'][0] - r['prefill_done_s'] for r in pre], .5)
        out['kv_bytes_logical_mean'] = st.mean(r['prompt_len'] * kv_per_tok for r in pre)
    tok = sum(r['tokens'] for r in reqs)
    js = [x.get('counter_j') for x in t.get('telemetry', [])]
    if tok and js and all(isinstance(j, (int, float)) for j in js):
        out['joules_per_output_token_two_boards'] = sum(js) / tok
        out['joules_by_board'] = js
    m = t.get('metrics') or []
    for k in sorted({k for x in m for k in x if k != 't'}):
        v = [x[k] for x in m if k in x]
        if not v:
            continue
        if k.endswith('num_preemptions_total'):  # counter: preemptions during this trace
            out[k.replace('_total', '_trace')] = v[-1] - v[0]
        else:
            out[k + '_max'] = max(v)
    out['routes'] = sorted({r.get('route') for r in reqs if r.get('route')})
    return out


def disagg(a):
    root = Path(a.root)
    out = Path(a.out or root / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    slo = json.loads(Path(a.slo).read_text())
    plan = json.loads((root / 'launch_plan.json').read_text())
    jr = journal(root)
    rows = []
    for s in plan['steps']:
        if s['experiment'] not in ('e5', 'e5s', 'e6', 'e7', 'e8'):
            continue
        rec = final(jr.get(s['id'], []))
        if not rec or not rec.get('command_completed'):
            rows.append({'id': s['id'], 'status': 'missing' if not rec else rec.get('failure_class'),
                         'message': (rec or {}).get('failure_message')})
            continue
        run = run_dir(rec)
        meta = json.loads((run / 'run.json').read_text())
        kv = meta['layout']['kv_bytes_per_token']
        lim = slo[f"{s['model']}_{s['workload']}"]
        for tr in meta['traces']:
            t = json.loads((run / tr['file']).read_text())
            rows.append({'id': s['id'], 'experiment': s['experiment'], 'deployment': s['deployment'], 'workload': s['workload'],
                         'rate': t['rate'], 'seed': t.get('seed'), 'gpus': len(meta['gpus']), 'prefill_workers':
                         s.get('prefill_workers'), 'decode_kv_gib': s.get('decode_kv_gib'), 'decode_alloc': s.get('decode_alloc'), 'decode_cus': s.get('decode_cus'),
                         'trace_sha256': t['trace_sha256'], **disagg_trace(t, lim, kv)})
    (out / 'e5_tables.json').write_text(json.dumps(rows, indent=1, default=str) + '\n')
    print(f'e5 rows {len(rows)} -> {out}')


def serve_cmd(a):
    root = Path(a.root)
    out = Path(a.out or root / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    slo = json.loads(Path(a.slo).read_text())
    plan = json.loads((root / 'launch_plan.json').read_text())
    jr = journal(root)
    rows = []
    for s in plan['steps']:
        if s['experiment'] not in ('e2c', 'e2t', 'e2h', 'e3v'):
            continue
        rec = final(jr.get(s['id'], []))
        base = {k: s.get(k) for k in ('id', 'experiment', 'model', 'workload', 'alloc', 'pname', 'policy', 'budget', 'whole',
                                      'family', 'P', 'mode', 'cus', 'grid')}
        if not rec or not rec.get('command_completed'):
            rows.append(dict(base, status='missing' if not rec else rec.get('failure_class'), message=(rec or {}).get('failure_message')))
            continue
        wl = s.get('workload') or 'balanced'
        lim = slo.get(f"{s['model']}_{wl}") or slo.get(f"{s['model']}_balanced")
        _, tr = serve_run(run_dir(rec), lim)
        for r in tr:
            rows.append(dict(base, status='complete', **r))
    (out / 'serve_tables.json').write_text(json.dumps(rows, indent=1, default=str) + '\n')
    print(f'serve rows {len(rows)} -> {out}')


# ---------------------------------------------------------------- tuning selection (design sections 4 and 12)
FAMILY = {'whole': 'whole', 'sep2048': 'separate', 'mix2048': 'mixed', 'mix256': 'mixed', 'mix512': 'mixed', 'mix1024': 'mixed',
          'mix4096': 'mixed', 'sep512': 'separate', 'sep4096': 'separate'}


def sustainable(rows):
    """Highest rate r such that every trace at rates <= r has attainment >= 0.9 and no incomplete request."""
    best = 0.0
    for rate in sorted({r['rate'] for r in rows}):
        rs = [r for r in rows if r['rate'] == rate]
        if all(r.get('attainment', 0) >= 0.9 and r['incomplete'] == 0 for r in rs):
            best = rate
        else:
            break
    return best


def select(a):
    rows = json.loads(Path(a.serve).read_text())
    s1p = json.loads(Path(a.params).read_text())
    dev = [r for r in rows if r.get('status') == 'complete' and r['experiment'] in ('e2c', 'e2t') and r['model'] == 'q32'
           and r['arrival'] == 'poisson']
    out = {'lambda_star': s1p['lambda_star'], 'tuned': {}, 'best_colocated': {}, 'selection': {}}
    for wl in ('dheavy', 'balanced', 'pheavy'):
        cand = []
        for alloc in ('full', '128t'):
            for fam in ('whole', 'separate', 'mixed'):
                opts = []
                for pn in sorted({r['pname'] for r in dev if r['workload'] == wl and r['alloc'] == alloc and FAMILY[r['pname']] == fam}):
                    rs = [r for r in dev if r['workload'] == wl and r['alloc'] == alloc and r['pname'] == pn]
                    s101 = [r for r in rs if r['seed'] == 101]
                    sus = sustainable(s101)
                    rates = sorted({r['rate'] for r in rs})
                    hi = [r for r in rs if r['rate'] == rates[-1]]
                    gp = st.mean(r.get('goodput_rps') or 0 for r in hi) if hi else 0
                    opts.append({'pname': pn, 'budget': rs[0]['budget'], 'sustainable_rps': sus, 'goodput_at_top_rate': gp,
                                 'seeds': sorted({r['seed'] for r in rs})})
                if not opts:
                    continue
                opts.sort(key=lambda o: (-o['sustainable_rps'], -o['goodput_at_top_rate'], o['budget']))
                out['tuned'][f'q32_{wl}_{alloc}_{fam}'] = opts[0]['pname']
                out['selection'][f'q32_{wl}_{alloc}_{fam}'] = opts
                cand.append((opts[0], fam, alloc))
        b = sorted(cand, key=lambda x: (-x[0]['sustainable_rps'], -x[0]['goodput_at_top_rate'], x[2] != 'full'))[0]
        pol = {'whole': 'separate', 'separate': 'separate', 'mixed': 'mixed'}[b[1]]
        out['best_colocated'][wl] = {'policy': pol, 'budget': b[0]['budget'], 'whole': b[1] == 'whole', 'alloc': b[2],
                                     'pname': b[0]['pname'], 'cus': None if b[2] == 'full' else 128,
                                     'grid': None if b[2] == 'full' else 128}
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'tuned': out['tuned'], 'best_colocated': out['best_colocated']}, indent=1))


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = q.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('s1')
    r.add_argument('--root', required=True)
    r.add_argument('--out')
    for nm in ('serve', 'disagg'):
        r = s.add_parser(nm)
        r.add_argument('--root', required=True)
        r.add_argument('--slo', required=True)
        r.add_argument('--out')
    r = s.add_parser('select')
    r.add_argument('--serve', required=True)
    r.add_argument('--params', required=True)
    r.add_argument('--out', required=True)
    a = q.parse_args()
    {'s1': s1, 'serve': serve_cmd, 'disagg': disagg, 'select': select}[a.cmd](a)
