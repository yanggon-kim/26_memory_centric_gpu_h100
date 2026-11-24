#!/usr/bin/env python3
"""Long-context campaign analysis (rules: docs/amd/lc_campaign_design_20261124.md). CPU only; run on a devel node.

  lc_analyze.py lc1 --root ROOT [--out DIR]      L0 anchors, L1, L2, L3 part A, L4, SLO/capacity pilot
  lc_analyze.py smoke --root ROOT [--out DIR]    smoke status, kernel classes, short tables

Kernel classes (ROCM_ATTN, verified on the November 18 traces and in smoke): decode_attn = paged_attention*,
prefill_attn = Triton _fwd_kernel / context or flash attention, kv_update = reshape_and_cache*, linear = hipBLASLt Cijk* /
wvSplitK* / gemm, other = norms, activations, rotary, sampling, copies. Busy time = union of kernel intervals inside a
marker range; host gap = range span - busy. Counter values are summed over the dispatches inside a range; units:
FETCH_SIZE and WRITE_SIZE in KiB (validated on gfx942 by the G4 audit), TCC_EA0_RDREQ_DRAM_sum raw requests,
SQ_INSTS_VALU_MFMA_MOPS_BF16 x 512 = issued MFMA FLOPs (rocprof-compute convention).
"""
import argparse
import bisect
import collections
import csv
import glob
import json
from pathlib import Path
import re
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cp'))
import cp_analyze as A  # noqa: E402

KV_TOK = 262144                # bytes of KV per token, Qwen2.5-32B BF16
WEIGHTS = 63.97e9              # logical weight bytes per full forward (goal G7)
LIN_FLOP_TOK = 2 * 31.98e9     # ≈ 2 x parameters per token (upper bound: includes embedding rows)
CLASSES = ('linear', 'decode_attn', 'prefill_attn', 'kv_update', 'other')
MARK = re.compile(r'^(cp|lc)\|')


def kclass(name):
    n = name.lower()
    if 'paged_attention' in n:
        return 'decode_attn'
    if name.startswith('_fwd_kernel') or any(k in n for k in ('context_attention', 'fmha', 'flash', 'attn_fwd',
                                                                'prefix_prefill', 'unified_attention')):
        return 'prefill_attn'
    if 'reshape_and_cache' in n:
        return 'kv_update'
    if any(k in n for k in ('wvsplitk', 'gemm', 'cijk', 'matmul', 'skinny', 'hipblaslt')):
        return 'linear'
    return 'other'


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def union_ns(iv):
    tot, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def load_profile(adir):
    """{tag: {busy_ms, span_ms, classes ms, kernels, counters{name: sum}}} over marker ranges of one rocprofv3 run."""
    base = Path(adir) / 'rocprof'
    kt = glob.glob(str(base / '**' / '*kernel_trace.csv'), recursive=True)
    mt = glob.glob(str(base / '**' / '*marker_api_trace.csv'), recursive=True)
    cc = glob.glob(str(base / '**' / '*counter_collection.csv'), recursive=True)
    if not kt or not mt:
        return None
    ranges = []
    for f in mt:
        for r in csv.DictReader(open(f)):
            nm = r.get('Function') or r.get('Kind_Name') or ''
            if MARK.match(nm):
                ranges.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), nm[3:]))
    ranges.sort()
    starts = [x[0] for x in ranges]
    ctr = collections.defaultdict(dict)
    for f in cc:
        for r in csv.DictReader(open(f)):
            d = ctr[int(r['Dispatch_Id'])]
            d[r['Counter_Name']] = d.get(r['Counter_Name'], 0.0) + float(r['Counter_Value'])
    out = {}
    iv = collections.defaultdict(list)
    names = collections.defaultdict(collections.Counter)
    for f in kt:
        for r in csv.DictReader(open(f)):
            s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
            i = bisect.bisect_right(starts, s) - 1
            if i < 0 or s > ranges[i][1]:
                continue
            tag = ranges[i][2]
            o = out.setdefault(tag, {c: 0.0 for c in CLASSES} | {'kernels': 0, 'counters': collections.Counter(),
                                                                  'span_ms': (ranges[i][1] - ranges[i][0]) / 1e6})
            cl = kclass(r['Kernel_Name'])
            o[cl] += (e - s) / 1e6
            o['kernels'] += 1
            iv[tag].append((s, e))
            names[cl][r['Kernel_Name'][:90]] += 1
            for k, v in ctr.get(int(r['Dispatch_Id']), {}).items():
                o['counters'][k] += v
    for tag, o in out.items():
        o['busy_ms'] = union_ns(iv[tag]) / 1e6
        o['host_gap_ms'] = o['span_ms'] - o['busy_ms']
        o['counters'] = dict(o['counters'])
    return {'ranges': out, 'top_kernels': {k: v.most_common(12) for k, v in names.items()}}


def steps_of(plan, prefix):
    return [s for s in plan['steps'] if s['experiment'] == prefix]


def done_dir(jr, sid):
    rec = A.final(jr.get(sid, []))
    if rec and rec.get('command_completed'):
        return Path(rec['attempt_dir'])
    return None


def status_rows(plan, jr):
    out = []
    for s in plan['steps']:
        rec = A.final(jr.get(s['id'], []))
        out.append({'id': s['id'], 'experiment': s['experiment'], 'attempts': len(jr.get(s['id'], [])),
                    'completed': bool(rec and rec.get('command_completed')), 'failure_class': (rec or {}).get('failure_class'),
                    'failure_message': ((rec or {}).get('failure_message') or '')[:300]})
    return out


# ------------------------------------------------------------------ L0
def anchors(plan, jr, s1_tables):
    res = {}
    d = done_dir(jr, next((s['id'] for s in steps_of(plan, 'lc0') if s['script'].endswith('cp_replay.py')), ''))
    if d:
        t = A.e1_tables(d / 'run')
        prev = {c['C']: c for c in s1_tables['e1']['e1_q32_full']['cells'] if c['S'] == 8192 and c['N'] == 8}
        rows = []
        for c in t['cells']:
            p = prev.get(c['C'])
            if not p:
                continue
            m_now, m_prev = c['t_mix_ms'] - c['t_ref_ms'], p['t_mix_ms'] - p['t_ref_ms']
            ok_ref = abs(c['t_ref_ms'] - p['t_ref_ms']) <= max(0.05 * p['t_ref_ms'], 1.0)
            ok_m = abs(m_now - m_prev) <= max(0.05 * abs(m_prev), 1.0)
            rows.append({'C': c['C'], 't_ref_ms': c['t_ref_ms'], 't_ref_ms_s1': p['t_ref_ms'], 'marginal_ms': m_now,
                         'marginal_ms_s1': m_prev, 'pass': ok_ref and ok_m})
        res['step_anchor'] = rows
    sv = next((s for s in steps_of(plan, 'lc0') if s['script'].endswith('cp_serve.py')), None)
    d = done_dir(jr, sv['id']) if sv else None
    if d:
        meta = json.loads((d / 'run' / 'run.json').read_text())
        t = json.loads((d / 'run' / meta['traces'][0]['file']).read_text())
        s = A.trace_summary(t)
        dc, dt = s['step_fraction_count'].get('decode', 0), s['step_fraction_time'].get('decode', 0)
        res['serving_anchor'] = {'decode_share_steps': dc, 'decode_share_exec': dt, 'complete': s['complete'],
                                 'pass': 0.950 <= dc <= 0.965 and 0.864 <= dt <= 0.910}
    return res


# ------------------------------------------------------------------ L1
def l1(plan, jr):
    arms = {}
    for s in steps_of(plan, 'lc1'):
        d = done_dir(jr, s['id'])
        if d:
            arms[s.get('arm', s['id'])] = [r for r in A.jl(d / 'run' / 'prompts.jsonl') if r['rep'] >= 0]
    out = {'timing': [], 'ratios': [], 'agreement': []}
    whole = {(r['L'], r['rep']): r for r in arms.get('whole', [])}
    for an, rows in sorted(arms.items()):
        by = collections.defaultdict(list)
        for r in rows:
            by[r['L']].append(r)
        for L, v in sorted(by.items()):
            starts = v[0]['chunk_start']
            out['timing'].append({'arm': an, 'L': L, 'reps': len(v), 'chunks': v[0]['n_chunks'], 'chunk_tokens': v[0]['chunk_tokens'][:3],
                                  'ttft_ms': 1e3 * med([r['ttft_s'] for r in v]), 'exec_ms': 1e3 * med([r['exec_s_sum'] for r in v]),
                                  'gpu_event_ms': med([r['gpu_event_ms'] for r in v]),
                                  'revisit_tokens': sum(starts), 'revisit_gb': sum(starts) * KV_TOK / 1e9,
                                  'weight_reads_gb': v[0]['n_chunks'] * WEIGHTS / 1e9})
            if an != 'whole':
                pr = [(r, whole[(L, r['rep'])]) for r in v if (L, r['rep']) in whole]
                same = all(a['prompt_sha256'] == b['prompt_sha256'] for a, b in pr)
                for key in ('ttft_s', 'exec_s_sum', 'gpu_event_ms'):
                    xs = [a[key] / b[key] for a, b in pr if a[key] and b[key]]
                    lo, hi = A.boot_ci(xs)
                    out['ratios'].append({'arm': an, 'L': L, 'metric': key, 'pairs': len(xs), 'ratio_median': med(xs),
                                          'ci95': [lo, hi], 'paired_same_prompt': same})
                out['agreement'].append({'arm': an, 'L': L, 'first_token_match': sum(a['first_token'] == b['first_token'] for a, b in pr),
                                         'pairs': len(pr)})
    # kernel trace attribution per (arm, L): sum over chunk ranges per rep, median over reps
    out['ktrace'] = []
    for s in steps_of(plan, 'lc1p'):
        d = done_dir(jr, s['id'])
        prof = load_profile(d) if d else None
        if not prof:
            continue
        per = collections.defaultdict(lambda: collections.Counter())
        for tag, o in prof['ranges'].items():
            m = re.match(r'L(\d+)_r(-?\d+)_k(\d+)', tag)
            if not m or int(m.group(2)) < 0:
                continue
            c = per[(int(m.group(1)), int(m.group(2)))]
            for k in CLASSES + ('busy_ms', 'span_ms', 'host_gap_ms'):
                c[k] += o[k]
            c['steps'] += 1
        byL = collections.defaultdict(list)
        for (L, rep), c in per.items():
            byL[L].append(c)
        for L, v in sorted(byL.items()):
            out['ktrace'].append({'arm': s.get('arm', s['id']), 'L': L, **{k: med([c[k] for c in v]) for k in CLASSES + ('busy_ms', 'span_ms',
                                                                                                         'host_gap_ms', 'steps')}})
        out.setdefault('top_kernels', {})[s.get('arm', s['id'])] = prof['top_kernels']
    # eager calibration and counters
    out['eager'] = []
    for s in steps_of(plan, 'lc1e'):
        d = done_dir(jr, s['id'])
        if d:
            rows = [r for r in A.jl(d / 'run' / 'prompts.jsonl') if r['rep'] >= 0]
            for L in sorted({r['L'] for r in rows}):
                out['eager'].append({'arm': s.get('arm', s['id']), 'L': L, 'exec_ms': 1e3 * med([r['exec_s_sum'] for r in rows if r['L'] == L])})
    out['counters'] = []
    for s in steps_of(plan, 'lc1c'):
        d = done_dir(jr, s['id'])
        prof = load_profile(d) if d else None
        if not prof:
            out['counters'].append({'arm': s.get('arm'), 'counter': s.get('counter'), 'status': 'missing'})
            continue
        per = collections.Counter()
        busy = collections.Counter()
        for tag, o in prof['ranges'].items():
            m = re.match(r'L(\d+)_r(-?\d+)_k(\d+)', tag)
            if m and int(m.group(2)) == 0:
                per[int(m.group(1))] += sum(o['counters'].values())
                busy[int(m.group(1))] += o['busy_ms']
        for L in sorted(per):
            out['counters'].append({'arm': s.get('arm'), 'counter': s.get('counter'), 'L': L, 'value': per[L], 'busy_ms_profiled': busy[L]})
    return out


# ------------------------------------------------------------------ L2 / L4 replay tables
def replay_cells(run):
    rows = [r for r in A.jl(run / 'steps.jsonl') if r.get('role') and r.get('rep', -1) >= 0]
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[(r['S'], r['N'], r.get('P', 0), r['C'], r['rep'])][r['role']] = r
    cells = collections.defaultdict(list)
    for (S, N, P, Cn, rep), d in pairs.items():
        cells[(S, N, P, Cn)].append(d)
    out = {}
    for k, v in cells.items():
        out[k] = {'pairs': len(v), 't_mix_ms': 1e3 * med([d['mix']['exec_s'] for d in v if 'mix' in d]) if any('mix' in d for d in v) else None,
                  't_ref_ms': 1e3 * med([d['ref']['exec_s'] for d in v if 'ref' in d]) if any('ref' in d for d in v) else None,
                  'ratio': med([d['mix']['exec_s'] / d['ref']['exec_s'] for d in v if 'mix' in d and 'ref' in d]),
                  'ratios': [d['mix']['exec_s'] / d['ref']['exec_s'] for d in v if 'mix' in d and 'ref' in d],
                  'decode_ctx_max': med([d['mix'].get('decode_ctx_max') for d in v if 'mix' in d]),
                  'tokens': next((d['mix'].get('total_tokens') for d in v if 'mix' in d), None)}
    return out


def profile_cells(prof):
    by = collections.defaultdict(list)
    for tag, o in prof['ranges'].items():
        m = re.match(r'S(\d+)_N(\d+)_P(\d+)_C(\d+)_r(-?\d+)_(ref|mix)', tag)
        if m and int(m.group(5)) >= 0:
            by[(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(6))].append(o)
    out = {}
    for k, v in by.items():
        out[k] = {c: med([o[c] for o in v]) for c in CLASSES + ('busy_ms', 'span_ms', 'host_gap_ms')}
        out[k]['counters'] = {n: med([o['counters'].get(n) for o in v]) for n in {n for o in v for n in o['counters']}}
    return out


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / ss if ss else None
    return {'intercept': a, 'slope': b, 'r2': r2, 'n': n}


def l2(plan, jr):
    cells = {}
    for s in steps_of(plan, 'lc2'):
        d = done_dir(jr, s['id'])
        if d:
            cells.update(replay_cells(d / 'run'))
    anc = {(P, Cn): v['t_mix_ms'] for (S, N, P, Cn), v in cells.items() if N == 0 and v['t_mix_ms']}
    rows = []
    for (S, N, P, Cn), v in sorted(cells.items()):
        if N == 0:
            continue
        sep = v['t_ref_ms'] + anc[(P, Cn)] if Cn and (P, Cn) in anc and v['t_ref_ms'] else None
        rows.append({'S': S, 'N': N, 'P': P, 'C': Cn, 'pairs': v['pairs'], 't_ref_ms': v['t_ref_ms'], 't_mix_ms': v['t_mix_ms'],
                     'marginal_ms': (v['t_mix_ms'] - v['t_ref_ms']) if v['t_mix_ms'] and v['t_ref_ms'] else None,
                     'ratio': v['ratio'], 'ci95': list(A.boot_ci(v['ratios'])), 't_chunk_alone_ms': anc.get((P, Cn)),
                     'separate_ms': sep, 'hybrid_saving_frac': (sep - v['t_mix_ms']) / sep if sep else None,
                     'tokens': v['tokens']})
    prof = {}
    for s in steps_of(plan, 'lc2p'):
        d = done_dir(jr, s['id'])
        p = load_profile(d) if d else None
        if p:
            prof.update(profile_cells(p))
    prow = [dict(S=k[0], N=k[1], P=k[2], C=k[3], role=k[4], **{c: v[c] for c in CLASSES + ('busy_ms', 'span_ms', 'host_gap_ms')},
                 coverage=1 - (v['other'] / v['busy_ms']) if v['busy_ms'] else None) for k, v in sorted(prof.items())]
    ref = [r for r in prow if r['role'] == 'ref' and r['N'] > 0]
    h2a = fit([r['N'] * r['S'] for r in ref], [r['decode_attn'] for r in ref]) if ref else None
    return {'cells': rows, 'anchors_ms': {f'P{P}_C{Cn}': t for (P, Cn), t in sorted(anc.items())}, 'profile': prow,
            'decode_attn_fit_vs_NS': h2a}


# ------------------------------------------------------------------ L3 A
def l3a(plan, jr):
    out = []
    for s in steps_of(plan, 'lc3a'):
        d = done_dir(jr, s['id'])
        if not d:
            continue
        meta = json.loads((d / 'run' / 'run.json').read_text())
        per = collections.defaultdict(list)
        for c in meta['cases']:
            if c['rep'] < 0:
                continue
            x = json.loads((d / 'run' / c['file']).read_text())
            st_ = x['steps']
            tot = sum(z['exec_s'] or 0 for z in st_) or 1
            dec = [z for z in st_ if z['kind'] == 'decode']
            last_pf = max((i for i, z in enumerate(st_) if z['prefill']), default=-1)
            tail = st_[last_pf + 1:]
            ia = None
            if x['I'] and x['inject_at_s'] is not None and len(x['a_times']) > 2:
                gaps = [(b - a_) * 1e3 for a_, b in zip(x['a_times'], x['a_times'][1:]) if b >= x['inject_at_s']]
                ia = max(gaps[:64]) if gaps else None
            per[(x['O'], x['I'])].append({'steps': len(st_), 'decode_frac_steps': len(dec) / len(st_),
                                          'decode_frac_exec': sum(z['exec_s'] or 0 for z in dec) / tot,
                                          'tail_start_step': last_pf + 1, 'tail_frac_exec': sum(z['exec_s'] or 0 for z in tail) / tot,
                                          'a_itl_p50_ms': med([(b - a_) * 1e3 for a_, b in zip(x['a_times'], x['a_times'][1:])]),
                                          'a_max_itl_after_inject_ms': ia,
                                          'b_ttft_ms': (x['b_times'][0] - x['inject_at_s']) * 1e3 if x['I'] and x['b_times'] else None,
                                          'span_s': x['span_s']})
        for (O, I), v in sorted(per.items()):
            out.append({'policy': s.get('pname', s['id']), 'O': O, 'I': I, 'reps': len(v), **{k: med([z[k] for z in v]) for k in v[0]}})
    prof = []
    for s in steps_of(plan, 'lc3p'):
        d = done_dir(jr, s['id'])
        p = load_profile(d) if d else None
        if not p:
            continue
        meta = json.loads((d / 'run' / 'run.json').read_text())
        for c in meta['cases']:
            if c['rep'] < 0:
                continue
            x = json.loads((d / 'run' / c['file']).read_text())
            busy = collections.Counter()
            aligned = 0
            for i, z in enumerate(x['steps']):
                o = p['ranges'].get(f"O{x['O']}_I{x['I']}_r{x['rep']}_s{i}")
                if o:
                    busy[z['kind']] += o['busy_ms']
                    aligned += 1
            tot = sum(busy.values()) or 1
            prof.append({'policy': s.get('pname', s['id']), 'O': x['O'], 'I': x['I'], 'aligned_steps': aligned, 'steps': len(x['steps']),
                         'gpu_busy_ms': dict(busy), 'decode_frac_gpu_busy': busy['decode'] / tot})
    return {'cases': out, 'profiled': prof}


# ------------------------------------------------------------------ L4
def l4(plan, jr):
    out = {}
    cal = {}
    for s in steps_of(plan, 'lc4e'):
        d = done_dir(jr, s['id'])
        if d:
            cal = replay_cells(d / 'run')
    out['eager_ms'] = [{'S': k[0], 'N': k[1], 'P': k[2], 'C': k[3], 't_ref_ms': v['t_ref_ms'], 't_mix_ms': v['t_mix_ms']}
                       for k, v in sorted(cal.items())]
    stream = {}
    for s in steps_of(plan, 'lc4s'):
        d = done_dir(jr, s['id'])
        if d and (d / 'out.json').exists():
            lines = [json.loads(x) for x in (d / 'out.json').read_text().splitlines() if x.startswith('{')]
            g = [x.get('gb_per_s') for x in lines if x.get('gb_per_s') and not x.get('final')]
            stream[s['id']] = med(g)
    out['stream_gb_s'] = stream
    ref_gbs = med([v for k, v in stream.items() if '_full_' in k])
    ctr = collections.defaultdict(dict)
    for s in steps_of(plan, 'lc4c'):
        d = done_dir(jr, s['id'])
        p = load_profile(d) if d else None
        if not p:
            continue
        for k, v in profile_cells(p).items():
            ctr[k][s.get('counter', (s.get('pmc') or ['?'])[0])] = sum(x for x in v['counters'].values() if x is not None)
            ctr[k].setdefault('busy_ms_profiled', v['busy_ms'])
    rows = []
    for (S, N, P, Cn, role), v in sorted(ctr.items()):
        cell = cal.get((S, N, P, Cn), {})
        t = cell.get('t_mix_ms' if role == 'mix' else 't_ref_ms')
        tok = N + (Cn if role == 'mix' else 0)
        logical_rd = WEIGHTS + N * S * KV_TOK + ((P + Cn) * KV_TOK if role == 'mix' and Cn else 0)
        fetch = v.get('FETCH_SIZE')
        r = {'S': S, 'N': N, 'P': P, 'C': Cn, 'role': role, 'eager_ms': t, 'fetch_gb': fetch * 1024 / 1e9 if fetch else None,
             'write_gb': v['WRITE_SIZE'] * 1024 / 1e9 if v.get('WRITE_SIZE') else None, 'rdreq_dram': v.get('TCC_EA0_RDREQ_DRAM_sum'),
             'mfma_tflop': v['SQ_INSTS_VALU_MFMA_MOPS_BF16'] * 512 / 1e12 if v.get('SQ_INSTS_VALU_MFMA_MOPS_BF16') else None,
             'linear_tflop_analytic': tok * LIN_FLOP_TOK / 1e12, 'logical_read_gb': logical_rd / 1e9}
        if fetch and t:
            r['fetch_over_logical'] = fetch * 1024 / logical_rd
            r['achieved_gb_s'] = fetch * 1024 / 1e9 / (t / 1e3)
            r['achieved_over_stream'] = r['achieved_gb_s'] / ref_gbs if ref_gbs else None
        rows.append(r)
    out['counters'] = rows
    # competition
    comp = []
    for s in steps_of(plan, 'lc4m'):
        if s['kind'] != 'engine':
            continue
        d = done_dir(jr, s['id'])
        if not d:
            comp.append({'id': s['id'], 'status': 'missing'})
            continue
        cells = replay_cells(d / 'run')
        gbs = A.competitor_gbs(d)
        for k, v in cells.items():
            comp.append({'block': s['block'], 'treatment': s['treatment'], 'S': k[0], 'N': k[1], 'P': k[2], 'C': k[3],
                         't_ref_ms': v['t_ref_ms'], 't_mix_ms': v['t_mix_ms'], 'competitor_gb_s': gbs})
    base = {(r['block'], r['S'], r['N'], r['C']): r for r in comp if r.get('treatment') == 'absent'}
    summ = collections.defaultdict(list)
    for r in comp:
        b = base.get((r.get('block'), r.get('S'), r.get('N'), r.get('C')))
        if not b or 't_ref_ms' not in r:
            continue
        if r['t_ref_ms'] and b['t_ref_ms']:
            summ[(r['treatment'], r['S'], r['N'], r['C'], 'ref')].append((r['t_ref_ms'] / b['t_ref_ms'], r['competitor_gb_s']))
        if r['C'] and r['t_mix_ms'] and b['t_mix_ms']:
            summ[(r['treatment'], r['S'], r['N'], r['C'], 'mix')].append((r['t_mix_ms'] / b['t_mix_ms'], r['competitor_gb_s']))
    out['competition'] = [{'treatment': k[0], 'S': k[1], 'N': k[2], 'C': k[3], 'role': k[4], 'blocks': len(v),
                           'ratio_median': med([x[0] for x in v]), 'ratios': [x[0] for x in v],
                           'competitor_gb_s': med([x[1] for x in v if x[1]])} for k, v in sorted(summ.items())]
    place = {}
    for s in steps_of(plan, 'lc4m'):
        if s['kind'] == 'native':
            rec = A.final(jr.get(s['id'], []))
            place[s['id']] = bool(rec and rec.get('check_passed'))
    out['placement_passed'] = place
    # known-byte audit
    d = next((done_dir(jr, s['id']) for s in steps_of(plan, 'lc4a')), None)
    if d and (d / 'run' / 'counter_audit.json').exists():
        out['audit'] = audit_judgment(json.loads((d / 'run' / 'counter_audit.json').read_text()))
    return out


def audit_judgment(aud):
    """Per counter (KiB unit for *_SIZE, 64-byte requests otherwise): ratio for streaming reads (1 GiB, 4 GiB) and for the
    MALL-resident read (128 MiB x 16). Pre-MALL request side if both ~1; post-MALL candidate if streaming ~1 and
    MALL-resident <= 0.25."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in aud.get('runs', []):
        if r.get('returncode') == 0 and r.get('counter_total') is not None:
            by[r['counter']][r['load']].append(r['ratio'])
    out = {}
    for c, loads in by.items():
        unit = 'KiB' if c.endswith('_SIZE') else 'req64'
        f = lambda ld: med([x.get(unit) for x in loads.get(ld, [])])  # noqa: E731
        s1, s2, m, l2 = f('read1g_x4'), f('read4g_x1'), f('read128m_x16'), f('read8m_x256')
        w = med([x.get(unit) for x in loads.get('write4g', [])])
        cls = 'unclassified'
        if s1 and s2 and 0.8 <= s1 <= 1.25 and 0.8 <= s2 <= 1.25:
            cls = ('pre-MALL request-side read' if m and m >= 0.75 else 'post-MALL read candidate' if m is not None and m <= 0.25
                   else 'read, ambiguous MALL behaviour')
        elif w and 0.8 <= w <= 1.25:
            cls = 'write, streaming-consistent'
        out[c] = {'unit': unit, 'stream_1g': s1, 'stream_4g': s2, 'mall_128m': m, 'l2_8m': l2, 'write_4g': w, 'class': cls}
    return out


# ------------------------------------------------------------------ pilot
def pilot(plan, jr):
    slo, cap = {}, {}
    for s in steps_of(plan, 'pilot'):
        d = done_dir(jr, s['id'])
        if not d or not s['id'].startswith('pilot_slo_'):
            continue
        meta = json.loads((d / 'run' / 'run.json').read_text())
        t = json.loads((d / 'run' / meta['traces'][0]['file']).read_text())
        slo[s['workload']] = A.slo_from_pilot(t)
    rows = {}
    for s in steps_of(plan, 'pilot'):
        d = done_dir(jr, s['id'])
        if not d or not s['id'].startswith('pilot_cap_') or s['workload'] not in slo:
            continue
        _, rs = A.serve_run(d / 'run', slo[s['workload']])
        rows[s['workload']] = [{k: r.get(k) for k in ('rate', 'attainment', 'incomplete', 'complete', 'ttft_p90_s', 'itl_p50_ms',
                                                      'req_itl_p95_median_ms', 'goodput_rps', 'step_fraction_time',
                                                      'effective_p_to_d', 'preemptions', 'max_waiting')} for r in rs]
        ok = [r['rate'] for r in rs if (r.get('attainment') or 0) >= 0.9 and r['incomplete'] == 0]
        lad = sorted(r['rate'] for r in rs)
        sus = 0
        for x in lad:
            if x in ok:
                sus = x
            else:
                break
        cap[s['workload']] = sus or None
    return {'slo': slo, 'lambda_star': cap, 'ladder': rows}


def lc1(a):
    root = Path(a.root)
    out = Path(a.out or root / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / 'launch_plan.json').read_text())
    jr = A.journal(root)
    s1 = json.loads(Path(a.s1_tables).read_text())
    res = {'plan': plan['plan'], 'status': status_rows(plan, jr)}
    for name, fn in (('anchors', lambda: anchors(plan, jr, s1)), ('l1', lambda: l1(plan, jr)), ('l2', lambda: l2(plan, jr)),
                     ('l3a', lambda: l3a(plan, jr)), ('l4', lambda: l4(plan, jr)), ('pilot', lambda: pilot(plan, jr))):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001  (a family's failure is recorded, the others still run)
            import traceback
            res[name] = {'error': repr(e), 'traceback': traceback.format_exc()}
        print(f'{name} done', flush=True)
    (out / 'lc_tables.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(f'-> {out / "lc_tables.json"}')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = q.add_subparsers(dest='cmd', required=True)
    for n in ('lc1', 'smoke'):
        r = s.add_parser(n)
        r.add_argument('--root', required=True)
        r.add_argument('--out')
        r.add_argument('--s1-tables', default=str(Path(__file__).resolve().parents[3] /
                                                  'results_published/amd_20261118_mi300x_cp_s1/s1_tables.json'))
    a = q.parse_args()
    lc1(a)
