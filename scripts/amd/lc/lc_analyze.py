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
        if r.get('returncode') == 0 and r.get('counter_total') is not None and r.get('ratio'):
            by[r['counter']][r['load']].append(r['ratio'])
    out = {}
    for c, loads in by.items():
        # unit: KiB for *_SIZE; otherwise the request size whose streaming-read ratio is nearest 1 (stated, not assumed)
        unit = 'KiB'
        if not c.endswith('_SIZE'):
            def dist(u):
                v = med([x.get(u) for x in loads.get('read1g_x4', [])])
                return abs(v - 1) if v is not None else 9e9
            unit = min(('req32', 'req64', 'req128'), key=dist)
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
        if not d or 'pilot_slo_' not in s['id']:
            continue
        meta = json.loads((d / 'run' / 'run.json').read_text())
        t = json.loads((d / 'run' / meta['traces'][0]['file']).read_text())
        slo[s['workload']] = A.slo_from_pilot(t)
    rows = {}
    for s in steps_of(plan, 'pilot'):
        d = done_dir(jr, s['id'])
        if not d or 'pilot_cap_' not in s['id'] or s['workload'] not in slo:
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


# ------------------------------------------------------------------ serving (lc2 / lc3: L3 part B and L5)
SERVE_EXP = ('lc5d', 'lc3b', 'lc3s', 'lc5h', 'lc5b', 'lc5r')
SERVE_KEYS = ('rate', 'arrival', 'seed', 'attainment', 'incomplete', 'complete', 'goodput_rps', 'goodput_tok_s', 'ttft_p50_s',
              'ttft_p90_s', 'itl_p50_ms', 'itl_p99_ms', 'req_itl_p95_median_ms', 'req_itl_max_p99_ms', 'output_tokens_per_s',
              'step_fraction_time', 'step_fraction_count', 'effective_p_to_d', 'preemptions', 'max_waiting',
              'joules_per_output_token')


def sustained(rows, att=0.9):
    """highest ladder rate below which every rate (inclusive) attains >= att with nothing incomplete (pilot rule)"""
    sus = None
    for r in sorted(rows, key=lambda r: r['rate']):
        if (r.get('attainment') or 0) >= att and r['incomplete'] == 0:
            sus = r['rate']
        else:
            break
    return sus


def serving(plan, jr, slo, stall=()):
    rows = []
    for s in plan['steps']:
        if s['experiment'] not in SERVE_EXP:
            continue
        d = done_dir(jr, s['id'])
        if not d:
            continue
        wl = s['workload']
        sl = slo.get((s.get('model', 'q32'), wl)) or slo.get(wl)  # 7B rows use the 7B pilot SLO (amendment 3)
        if sl is None:
            continue
        _, rs = A.serve_run(d / 'run', dict(sl, stall_ms_grid=stall))
        for r in rs:
            rows.append(dict({k: r.get(k) for k in SERVE_KEYS + tuple(k for k in r if '_stall' in k)}, experiment=s['experiment'], step=s['id'], workload=wl,
                             policy=s.get('pname') or f"{s.get('policy')}{s.get('budget')}", budget=s.get('budget'), model=s.get('model', 'q32')))
    caps = collections.defaultdict(dict)
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r['experiment'], r['model'], r['workload'], r['policy'], r['seed'], r['arrival'])].append(r)
    for k, v in grp.items():
        if len(v) > 1:
            caps['|'.join(map(str, k))] = {'sustained_rps': sustained(v), 'max_goodput_rps': max(x['goodput_rps'] or 0 for x in v),
                                           'rates': len(v)}
    return {'rows': rows, 'capacity': caps}


def select(serv, exp='lc5d', att='attainment', gp='goodput_rps'):
    """amendment 2: per workload, the best mixed budget and the better separate policy, by goodput (summed over the dev seeds)
    at the highest grid rate where any policy reaches attainment >= 0.9; ties to the smaller budget."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in serv['rows']:
        if r['experiment'] == exp:
            by[r['workload']][r['policy']].append(r)
    out = {}
    for wl, pols in by.items():
        ok_rates = sorted({round(r['rate'], 6) for v in pols.values() for r in v if (r.get(att) or 0) >= 0.9})
        if not ok_rates:
            out[wl] = {'note': 'no policy reaches attainment 0.9 at any rate'}
            continue
        x = ok_rates[-1]
        score = {p: sum(r.get(gp) or 0 for r in v if round(r['rate'], 6) == x) for p, v in pols.items()}
        # ties to the smaller budget, as written in amendment 2; whole = one 32768-token prefill
        bud = lambda p: 32768 if p == 'whole' else int(p.lstrip('mixsep'))  # noqa: E731
        mixed = sorted((p for p in score if p.startswith('mix')), key=lambda p: (-round(score[p], 6), bud(p)))
        sep = sorted((p for p in score if not p.startswith('mix')), key=lambda p: (-round(score[p], 6), bud(p)))
        out[wl] = {'rate': x, 'score': score, 'mixed': mixed[0] if mixed else None, 'separate': sep[0] if sep else None}
    return out


def lc_serve_plan(a):
    root = Path(a.root)
    out = Path(a.out or root / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / 'launch_plan.json').read_text())
    jr = A.journal(root)
    prm = json.loads(Path(a.params).read_text())
    res = {'plan': plan['plan'], 'status': status_rows(plan, jr), 'params': prm}
    slo = dict(prm['slo'])
    q7 = {}
    for s in steps_of(plan, 'pilot'):  # 7B SLO pilot inside lc3, frozen rule (5 x p90 TTFT, 5 x median per-request p95 ITL)
        d = done_dir(jr, s['id'])
        if d and s['id'].startswith('q7_pilot_slo_'):
            meta = json.loads((d / 'run' / 'run.json').read_text())
            q7[s['workload']] = A.slo_from_pilot(json.loads((d / 'run' / meta['traces'][0]['file']).read_text()))
            slo[('q7', s['workload'])] = q7[s['workload']]
    if q7:
        res['q7_slo'] = q7
    for name, fn in (('serving', lambda: serving(plan, jr, slo, prm.get('stall_ms_grid', ()))), ('l4', lambda: l4(plan, jr))):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            res[name] = {'error': repr(e), 'traceback': traceback.format_exc()}
        print(f'{name} done', flush=True)
    if 'rows' in res.get('serving', {}):
        res['selection'] = select(res['serving'])
        # secondary, stall-aware view (amendment 3): reported beside the primary selection, never replaces it
        res['selection_stall'] = {int(m): select(res['serving'], att=f'attainment_stall{int(m)}', gp=f'goodput_stall{int(m)}_rps')
                                  for m in prm.get('stall_ms_grid', ())}
    (out / 'lc_tables.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(f'-> {out / "lc_tables.json"}')


def decide(rows, sel, exp='lc5h', model='q32'):
    """Design section 7 + amendment 3: per workload and SLO, the selected mixed policy vs each of the selected separate
    policy and whole. All-seed sustained rate = highest ladder rate at which every seed attains >= 0.9 with nothing
    incomplete, at that rate and every lower one. win / loss if >= 1 ladder step apart, else uncertain."""
    slos = {'primary': ('attainment', 'goodput_rps'), 'stall1000': ('attainment_stall1000', 'goodput_stall1000_rps')}
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r['experiment'] == exp and r.get('model', 'q32') == model:
            by[r['workload']][r['policy']].append(r)
    out = {}
    for wl, pols in by.items():
        ladder = sorted({round(r['rate'], 6) for v in pols.values() for r in v})
        seeds = sorted({r['seed'] for v in pols.values() for r in v})
        res = {'ladder': ladder, 'seeds': seeds}
        for name, (att, gp) in slos.items():
            cap, good = {}, {}
            for p, v in pols.items():
                cell = {(round(r['rate'], 6), r['seed']): r for r in v}
                sus, complete = None, True
                for x in ladder:
                    rs = [cell.get((x, sd)) for sd in seeds]
                    if any(r is None for r in rs):
                        complete = False
                        break
                    if all((r.get(att) or 0) >= 0.9 and r['incomplete'] == 0 for r in rs):
                        sus = x
                    else:
                        break
                cap[p] = {'sustained_rps': sus, 'censored': sus == ladder[-1], 'complete_ladder': complete,
                          'step': ladder.index(sus) if sus is not None else -1}
                good[p] = {str(x): st.mean([cell[(x, sd)].get(gp) or 0 for sd in seeds if (x, sd) in cell])
                           for x in ladder if any((x, sd) in cell for sd in seeds)}
            m = sel[wl]['mixed']
            dec = {}
            for other in dict.fromkeys([sel[wl]['separate'], 'whole', 'mix2048']):
                if other == m or other not in cap or m not in cap:
                    continue
                d = cap[m]['step'] - cap[other]['step']
                note = []
                if cap[m]['censored'] and cap[other]['censored']:
                    note.append('both censored at the ladder top')
                if cap[m]['sustained_rps'] is None or cap[other]['sustained_rps'] is None:
                    note.append('below the lowest ladder rate: ' + ', '.join(x for x in (m, other) if cap[x]['sustained_rps'] is None))
                if not (cap[m]['complete_ladder'] and cap[other]['complete_ladder']):
                    note.append('incomplete ladder')
                dec[f'{m}_vs_{other}'] = {'steps': d, 'decision': 'win' if d >= 1 else 'loss' if d <= -1 else 'uncertain',
                                          'note': '; '.join(note)}
            res[name] = {'capacity': cap, 'goodput_mean': good, 'decision': dec}
        out[wl] = res
    return out


def lc3_decide(a):
    rows, prm = [], json.loads(Path(a.params).read_text())
    for f in a.tables:
        t = json.loads(Path(f).read_text())
        rows += t['serving']['rows']
    res = {'sources': a.tables, 'selection': prm['selection']}
    res['heldout'] = decide(rows, prm['selection'], 'lc5h')
    res['bursty'] = decide(rows, prm['selection'], 'lc5b')
    res['q7'] = decide(rows, prm['selection'], 'lc5r', 'q7')
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + '\n')
    for k in ('heldout', 'bursty', 'q7'):
        for wl, v in res[k].items():
            for sl in ('primary', 'stall1000'):
                caps = {p: c['sustained_rps'] for p, c in v[sl]['capacity'].items()}
                print(k, wl, sl, caps, {d: (x['decision'], x['note']) for d, x in v[sl]['decision'].items()})
    print('->', a.out)


def ld_rows(root, prm):
    """amendment 5 tables: one row per (step, rate) from cp_disagg traces; 10% warm-up excluded (cp_analyze.disagg_trace)"""
    plan = json.loads((root / 'launch_plan.json').read_text())
    jr = A.journal(root)
    rows, status = [], []
    stall = prm.get('stall_ms_grid', ())
    for s in plan['steps']:
        rec = A.final(jr.get(s['id'], []))
        status.append({'id': s['id'], 'status': 'complete' if rec and rec.get('command_completed') else
                       ('missing' if not rec else rec.get('failure_class')), 'attempt': (rec or {}).get('attempt'),
                       'message': (rec or {}).get('failure_message')})
        if not rec or not rec.get('command_completed'):
            continue
        run = A.run_dir(rec)
        dead = sorted(f.name for f in run.glob('server_w*.log') if 'EngineDeadError' in f.read_text(errors='replace'))
        status[-1]['engine_dead'] = dead  # a decode/prefill engine died mid-step (amendment 6): rates after it fail
        meta = json.loads((run / 'run.json').read_text())
        kv = meta['layout']['kv_bytes_per_token']
        for tr in meta['traces']:
            t = json.loads((run / tr['file']).read_text())
            r = {'step': s['id'], 'engine_dead': bool(dead), 'experiment': 'ld', 'workload': s['workload'], 'policy': s['deployment'],
                 'deployment': s['deployment'], 'prefill_workers': s.get('prefill_workers'), 'rate': t['rate'],
                 'seed': t['seed'], 'arrival': t['arrival'], 'model': 'q32', 'span_s': t['span_s'],
                 'unfinished': t['unfinished'], **A.disagg_trace(t, prm['slo'][s['workload']], kv, stall)}
            pf = [x for x in t['requests'] if x.get('prefill_s') is not None and x['tokens'] == x['max_tokens']]
            if pf:  # transfer + decode admission: first decode token after the prefill response
                r['kv_handoff_p50_s'] = A.pct([x['token_s'][0] - x['prefill_done_s'] for x in pf], .5)
                r['kv_handoff_p99_s'] = A.pct([x['token_s'][0] - x['prefill_done_s'] for x in pf], .99)
            rows.append(r)
    return rows, status


def ld_decide(rows):
    """amendment 5 decision: per workload and SLO, all-seed sustained rate per deployment (decide()); the best
    disaggregated ratio vs the best colocated deployment, plus every disaggregated ratio vs each colocated one."""
    out = {}
    for wl in sorted({r['workload'] for r in rows}):
        sub = [r for r in rows if r['workload'] == wl]
        deps = sorted({r['policy'] for r in sub})
        res = decide(sub, {wl: {'mixed': deps[0], 'separate': deps[0]}}, 'ld')[wl]
        for sl in ('primary', 'stall1000'):
            cap = res[sl]['capacity']
            colo = [d for d in deps if d.startswith('colo')]
            dis = [d for d in deps if d.startswith('d')]
            key = lambda d: (cap[d]['step'], d)  # noqa: E731
            dec = {}
            res[sl]['decision'] = dec
            if not colo or not dis:  # partial data (e.g. the smoke): capacities only, no comparison
                continue
            bc, bd = max(colo, key=key), max(dis, key=key)
            for x in dis + ([f'best:{bd}'] if dis else []):
                xd = x.split(':')[-1]
                for y in colo + ([f'best:{bc}'] if x.startswith('best') else []):
                    if x.startswith('best') != y.startswith('best'):
                        continue
                    yd = y.split(':')[-1]
                    d = cap[xd]['step'] - cap[yd]['step']
                    note = [n for n, c in (('both infeasible', cap[xd]['step'] < 0 and cap[yd]['step'] < 0),
                                           ('both censored', cap[xd]['censored'] and cap[yd]['censored'])) if c]
                    dec[f'{xd}_vs_{yd}' if not x.startswith('best') else f'best_disagg({xd})_vs_best_colocated({yd})'] = {
                        'steps': d, 'note': '; '.join(note),
                        'decision': 'uncertain' if note and d == 0 else 'win' if d >= 1 else 'loss' if d <= -1 else 'uncertain'}
        out[wl] = res
    return out


def ld_cmd(a):
    prm = json.loads(Path(a.params).read_text())
    out = Path(a.out or Path(a.root[-1]) / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    rows, status = [], []
    for rt in a.root:  # an ldfx re-run root (amendment 6) after the ld root: <step>_fx replaces <step>
        r1, s1 = ld_rows(Path(rt), prm)
        rows, status = rows + r1, status + s1
    redone = {r['step'][:-3] for r in rows if r['step'].endswith('_fx')}
    for x in status:
        x['replaced_by_fx'] = x['id'] in redone
    rows = [r for r in rows if r['step'] not in redone]
    res = {'status': status, 'rows': rows, 'decision': ld_decide(rows) if rows else {}}
    (out / 'ld_tables.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    for wl, v in res['decision'].items():
        for sl in ('primary', 'stall1000'):
            print(wl, sl, {p: c['sustained_rps'] for p, c in v[sl]['capacity'].items()},
                  {k: d['decision'] + (f" ({d['note']})" if d['note'] else '') for k, d in v[sl]['decision'].items()})
    print(sum(x['status'] == 'complete' for x in status), '/', len(status), 'steps complete;',
          'engine died in:', [x['id'] + (' (replaced)' if x['replaced_by_fx'] else '') for x in status if x.get('engine_dead')],
          '->', out / 'ld_tables.json')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = q.add_subparsers(dest='cmd', required=True)
    for n in ('lc1', 'smoke'):
        r = s.add_parser(n)
        r.add_argument('--root', required=True)
        r.add_argument('--out')
        r.add_argument('--s1-tables', default=str(Path(__file__).resolve().parents[3] /
                                                  'results_published/amd_20261118_mi300x_cp_s1/s1_tables.json'))
        r.add_argument('--model', choices=('q32', 'q7'), default='q32')
    for n in ('lc2', 'lc3'):
        r = s.add_parser(n)
        r.add_argument('--root', required=True)
        r.add_argument('--out')
        r.add_argument('--params', required=True)
    r = s.add_parser('ld')  # amendment 5: disaggregation at long prompts (amendment 6: --root ld_root ldfx_root)
    r.add_argument('--root', required=True, nargs='+')
    r.add_argument('--params', required=True)
    r.add_argument('--out')
    r = s.add_parser('decide')  # held-out decisions from one or more lc3 / lc3fx analysis tables
    r.add_argument('--tables', nargs='+', required=True)
    r.add_argument('--params', required=True)
    r.add_argument('--out', required=True)
    a = q.parse_args()
    if getattr(a, 'model', 'q32') == 'q7':  # Qwen2.5-7B: 28 layers x 4 KV heads x 128 x (K, V) x 2 B; 7.62 B params
        KV_TOK, WEIGHTS, LIN_FLOP_TOK = 57344, 15.23e9, 2 * 7.62e9
    (lc1 if a.cmd in ('lc1', 'smoke') else lc3_decide if a.cmd == 'decide' else ld_cmd if a.cmd == 'ld' else lc_serve_plan)(a)
