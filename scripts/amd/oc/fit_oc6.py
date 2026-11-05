#!/usr/bin/env python3
"""OC6 projection-family overlap model: fit on the development job, freeze, judge on the held-out job (CPU only).

Model (docs/amd/overlap_calibration_execution_plan_20261105.md section 6), decode projection wvSplitK bf16, N = 1:

    C = issued_MFMA(m, k, g) / R_mfma4(n, g)     R_mfma4: mfma4 wave-instructions per second at (mask n, grid g)
    M = W(m, k) / B_ref(n, g)                     W = 2 m k bytes, B_ref: matched streaming read
    O = O_ref(n, g)                               tiny wvSplitK (64 x 256) per-call time
    T_pred = O + C + M - eta * min(C, M)

Every input is the median over acquisitions of per-acquisition medians (10 trials each). issued_MFMA comes from the
per-dispatch SQ_INSTS_MFMA rows (`oc6_counts`), fitted on geometry; the pre-registered form a*m*k + b*g is used when
its largest relative residual is <= 1 %, otherwise the extended form a*m*k + b*m + c*g + d is used and flagged.
eta: one per cohort, least squares on log time over the development cells (grid search on [-0.5, 1.5]), with the
profile SSE, an approximate 95 % profile interval and the (O-scale, eta) trade-off over O x {0.5 .. 1.5}.

  fit_oc6.py fit   --root <dev run root> --cohort mi300x --out docs/amd/oc/oc6_frozen_mi300x.json
  fit_oc6.py judge --root <held-out run root> --frozen docs/amd/oc/oc6_frozen_mi300x.json --out validation.json
"""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oc_att  # noqa: E402

MFMA4_FLOPS = 2048
CELL = re.compile(r'_n(\d+)_g(\d+)(?:_a(\d+))?$')


def jl(p):
    lines = [x for x in Path(p).read_text().splitlines() if x.strip().startswith('{')]
    return json.loads(lines[-1]) if lines else None


def attempts(root):
    out = {}
    for f in sorted((root / 'journal').glob('*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get('step') and 't_end' in r:
                out[r['step']] = r
    return out


def lstsq(X, y):
    """Normal equations (small, well-conditioned systems only)."""
    n = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(n)]
    for i in range(n):  # Gauss-Jordan
        p = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        for r in range(n):
            if r != i and A[i][i]:
                f = A[r][i] / A[i][i]
                A[r] = [x - f * z for x, z in zip(A[r], A[i])]
                b[r] -= f * b[i]
    return [b[i] / A[i][i] for i in range(n)]


def load(root):
    plan = json.loads((root / 'launch_plan.json').read_text())
    att = attempts(root)
    refs = collections.defaultdict(lambda: collections.defaultdict(list))  # (n, g) -> kind -> [per-acq median]
    kern = collections.defaultdict(list)  # (model, shape, n, g) -> [per-acq median_us]
    meta = {}
    counts = []
    att_dirs = []
    for s in plan['steps']:
        rec = att.get(s['id'])
        if not rec or rec.get('returncode') != 0:
            continue
        d = Path(rec['attempt_dir'])
        m = CELL.search(s.get('cell', ''))
        if s['experiment'] == 'oc23_refs' and m and (d / 'out.json').exists():
            r = jl(d / 'out.json')
            n, g = int(m.group(1)), int(m.group(2))
            if r and r.get('mode') == 'mfma4':
                refs[(n, g)]['R_mfma4'].append(r['useful_tflop_per_s'] * 1e12 / MFMA4_FLOPS)
            elif r and r.get('mode') == 'read':
                refs[(n, g)]['B_ref'].append(r['useful_gb_per_s'] * 1e9)
        elif s['experiment'] in ('oc6_kernels', 'oc6_heldout_kernels') and m and (d / 'k.json').exists():
            n, g = int(m.group(1)), int(m.group(2))
            for r in json.loads((d / 'k.json').read_text())['rows']:
                if r.get('error') or r.get('route') != 'wvSplitK':
                    continue
                if r['shape'] == 'tiny':
                    refs[(n, g)]['O_ref'].append(r['median_us'] * 1e-6)
                elif r['history'] == 'stream':
                    kern[(r['model'], r['shape'], n, g)].append(r['median_us'] * 1e-6)
                    meta[(r['model'], r['shape'])] = (r['m'], r['k'])
        elif s['experiment'] == 'oc6_counts':
            for f in d.glob('oc_counters.json'):
                for run in json.loads(f.read_text())['runs']:
                    if run.get('group') != 'mfma' or not run.get('call_order'):
                        continue
                    rows, calls = run.get('dispatch_rows') or [], run['call_order']
                    ok = len(rows) == len(calls)
                    for c, r in zip(calls, rows if ok else []):
                        counts.append(dict(c, mask=s.get('level'), mfma=r['counters'].get('SQ_INSTS_MFMA'), kernel=r['kernel']))
                    if not ok:
                        counts.append({'error': f'{len(rows)} dispatch rows for {len(calls)} calls', 'mask': s.get('level')})
        elif s['experiment'] == 'oc4_att':
            att_dirs.append((s.get('level'), d))
    return plan, refs, kern, meta, counts, att_dirs


def fit_issued(counts):
    pts = [c for c in counts if c.get('mfma') is not None]
    if not pts:
        return {'status': 'unavailable'}
    out = {'n_points': len(pts), 'forms': {}}
    for name, feats in (('a*m*k + b*g', lambda c: [c['m'] * c['k'], c['grid']]),
                        ('a*m*k + b*m + c*g + d', lambda c: [c['m'] * c['k'], c['m'], c['grid'], 1.0])):
        X = [feats(c) for c in pts]
        y = [c['mfma'] for c in pts]
        coef = lstsq(X, y)
        res = [abs(sum(a * b for a, b in zip(coef, x)) / v - 1) for x, v in zip(X, y)]
        out['forms'][name] = {'coef': coef, 'max_rel_residual': max(res), 'median_rel_residual': st.median(res)}
    pre = out['forms']['a*m*k + b*g']
    out['used'] = 'a*m*k + b*g' if pre['max_rel_residual'] <= 0.01 else 'a*m*k + b*m + c*g + d'
    out['amended'] = out['used'] != 'a*m*k + b*g'
    by_mask = collections.defaultdict(dict)
    for c in pts:
        by_mask[(c['model'], c['shape'], c['grid'])][c['mask']] = c['mfma']
    out['mask_independent'] = all(len(set(v.values())) == 1 for v in by_mask.values() if len(v) > 1)
    out['measured'] = pts
    return out


def issued(fi, m, k, g):
    coef = fi['forms'][fi['used']]['coef']
    x = [m * k, g] if fi['used'] == 'a*m*k + b*g' else [m * k, m, g, 1.0]
    return sum(a * b for a, b in zip(coef, x))


def cells(refs, kern, meta, fi, measured_counts=True):
    lookup = {(c['model'], c['shape'], c['grid']): c['mfma'] for c in fi.get('measured', []) if c.get('mask') is not None}
    out = []
    for (model, shape, n, g), ts in sorted(kern.items()):
        r = refs.get((n, g), {})
        if not (r.get('R_mfma4') and r.get('B_ref') and r.get('O_ref')):
            out.append({'model': model, 'shape': shape, 'n': n, 'g': g, 'status': 'missing reference'})
            continue
        m, k = meta[(model, shape)]
        I = lookup.get((model, shape, g)) if measured_counts else None
        src = 'measured' if I else 'fitted'
        I = I or issued(fi, m, k, g)
        R, B, O = st.median(r['R_mfma4']), st.median(r['B_ref']), st.median(r['O_ref'])
        out.append({'model': model, 'shape': shape, 'm': m, 'k': k, 'n': n, 'g': g, 'T_meas': st.median(ts), 'T_acq': ts,
                    'issued_mfma': I, 'issued_source': src, 'R_mfma4': R, 'B_ref': B, 'C': I / R, 'M': 2 * m * k / B, 'O': O,
                    'acq': len(ts)})
    return out


def pred(c, eta, oscale=1.0):
    return c['O'] * oscale + c['C'] + c['M'] - eta * min(c['C'], c['M'])


def sse(cs, eta, oscale=1.0):
    return sum((math.log(pred(c, eta, oscale)) - math.log(c['T_meas'])) ** 2 for c in cs)


def fit_eta(cs, oscale=1.0):
    grid = [-0.5 + i * 0.001 for i in range(2001)]
    best = min(grid, key=lambda e: sse(cs, e, oscale))
    return best, sse(cs, best, oscale)


def att_shapes(att_dirs):
    out = []
    for level, d in att_dirs:
        for route in sorted(d.glob('att_gemv_*')):
            for u in sorted(route.glob('ui_output_agent_*_dispatch_*')):
                r = oc_att.dispatch(u)
                for cu, v in r['cus'].items():
                    if v['mfma_instructions']:
                        out.append({'mask': level, 'route': route.name, 'cu': cu, 'A_C_over_T': v['A_C_over_T'],
                                    'A_M_over_T': v['A_M_over_T'], 'X_over_T': v['X_over_T'],
                                    'X_over_A_C': v['X'] / v['A_C'] if v['A_C'] else None,
                                    'loads_retired_without_stall': v['loads_retired_without_stall']})
        for route in sorted(d.glob('att_native_split_*')):
            for u in sorted(route.glob('ui_output_agent_*_dispatch_*')):
                r = oc_att.dispatch(u)
                for cu, v in r['cus'].items():
                    if v['mfma_instructions'] or v['vmem_loads']:
                        out.append({'mask': level, 'route': route.name, 'cu': cu, 'both_types': bool(v['mfma_instructions'] and v['vmem_loads']),
                                    'A_C_over_T': v['A_C_over_T'], 'A_M_over_T': v['A_M_over_T'], 'X_over_T': v['X_over_T']})
    return out


def do_fit(a):
    plan, refs, kern, meta, counts, att_dirs = load(a.root)
    fi = fit_issued(counts)
    cs = [c for c in cells(refs, kern, meta, fi) if 'T_meas' in c]
    eta, s0 = fit_eta(cs)
    n = len(cs)
    thr = s0 * (1 + 3.84 / max(n - 1, 1))
    prof = {round(-0.5 + i * 0.05, 2): sse(cs, -0.5 + i * 0.05) for i in range(41)}
    inside = [e for e in (-0.5 + i * 0.001 for i in range(2001)) if sse(cs, e) <= thr]
    trade = {s: dict(zip(('eta', 'sse'), fit_eta(cs, s))) for s in (0.5, 0.75, 1.0, 1.25, 1.5)}
    for c in cs:
        c['T_pred'] = pred(c, eta)
        c['rel_err'] = c['T_pred'] / c['T_meas'] - 1
        c['feasible'] = max(c['C'], c['M']) <= c['T_meas'] - c['O'] <= c['C'] + c['M']
        c['C_over_M'] = c['C'] / c['M']
    frozen = {
        'version': 'oc6_v0', 'cohort': a.cohort, 'family': 'decode projection, wvSplitK bf16, N = 1',
        'model': 'T_pred = O_ref(n,g) + issued_MFMA(m,k,g)/R_mfma4(n,g) + 2mk/B_ref(n,g) - eta*min(C, M)',
        'dev_root': str(a.root), 'dev_plan_sha256': plan.get('plan_sha256'), 'dev_cells': plan.get('dev_cells'),
        'eta': eta, 'sse_log': s0, 'n_cells': n, 'eta_profile_95': [min(inside), max(inside)] if inside else None,
        'eta_profile_sse': prof, 'o_scale_tradeoff': trade, 'issued_mfma_fit': {k: v for k, v in fi.items() if k != 'measured'},
        'issued_mfma_measured': fi.get('measured'),
        'references': {f'{n}_{g}': {k: st.median(v) for k, v in r.items()} for (n, g), r in sorted(refs.items())},
        'references_acq': {f'{n}_{g}': r for (n, g), r in sorted(refs.items())},
        'dev_fit': {'cells': cs, 'median_signed_err': st.median(c['rel_err'] for c in cs) if cs else None,
                    'within_10pct': sum(abs(c['rel_err']) <= 0.10 for c in cs), 'infeasible': [f"{c['model']}:{c['shape']}:{c['n']}/{c['g']}" for c in cs if not c['feasible']]},
        'att_per_shape': att_shapes(att_dirs),
        'acceptance': 'held-out: >= 90 % of cells |T_pred/T_meas - 1| <= 0.10 and median signed error within +-5 %',
        'note': 'frozen before any held-out acquisition; no refit on held-out data',
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(frozen, indent=1, default=str) + '\n')
    print(f"{a.cohort}: eta={eta:.3f} 95%~{frozen['eta_profile_95']} n={n} within10={frozen['dev_fit']['within_10pct']}/{n} "
          f"median_err={frozen['dev_fit']['median_signed_err']} issued={fi.get('used')} max_res="
          f"{fi['forms'][fi['used']]['max_rel_residual'] if fi.get('forms') else None}")


def do_judge(a):
    fz = json.loads(a.frozen.read_text())
    plan, refs, kern, meta, counts, _ = load(a.root)
    fi = dict(fz['issued_mfma_fit'], measured=[])
    cs = [c for c in cells(refs, kern, meta, fi, measured_counts=False) if 'T_meas' in c]
    for c in cs:
        c['T_pred'] = pred(c, fz['eta'])
        c['rel_err'] = c['T_pred'] / c['T_meas'] - 1
        c['pass'] = abs(c['rel_err']) <= 0.10
    chk = []
    for x in counts:  # issued-MFMA function vs held-out measured counts (reported, not a criterion)
        if x.get('mfma'):
            chk.append({**{k: x[k] for k in ('model', 'shape', 'grid', 'mask', 'mfma')}, 'fitted': issued(fi, x['m'], x['k'], x['grid'])})
    per_model = collections.defaultdict(lambda: [0.0, 0.0])
    for c in cs:
        per_model[(c['model'], c['n'], c['g'])][0] += c['T_pred']
        per_model[(c['model'], c['n'], c['g'])][1] += c['T_meas']
    npass = sum(c['pass'] for c in cs)
    med = st.median(c['rel_err'] for c in cs) if cs else None
    res = {'frozen': str(a.frozen), 'heldout_root': str(a.root), 'heldout_plan_sha256': plan.get('plan_sha256'),
           'eta': fz['eta'], 'cells': cs, 'n_cells': len(cs), 'n_pass': npass, 'median_signed_err': med,
           'worst': max(cs, key=lambda c: abs(c['rel_err'])) if cs else None,
           'per_model_aggregate': [{'model': k[0], 'n': k[1], 'g': k[2], 'pred': v[0], 'meas': v[1], 'rel_err': v[0] / v[1] - 1}
                                   for k, v in sorted(per_model.items())],
           'issued_check': chk,
           'accepted': bool(cs) and npass >= 0.9 * len(cs) and abs(med) <= 0.05}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(f"judge: {npass}/{len(cs)} within 10 %, median signed {med}, accepted={res['accepted']}")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = q.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--root', type=Path, required=True)
    f.add_argument('--cohort', required=True)
    f.add_argument('--out', type=Path, required=True)
    j = sub.add_parser('judge')
    j.add_argument('--root', type=Path, required=True)
    j.add_argument('--frozen', type=Path, required=True)
    j.add_argument('--out', type=Path, required=True)
    a = q.parse_args()
    do_fit(a) if a.cmd == 'fit' else do_judge(a)
