#!/usr/bin/env python3
"""E7 (CPU only): one-parameter service-response model of the CU-restriction penalty, fitted on a training
subset and checked on held-out models and CU levels, with blind predictions for levels not yet measured.

Model (conditional, fixed other cost): R(n) = T(n)/T(304) = (1 - s) + s * P_x(n), where P_x(n) is the native
penalty of one E4 service control x at the same CU allocation (metric(304)/metric(n) for rates,
metric(n)/metric(304) for latencies) and s is the fraction of decode time that scales like x. s and x are
statistical fit quantities, not a physical partition of decode time.

Predeclared procedure (written before the v1.3 extension jobs measure 272/248/216/184):
  training     Qwen32B production E2 point ratios at n = 232, 152, 96 (one cohort at a time)
  fit          s by least squares through the origin on (P_x - 1, R - 1); x chosen among all E4 rows by
               training RMSE (one parameter per row)
  held-out     Qwen32B n = 200 (same jobs, not used in the fit); Qwen14B at every level (other model)
  blind        Qwen32B n = 272, 248, 216, 184 with P_x linearly interpolated in n between measured E4 levels
  acceptance   a held-out or blind point is predicted when |R_obs - R_pred| <= 0.02 + (CI half-width of R_obs);
               the model is validated on a cohort only if every held-out and blind point is predicted

Revision r2 (made after seeing the unconstrained r1 fit, so its held-out checks on existing data are not
blind; only the fine-level predictions are): r1 selected latency rows and rows with s < 0 or s > 1, which
have no meaning as a fraction of decode time. r2 (--constrained) admits throughput controls only and requires
0 <= s <= 1. Both are reported.

  e7_model.py --cohort mi300x --e4 <auto tables.json> --e2 <production tables.json> [--e2-blind <extension tables.json>]
              --out e7_mi300x.json
"""
import argparse
import json
import math
from pathlib import Path

LATENCY = 'ns_per_dependent_load'
TRAIN, HELD_LEVEL, BLIND = (232, 152, 96), (200,), (272, 248, 216, 184)


def penalties(e4):
    rows = {}
    for name, lv in e4.get('e4', {}).get('e4', {}).items():  # production E4 (3 acquisitions), not e4_pilot
        if '304' not in lv:
            continue
        ref = lv['304']['mean']
        lat = lv['304'].get('metric') == LATENCY
        rows[name] = {int(n): (v['mean'] / ref if lat else ref / v['mean']) for n, v in lv.items() if v.get('mean')}
    return rows


def interp(p, n):
    if n in p:
        return p[n]
    xs = sorted(p)
    lo = max(x for x in xs if x < n)
    hi = min(x for x in xs if x > n)
    return p[lo] + (p[hi] - p[lo]) * (n - lo) / (hi - lo)


def observed(e2, model):
    lv = e2.get('e2', {}).get(f'e2:{model}', {}).get('levels', {})
    out = {}
    for n, e in lv.items():
        r = e.get('R')
        if r:
            out[int(n)] = (r['ratio'], (r['ci95'][1] - r['ci95'][0]) / 2)
    return out


def main(a):
    e4 = json.loads(Path(a.e4).read_text())
    e2 = json.loads(Path(a.e2).read_text())
    rows = penalties(e4)
    q32, q14 = observed(e2, 'qwen32b'), observed(e2, 'qwen14b')
    fits = {}
    lat = {name for name, lv in e4.get('e4', {}).get('e4', {}).items() if lv.get('304', {}).get('metric') == LATENCY}
    for name, p in rows.items():
        if a.constrained and name in lat:
            continue
        if not all(n in p for n in TRAIN) or not all(n in q32 for n in TRAIN):
            continue
        x = [p[n] - 1 for n in TRAIN]
        y = [q32[n][0] - 1 for n in TRAIN]
        den = sum(v * v for v in x)
        if den == 0:
            continue
        s = sum(u * v for u, v in zip(x, y)) / den
        if a.constrained and not 0 <= s <= 1:
            continue
        rmse = math.sqrt(sum((s * u - v) ** 2 for u, v in zip(x, y)) / len(x))
        fits[name] = {'s': s, 'train_rmse': rmse}
    best = min(fits, key=lambda k: fits[k]['train_rmse'])
    s, p = fits[best]['s'], rows[best]

    def pred(n):
        return 1 + s * (interp(p, n) - 1)

    def check(obs, levels, label):
        out = []
        for n in levels:
            if n not in obs:
                continue
            r, hw = obs[n]
            out.append({'set': label, 'n': n, 'observed': r, 'ci_half_width': hw, 'predicted': pred(n),
                        'residual': r - pred(n), 'predicted_within_tolerance': abs(r - pred(n)) <= 0.02 + hw})
        return out
    held = check(q32, HELD_LEVEL, 'held-out level (Qwen32B)') + check(q14, [n for n in sorted(q14) if n != 304],
                                                                          'held-out model (Qwen14B)')
    res = {'cohort': a.cohort, 'revision': 'r2 constrained' if a.constrained else 'r1 unconstrained', 'scope': __doc__.strip(), 'e4_source': a.e4, 'e2_source': a.e2,
           'training': {str(n): q32[n][0] for n in TRAIN}, 'fits': fits, 'selected_control': best, 's': s,
           'held_out': held,
           'blind_predictions': {str(n): pred(n) for n in BLIND},
           'blind_check': None}
    if a.e2_blind:
        blind_obs = observed(json.loads(Path(a.e2_blind).read_text()), 'qwen32b')
        res['blind_check'] = check(blind_obs, BLIND, 'blind level (Qwen32B)')
        res['e2_blind_source'] = a.e2_blind
    pts = held + (res['blind_check'] or [])
    res['validated'] = bool(res['blind_check']) and all(x['predicted_within_tolerance'] for x in pts)
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(f"{a.cohort}: control={best} s={s:.3f} train_rmse={fits[best]['train_rmse']:.4f}")
    for k in sorted(fits, key=lambda k: fits[k]['train_rmse'])[:5]:
        print(f"   {k:14s} s={fits[k]['s']:.3f} rmse={fits[k]['train_rmse']:.4f}")
    for x in pts:
        print(f"   {x['set']:28s} n={x['n']:3d} obs={x['observed']:.4f} pred={x['predicted']:.4f} "
              f"res={x['residual']:+.4f} ok={x['predicted_within_tolerance']}")
    print('   blind predictions:', {k: round(v, 4) for k, v in res['blind_predictions'].items()})


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--cohort', required=True)
    q.add_argument('--e4', required=True)
    q.add_argument('--e2', required=True)
    q.add_argument('--e2-blind')
    q.add_argument('--constrained', action='store_true', help='r2: throughput controls only, 0 <= s <= 1')
    q.add_argument('--out', required=True)
    main(q.parse_args())
