#!/usr/bin/env python3
"""G8 held-out prediction model (goal follow-up, docs/amd/goal_based_followup_20261105.md §10). CPU only.

Model (frozen before any held-out acquisition):
    R(g) = 1 + k * max(0, B(g) - B0),   B(g) = HBM peak bandwidth / g   [GB/s per CU]
where g is the number of CUs that actually do the skinny-GEMM work (the shader-engine-balanced wvSplitK grid, which
equals the mask when the mask is balanced) and R is decode time relative to the stock full-device decoder in the same
block (Qwen32B graph B1/C4096). The architecture enters only through B(g) (peak bandwidth and CU count per cohort);
no cohort-specific offset is fitted. R < 1 seen on MI300X is treated as clock confounding, not as model signal.

Development data (already exposed, published): MI350X job 497082 (e6_balanced and e6_decode R at g = 224, 192, 160,
128) fits k and B0 by least squares over a B0 grid; MI300X job 499334 (g = 288..128, all R 0.993-0.998) is a
consistency check only (all its points lie below the fitted B0). MI325X has no balanced-grid data at all.

Held-out points (none measured before freezing): MI325X g in {288 (on the full 304-CU mask), 256, 224, 192, 160, 128, 96};
MI300X g in {160, 96}; MI350X g = 96. Observed values come from the G1 cells of the goal jobs (tuned grid = g).

Pre-registered verdict (judge): PASS iff every held-out |R_obs - R_pred| <= 0.05 AND the mean absolute error <= 0.025;
REJECT otherwise; UNAVAILABLE for points without >= 3 complete paired blocks. Clock confounding is reported, not
corrected. A rejected model is not refit on these points and called held out.

  g8_model.py freeze --out docs/amd/goal/g8_frozen_model.json
  g8_model.py judge --frozen <json> --observed <goal_tables.json>... --out <json>
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PUB = REPO / 'results_published'
PEAK = {'mi300x': 5300.0, 'mi325x': 6000.0, 'mi350x': 8000.0}   # GB/s, cohort.json hbm_peak_tb_s
MEASURED = {'mi300x': 4563.5, 'mi325x': 4950.1, 'mi350x': 6416.2}  # GB/s, seqread4g full device (cohort index)
FULL = {'mi300x': 304, 'mi325x': 304, 'mi350x': 256}
HELD_OUT = {'mi325x': [288, 256, 224, 192, 160, 128, 96], 'mi300x': [160, 96], 'mi350x': [96]}
INPUTS = ['amd_20261103_mi350x_cubw_v1_5_followup_job497082/hse_judgment.json',
          'amd_20261103_mi350x_cubw_v1_5_followup_job497082/cubw_tables.json',
          'amd_20261105_mi300x_cubw_v1_6_se_job499334/hse_judgment.json']


def dev_points():
    h = json.loads((PUB / INPUTS[0]).read_text())
    t = json.loads((PUB / INPUTS[1]).read_text())
    bal = {232: 224, 208: 192, 184: 160, 168: 160, 152: 128}
    pts = [('mi350x', bal[int(n)], v['R_balanced'][0], f'497082 e6_balanced mask {n}') for n, v in h['P3']['levels'].items()]
    for n, v in t['e2']['e6_decode:qwen32b']['levels'].items():
        if v.get('R'):
            pts.append(('mi350x', int(n), v['R']['ratio'], f'497082 e6_decode mask {n}'))
    j = json.loads((PUB / INPUTS[2]).read_text())
    for n, v in j['P3m']['a'].items():
        pts.append(('mi300x', int(n) - int(n) % 32, v['R_balanced']['ratio'], f'499334 e6_balanced mask {n}'))
    for n, v in j['P3m']['recorded_e6_decode'].items():
        pts.append(('mi300x', int(n), v['ratio'], f'499334 e6_decode mask {n}'))
    return pts


def predict(m, cohort, g, peak=PEAK):
    b = peak[cohort] / g
    return 1 + m['k'] * max(0.0, b - m['B0'])


def freeze(a):
    pts = dev_points()
    fit = [(PEAK[c] / g, r) for c, g, r, _ in pts if c == 'mi350x']
    best = None
    for i in range(3000, 6001):
        b0 = i / 100
        xs = [max(0.0, b - b0) for b, _ in fit]
        sxx = sum(x * x for x in xs)
        if not sxx:
            continue
        k = sum(x * (r - 1) for x, (_, r) in zip(xs, fit)) / sxx
        sse = sum((1 + k * x - r) ** 2 for x, (_, r) in zip(xs, fit))
        if best is None or sse < best[0]:
            best = (sse, b0, k)
    m = {'k': round(best[2], 6), 'B0': best[1], 'sse_dev_mi350x': best[0]}
    dev = [{'cohort': c, 'g': g, 'B_GBps_per_CU': round(PEAK[c] / g, 2), 'R_obs': r, 'R_pred': round(predict(m, c, g), 4),
            'source': s} for c, g, r, s in pts]
    held = [{'cohort': c, 'g': g, 'mask_n': FULL[c] if g == FULL[c] - FULL[c] % 32 else g, 'B_GBps_per_CU': round(PEAK[c] / g, 2),
             'R_pred': round(predict(m, c, g), 4),
             'R_pred_measured_bw_variant (not judged)': round(1 + m['k'] * max(0.0, MEASURED[c] / g - m['B0'] * MEASURED['mi350x'] / PEAK['mi350x']), 4)}
            for c, gs in HELD_OUT.items() for g in gs]
    out = {'model': 'R(g) = 1 + k*max(0, peak_bw/g - B0)', 'parameters': m, 'peak_GBps': PEAK, 'scope': __doc__.strip(),
           'inputs': {p: hashlib.sha256((PUB / p).read_bytes()).hexdigest() for p in INPUTS},
           'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'development': dev, 'held_out_predictions': held,
           'criteria': {'per_point_abs_error_max': 0.05, 'mean_abs_error_max': 0.025, 'min_paired_blocks': 3}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(m))
    for h in held:
        print(h)


def judge(a):
    fz = json.loads(a.frozen.read_text())
    obs = {}
    for f in a.observed:
        t = json.loads(Path(f).read_text())
        for row in t.get('g1', {}).get('tuned_vs_stock_full', []):
            obs[(t['cohort'], row['grid'], row['mask'])] = row
    rows, errs = [], []
    for h in fz['held_out_predictions']:
        o = obs.get((h['cohort'], h['g'], h['mask_n']))
        r = dict(h)
        if not o or (o.get('blocks') or 0) < fz['criteria']['min_paired_blocks']:
            r['status'] = 'unavailable'
        else:
            r.update(R_obs=o['ratio'], R_obs_ci95=o['ci95'], blocks=o['blocks'], abs_error=abs(o['ratio'] - h['R_pred']),
                     clock=o.get('clock'))
            r['status'] = 'measured'
            errs.append(r['abs_error'])
        rows.append(r)
    c = fz['criteria']
    if not errs:
        verdict = 'UNAVAILABLE'
    else:
        verdict = 'PASS' if (max(errs) <= c['per_point_abs_error_max'] and sum(errs) / len(errs) <= c['mean_abs_error_max']) else 'REJECT'
    out = {'verdict': verdict, 'measured_points': len(errs), 'held_out_points': len(rows),
           'mae': sum(errs) / len(errs) if errs else None, 'max_abs_error': max(errs) if errs else None,
           'frozen_sha256': hashlib.sha256(a.frozen.read_bytes()).hexdigest(), 'rows': rows}
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(verdict, out['mae'], out['max_abs_error'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--out', type=Path, required=True)
    j = sub.add_parser('judge'); j.add_argument('--frozen', type=Path, required=True)
    j.add_argument('--observed', nargs='+', required=True); j.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    freeze(a) if a.cmd == 'freeze' else judge(a)
