#!/usr/bin/env python3
"""OC6 secondary transfer check (execution plan §6; reported, not an acceptance criterion).

Frozen M1 and M6 (docs/amd/oc/oc6_frozen_<cohort>.json) predict wvSplitK projection durations; the observation is the
graph-mode Qwen2.5-7B B1 decode kernel trace (oc6_transfer, held-out job). Each decode_step_* marker range holds the
projection dispatches in layer order (qkv, o_proj, gate_up, down) x layers, then lm_head; shapes are assigned by that
position, and only ranges with exactly 4*layers+1 wvSplitK dispatches are used. References are the frozen (n, n) cell
at the traced grid; issued MFMA uses the frozen form. No refit.

  oc_transfer.py --frozen docs/amd/oc/oc6_frozen_<c>.json --trace-dir <attempt>/rocprof --config <HF config.json> --out f.json
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

ORDER = ('qkv', 'o_proj', 'gate_up', 'down')


def shapes(cfg):
    c = json.loads(Path(cfg).read_text())
    h, i, nh, kv = c['hidden_size'], c['intermediate_size'], c['num_attention_heads'], c['num_key_value_heads']
    hd = c.get('head_dim') or h // nh
    return {'qkv': (nh * hd + 2 * kv * hd, h), 'o_proj': (h, nh * hd), 'gate_up': (2 * i, h), 'down': (h, i),
            'lm_head': (c['vocab_size'], h)}, c['num_hidden_layers']


def main(a):
    fz = json.loads(a.frozen.read_text())
    sh, layers = shapes(a.config)
    mk = list(csv.DictReader(open(a.trace_dir / 'p_marker_api_trace.csv')))
    steps = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in mk if r['Function'].startswith('decode_step_'))
    ks = sorted(((int(r['Start_Timestamp']), int(r['End_Timestamp']), int(r['Grid_Size_X']) // int(r['Workgroup_Size_X']))
                 for r in csv.DictReader(open(a.trace_dir / 'p_kernel_trace.csv')) if 'wvSplitK' in r['Kernel_Name']))
    dur = {s: [] for s in sh}
    grids, used, j = set(), 0, 0
    for t0, t1 in steps:
        while j < len(ks) and ks[j][0] < t0:
            j += 1
        inr = []
        while j < len(ks) and ks[j][0] <= t1:
            inr.append(ks[j])
            j += 1
        if len(inr) != 4 * layers + 1:
            continue
        used += 1
        for p, (s0, s1, g) in enumerate(inr):
            dur['lm_head' if p == 4 * layers else ORDER[p % 4]].append((s1 - s0) * 1e-9)
            grids.add(g)
    g = grids.pop() if len(grids) == 1 else None
    r = fz['references'].get(f'{g}_{g}') if g else None
    rows = []
    for s, (m, k) in sh.items():
        if not dur[s] or not r:
            rows.append({'shape': s, 'status': 'unavailable', 'reason': 'no matching reference or dispatches'})
            continue
        C = 4 * m * math.ceil(k / 1024) / r['R_mfma4']
        M = 2 * m * k / r['B_ref']
        O = r['O_ref']
        T = st.median(dur[s])
        p1 = O + C + M - fz['eta'] * min(C, M)
        p6 = math.hypot(O, fz['kappa_m6'] * M)
        rows.append({'shape': s, 'm': m, 'k': k, 'dispatches': len(dur[s]), 'T_meas_median_s': T,
                     'T_meas_p10_p90_s': [st.quantiles(dur[s], n=10)[0], st.quantiles(dur[s], n=10)[-1]],
                     'C': C, 'M': M, 'O': O, 'T_pred_m1': p1, 'T_pred_m6': p6, 'rel_err_m1': p1 / T - 1, 'rel_err_m6': p6 / T - 1})
    ok = [x for x in rows if 'T_meas_median_s' in x]
    tick = {f'm{v}': (sum(x[f'T_pred_m{v}'] * (1 if x['shape'] == 'lm_head' else layers) for x in ok) /
                      sum(x['T_meas_median_s'] * (1 if x['shape'] == 'lm_head' else layers) for x in ok) - 1) for v in (1, 6)}
    out = {'cohort': fz['cohort'], 'frozen_note': fz['note'], 'grid_workgroups': g, 'decode_steps_used': used,
           'decode_steps_total': len(steps), 'layers': layers, 'rows': rows, 'per_tick_projection_rel_err': tick,
           'scope': 'secondary transfer check; graph-mode trace (profiled), not an acceptance criterion'}
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for x in ok:
        print(fz['cohort'], x['shape'], f"{x['T_meas_median_s']*1e6:.1f}us m1 {x['rel_err_m1']:+.3f} m6 {x['rel_err_m6']:+.3f}")
    print(fz['cohort'], 'grid', g, 'steps', used, '/', len(steps), 'per-tick', {k: round(v, 3) for k, v in tick.items()})


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--frozen', type=Path, required=True)
    q.add_argument('--trace-dir', type=Path, required=True)
    q.add_argument('--config', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
