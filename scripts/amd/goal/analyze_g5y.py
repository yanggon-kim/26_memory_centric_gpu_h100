#!/usr/bin/env python3
"""goal G5 on gfx942 with leak test v6 (shader-engine-balanced start-delay probe): reader. CPU only.
Pre-registered rule: docs/amd/goal_g5y_plan_20261109.md section 3 (frozen before data). Builds on
scripts/amd/cubw/analyze.py (acquisitions, start-delay probe counts), scripts/amd/goal/analyze_goal.py (gates, G5) and
scripts/amd/goal/analyze_g5x.py (storm kernel rate).

Leak test v6, per GPU (lane), level n and balanced probe list s (scripts/amd/goal/cumap.py):
  valid      the probe alone covers exactly the CUs of its list, and on every CU (launches without a probe workgroup
             + launches with two or more) / launches <= 0.02: one workgroup per CU in every launch
  e_c        excess late fraction of probe CU c beside a run vs the probe alone (v4 definition)
  T8, M      mean of the 8 largest e_c; max e_c
  natives    non-overlapping co-runners on 0..n-1: mem100, arith100, storm_lds, storm_small
  positive   full-LDS cuprobe on 0..n-1 plus 8 CUs of list s
  sensitive  positive T8 >= max native T8 + 0.10 and flags (e_c >= 0.10) >= 4
  ok         valid and sensitive and T8(decoder) <= max native T8 + (positive T8 - max native T8) / 2
             and M(decoder) <= max native M + 0.10
leak_ok_v6 = ok for every GPU x level x list (a missing run is not ok). G5 disposition: 'isolated (leak v6)' if
placement_ok, lifecycle_ok and leak_ok_v6, else 'observational: isolation not established'.

  analyze_g5y.py report --root <run root> [--out <dir>]
"""
import argparse
import json
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cubw'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import analyze_goal as AG  # noqa: E402
from analyze_g5x import storm_rate, DECODER_KPS  # noqa: E402

FLAG, POS_MIN, GAP, MARGIN_M, BALANCE = 0.10, 4, 0.10, 0.10, 0.02
LEVELS, SUBSETS = (256, 192, 128), (0, 1)


def leak_v6(root, plan, acqs, comp_only, attempts):
    meta = {s['cell']: dict(s['leak6'], id=s['id']) for s in plan['steps'] if s.get('leak6')}
    dec = {a['cell']: a for a in acqs if a['cell'] in meta}
    lanes = sorted(plan['lanes'])
    out = {}
    for lane in lanes:
        cmap = root / 'cumap' / f'{lane}.json'
        cm = json.loads(cmap.read_text()) if cmap.exists() else {}
        for n in LEVELS:
            for j in SUBSETS:
                key = f'{lane}_n{n}_s{j}'
                r = {'lane': lane, 'n': n, 'subset': j, 'ok': False,
                     'list': cm.get('masks', {}).get(f'probe_n{n}_s{j}'), 'positive_mask': cm.get('masks', {}).get(f'pos_n{n}_s{j}')}
                out[key] = r
                cells = {c: m for c, m in meta.items() if m['n'] == n and m['subset'] == j and c.endswith('_' + lane)}
                alone = next((comp_only.get(c, {}).get('gap') for c, m in cells.items() if m['role'] == 'alone'), None)
                if not r['list'] or not alone:
                    r['reason'] = 'no balanced list' if not r['list'] else 'no probe-alone run'
                    continue
                size = len(r['list'].split(','))
                L = alone['launches']
                cus = sorted(alone['seen'])
                r['valid_detail'] = {'list_cus': size, 'alone_cus': len(cus),
                                     'max_missing': max((L - alone['seen'][c]) / L for c in cus),
                                     'max_multi': max(alone['multi'].get(c, 0) / L for c in cus), 'launches': L}
                v = r['valid_detail']
                r['valid'] = v['alone_cus'] == size and v['max_missing'] <= BALANCE and v['max_multi'] <= BALANCE
                f0 = A.late_fraction(alone, cus)

                def stats(g):
                    f = A.late_fraction(g, cus)
                    e = sorted(f[c] - f0[c] for c in cus)
                    return {'T8': st.mean(e[-8:]), 'M': e[-1], 'flags': sum(x >= FLAG for x in e), 'launches': g['launches']}
                nat, pos, D = {}, None, None
                for c, m in cells.items():
                    if m['role'] == 'native' and comp_only.get(c, {}).get('gap'):
                        nat[m['treatment']] = stats(comp_only[c]['gap'])
                        if m['treatment'].startswith('storm'):
                            rec = A.final_attempt(attempts.get(m['id'], []))
                            nat[m['treatment']]['kernels_per_s'] = storm_rate(rec) if rec else None
                    elif m['role'] == 'positive' and comp_only.get(c, {}).get('gap'):
                        pos = stats(comp_only[c]['gap'])
                    elif m['role'] == 'decoder' and c in dec and (dec[c].get('competitor') or {}).get('gap'):
                        D = stats(dec[c]['competitor']['gap'])
                r.update(natives=nat, positive=pos, decoder=D)
                if len(nat) < len(('mem100', 'arith100', 'storm_lds', 'storm_small')) or not pos or not D:
                    r['reason'] = 'missing runs'
                    continue
                nT8, nM = max(x['T8'] for x in nat.values()), max(x['M'] for x in nat.values())
                r['sensitive'] = pos['T8'] >= nT8 + GAP and pos['flags'] >= POS_MIN
                r['threshold_T8'] = nT8 + (pos['T8'] - nT8) / 2
                r['threshold_M'] = nM + MARGIN_M
                r['ok'] = bool(r['valid'] and r['sensitive'] and D['T8'] <= r['threshold_T8'] and D['M'] <= r['threshold_M'])
    return out


def report(a):
    root = a.root
    out = a.out or root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    plan, acqs, nat, ptab, comp_only, attempts, samples = A.build(root)
    steps = {s['id']: s for s in plan['steps']}
    gates = AG.goal_gates(plan, acqs, ptab, comp_only)
    v6 = leak_v6(root, plan, acqs, comp_only, attempts)
    gates['leak_v6'] = v6
    gates['leak_v6_valid'] = bool(v6) and all(v.get('valid') for v in v6.values())
    gates['leak_v6_sensitive'] = bool(v6) and all(v.get('sensitive') for v in v6.values())
    gates['leak_ok_v6'] = bool(v6) and all(v['ok'] for v in v6.values())
    gates['leak_ok_v6_by_level'] = {n: all(v['ok'] for v in v6.values() if v['n'] == n) for n in LEVELS}
    iso = gates['placement_ok'] and gates['lifecycle_ok'] and gates['leak_ok_v6']
    g5 = AG.g5_surface(plan, acqs, steps)
    t = {'version': 'goal_g5y', 'cohort': plan['cohort'], 'plan_kind': plan['kind'], 'plan_sha256': plan.get('plan_sha256'),
         'decoder_kps_reference': DECODER_KPS.get(plan['cohort']), 'gates': gates, 'placement': ptab, 'g5': g5,
         'g5_disposition': 'isolated (leak v6)' if iso else 'observational: isolation not established',
         'acquisitions': [{k: v for k, v in x.items() if k not in ('clock', 'windows')} for x in acqs]}
    (out / 'g5y_tables.json').write_text(json.dumps(t, indent=1, default=str) + '\n')
    L = [f"# G5 with leak test v6 — {plan['cohort']} ({plan['kind']})", '',
         f"- placement_ok {gates['placement_ok']}, lifecycle_ok {gates['lifecycle_ok']}",
         f"- leak v6: valid {gates['leak_v6_valid']}, sensitive {gates['leak_v6_sensitive']}, ok **{gates['leak_ok_v6']}** "
         f"(by level {gates['leak_ok_v6_by_level']})",
         f"- G5 disposition: **{t['g5_disposition']}**", '', '## Leak v6 per GPU, level and balanced list', '',
         '| GPU / n / list | CUs | valid (missing / multi) | decoder T8 / M / flags | max native T8 / M | positive T8 / flags | thr T8 / M | ok |',
         '|---|---|---|---|---|---|---|---|']
    for k, r in sorted(v6.items(), key=lambda x: (x[1]['lane'], -x[1]['n'], x[1]['subset'])):
        tag = f"{r['lane']} / {r['n']} / s{r['subset']}"
        if not r.get('decoder') or not r.get('natives') or not r.get('positive'):
            L.append(f"| {tag} | — | {r.get('valid', '—')} | — | — | — | — | {r.get('reason')} |")
            continue
        D, n, p, v = r['decoder'], r['natives'], r['positive'], r['valid_detail']
        L.append(f"| {tag} | {v['list_cus']} | {r['valid']} ({v['max_missing']:.3f} / {v['max_multi']:.3f}) | "
                 f"{D['T8']:.3f} / {D['M']:.3f} / {D['flags']} | {max(x['T8'] for x in n.values()):.3f} / "
                 f"{max(x['M'] for x in n.values()):.3f} | {p['T8']:.3f} / {p['flags']} | "
                 f"{r['threshold_T8']:.3f} / {r['threshold_M']:.3f} | {r['ok']} |")
    L += ['', '## G5 (decoder time ratio vs competitor absent, same block)', '']
    for n, row in g5.items():
        cells = '; '.join(f"{tr} {AG.fmt(v['ratio_vs_absent'])}" + (f" ({v['competitor_gb_s']:.0f} GB/s)" if v.get('competitor_gb_s') else '')
                          for tr, v in row.items() if isinstance(v, dict) and v.get('ratio_vs_absent'))
        m = row.get('mem100_over_arith100')
        L.append(f"- n={n}: {cells}; mem100/arith100 {AG.fmt(m) if m else 'n/a'}")
    L += ['', 'Observational with respect to clock (controlled clocks not authorized).']
    (out / 'g5y_summary.md').write_text('\n'.join(L) + '\n')
    print(f"g5y tables written to {out}; leak_ok_v6={gates['leak_ok_v6']}")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = q.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('report')
    r.add_argument('--root', type=Path, required=True)
    r.add_argument('--out', type=Path)
    a = q.parse_args()
    report(a)
