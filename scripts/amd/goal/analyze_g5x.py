#!/usr/bin/env python3
"""goal G5 extension reader (gfx942): leak test v5 and the G5 surface. CPU only.
Pre-registered rule: docs/amd/goal_g5x_plan_20261107.md section 3 (frozen before data). Builds on
scripts/amd/cubw/analyze.py (acquisitions, start-delay probe counts) and scripts/amd/goal/analyze_goal.py (gates, G5).

Leak test v5, per GPU (lane) and level n (probe on n..FULL-1, decoder on 0..n-1):
  e_c        excess late fraction of probe CU c beside a run vs the probe alone (v4 definition)
  T8(run)    mean of the 8 largest e_c (8 = CUs a positive overlaps); M(run) = max e_c
  natives    non-overlapping co-runners on 0..n-1: mem100, arith100, storm_lds, storm_small
  storm valid  a storm's measured kernel launch rate >= the decoder's graph-decode kernel rate (DECODER_KPS)
  positives  cuprobe on 0..n+7 and the stock decoder on 0..n+7
  available  at least one valid storm
  sensitive  every positive: T8 >= max_native T8 + 0.10 and flags (e_c >= 0.10) >= 4
  ok         available and sensitive and T8(decoder) <= max_native T8 + (min_positive T8 - max_native T8) / 2
             and M(decoder) <= max_native M + 0.10
leak_ok_v5 = ok at every GPU and level. G5 disposition: 'isolated (leak v5)' if leak_ok_v5, else 'observational:
isolation not established' (mem100/arith100 reported as the contrast robust to shared-CU effects).

  analyze_g5x.py report --root <run root> [--out <dir>]
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

# graph-decode kernel launch rate of the tuned Qwen32B B1/C4096 decoder (goal_v1 G2 route traces, 63 decode steps,
# 731 kernels per step): 502425 n128/g128 and 502426 n160/g160; frozen in the plan before data
DECODER_KPS = {'mi300x': 39249.0, 'mi325x': 43725.0}
FLAG, POS_MIN, GAP, MARGIN_M = 0.10, 4, 0.10, 0.10


def storm_rate(rec):
    """kernels/s of a corunner from its own window lines (corunner.jsonl), steady windows only."""
    f = Path(rec['attempt_dir']) / 'corunner.jsonl'
    if not f.exists():
        return None
    wins = []
    for line in f.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if 't0_unix_ns' in d and 'launches' in d:
            wins.append(d)
    steady = wins[1:-1] or wins
    secs = sum((w['t1_unix_ns'] - w['t0_unix_ns']) / 1e9 for w in steady)
    return sum(w['launches'] for w in steady) / secs if secs else None


def leak_v5(plan, acqs, comp_only, attempts):
    full = plan['full_cus']
    kps = DECODER_KPS.get(plan['cohort'])
    steps = {s['id']: s for s in plan['steps']}
    out = {}
    dec = [a for a in acqs if a.get('competitor') and a['competitor'].get('mode') == 'gapprobe' and a['competitor'].get('gap')]
    for a in dec:
        mask = a['competitor']['mask'].split(':', 1)[1]
        lo, hi = (int(x) for x in mask.split('-'))
        if a['n'] != lo:
            continue
        rows = {k: v for k, v in comp_only.items() if v['probe_mask'] == mask and v['probe_mode'] == 'gapprobe'
                and v['lane'] == a['lane'] and v.get('gap')}
        alone = next((v['gap'] for v in rows.values() if not v['corunner']), None)
        r = {'lane': a['lane'], 'n': lo, 'probe_cus': hi - lo + 1, 'decoder_kps_reference': kps}
        if not alone:
            out[a['cell']] = dict(r, ok=False, reason='no probe-alone run')
            continue
        cus = sorted(alone['seen'])
        f0 = A.late_fraction(alone, cus)

        def stats(g):
            f = A.late_fraction(g, cus)
            e = sorted(f[c] - f0[c] for c in cus)
            return {'T8': st.mean(e[-8:]), 'M': e[-1], 'flags': sum(x >= FLAG for x in e), 'launches': g['launches']}
        nat, pos = {}, {}
        for k, v in rows.items():
            co = v['corunner']
            if not co:
                continue
            name = f"{co['treatment']}@{co['mask']}"
            if int(co['mask'].split('-')[1]) < lo:
                nat[name] = stats(v['gap'])
                if co['treatment'].startswith('storm'):
                    sid = next(s for s in steps.values() if s['kind'] == 'competitor_only' and s['cell'] == k)['id']
                    rec = A.final_attempt(attempts.get(sid, []))
                    nat[name]['kernels_per_s'] = storm_rate(rec) if rec else None
                    nat[name]['dispatch_matched'] = bool(kps and nat[name]['kernels_per_s'] and nat[name]['kernels_per_s'] >= kps)
            elif co['treatment'] == 'cuprobe':
                pos[name] = stats(v['gap'])
        for d in dec:
            if d['lane'] == a['lane'] and d['n'] > lo and d['competitor']['mask'] == a['competitor']['mask']:
                pos[f"stock_decoder@0-{d['n'] - 1}"] = stats(d['competitor']['gap'])
        D = stats(a['competitor']['gap'])
        r.update(natives=nat, positives=pos, decoder=D)
        if not nat or not pos:
            out[a['cell']] = dict(r, ok=False, reason='missing runs')
            continue
        nT8, nM = max(v['T8'] for v in nat.values()), max(v['M'] for v in nat.values())
        pT8 = min(v['T8'] for v in pos.values())
        r['available'] = any(v.get('dispatch_matched') for v in nat.values())
        r['sensitive'] = all(v['T8'] >= nT8 + GAP and v['flags'] >= POS_MIN for v in pos.values())
        r['threshold_T8'] = nT8 + (pT8 - nT8) / 2
        r['threshold_M'] = nM + MARGIN_M
        r['ok'] = bool(r['available'] and r['sensitive'] and D['T8'] <= r['threshold_T8'] and D['M'] <= r['threshold_M'])
        out[a['cell']] = r
    return out


def report(a):
    root = a.root
    out = a.out or root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    plan, acqs, nat, ptab, comp_only, attempts, samples = A.build(root)
    steps = {s['id']: s for s in plan['steps']}
    gates = AG.goal_gates(plan, acqs, ptab, comp_only)
    v5 = leak_v5(plan, acqs, comp_only, attempts)
    gates['leak_v5'] = v5
    gates['leak_v5_available'] = bool(v5) and all(v.get('available') for v in v5.values())
    gates['leak_v5_sensitive'] = bool(v5) and all(v.get('sensitive') for v in v5.values())
    gates['leak_ok_v5'] = bool(v5) and all(v.get('ok') for v in v5.values())
    iso = gates['placement_ok'] and gates['lifecycle_ok'] and gates['leak_ok_v5']
    g5 = AG.g5_surface(plan, acqs, steps)
    t = {'version': 'goal_g5x', 'cohort': plan['cohort'], 'plan_kind': plan['kind'], 'plan_sha256': plan.get('plan_sha256'),
         'gates': gates, 'placement': ptab, 'g5': g5,
         'g5_disposition': 'isolated (leak v5)' if iso else 'observational: isolation not established',
         'acquisitions': [{k: v for k, v in x.items() if k not in ('clock', 'windows')} for x in acqs]}
    (out / 'g5x_tables.json').write_text(json.dumps(t, indent=1, default=str) + '\n')
    L = [f"# G5 extension — {plan['cohort']} ({plan['kind']})", '',
         f"- placement_ok {gates['placement_ok']}, lifecycle_ok {gates['lifecycle_ok']}, leak_ok_v4 {gates.get('leak_ok_v4')}",
         f"- leak v5: available {gates['leak_v5_available']}, sensitive {gates['leak_v5_sensitive']}, ok **{gates['leak_ok_v5']}**",
         f"- G5 disposition: **{t['g5_disposition']}**", '', '## Leak v5 per GPU and level', '',
         '| cell | decoder T8 / M / flags | max native T8 / M | min positive T8 | thr T8 / M | storm kernels/s (matched) | ok |',
         '|---|---|---|---|---|---|---|']
    for k, r in sorted(v5.items()):
        if 'decoder' not in r:
            L.append(f"| {k} | — | — | — | — | — | {r.get('reason')} |")
            continue
        D, n, p = r['decoder'], r['natives'], r['positives']
        storms = ', '.join(f"{x.split('@')[0]} {v.get('kernels_per_s') or 0:.0f} ({v.get('dispatch_matched')})" for x, v in n.items() if x.startswith('storm'))
        L.append(f"| {k} | {D['T8']:.3f} / {D['M']:.3f} / {D['flags']} | {max(v['T8'] for v in n.values()):.3f} / "
                 f"{max(v['M'] for v in n.values()):.3f} | {min(v['T8'] for v in p.values()):.3f} | "
                 f"{r.get('threshold_T8', 0):.3f} / {r.get('threshold_M', 0):.3f} | {storms} | {r['ok']} |")
    L += ['', '## G5 (decoder time ratio vs competitor absent, same block)', '']
    for n, row in g5.items():
        cells = '; '.join(f"{tr} {AG.fmt(v['ratio_vs_absent'])}" + (f" ({v['competitor_gb_s']:.0f} GB/s)" if v.get('competitor_gb_s') else '')
                          for tr, v in row.items() if isinstance(v, dict) and v.get('ratio_vs_absent'))
        m = row.get('mem100_over_arith100')
        L.append(f"- n={n}: {cells}; mem100/arith100 {AG.fmt(m) if m else 'n/a'}")
    L += ['', 'Observational with respect to clock (controlled clocks not authorized).']
    (out / 'g5x_summary.md').write_text('\n'.join(L) + '\n')
    print(f"g5x tables written to {out}; leak_ok_v5={gates['leak_ok_v5']}")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = q.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('report')
    r.add_argument('--root', type=Path, required=True)
    r.add_argument('--out', type=Path)
    a = q.parse_args()
    report(a)
