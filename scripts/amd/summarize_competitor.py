#!/usr/bin/env python3
"""J6 summary: decoder ms/step on fixed CUs 0-231 vs an owned competitor on CUs 232-303.

For each case and treatment: mean of acquisition medians (HIP-event ms/step), ratio to the solo
treatment, and the competitor's achieved bandwidth/TFLOP/s averaged over windows that overlap the
decoder's measured trials (concurrency evidence). Shared-service sensitivity only; not a physical
change of memory interfaces.
"""
import glob
import json
from pathlib import Path
import re
import statistics as st
import sys


def main(root):
    rows = {}
    for f in glob.glob(str(root / 'j6_competitor/*_a[0-9]/run.json')):
        run_dir = Path(f).parent
        m = re.match(r'(.+)_(solo|read\d+|fma\d+)_a(\d+)$', run_dir.name)
        if not m:
            continue
        case, treat = m.group(1), m.group(2)
        d = json.loads(Path(f).read_text())
        cell = rows.setdefault(case, {}).setdefault(treat, {'ms': [], 'gbps': [], 'tflops': [], 'statuses': []})
        cell['statuses'].append(d['status'])
        if d['status'] != 'complete':
            continue
        trials = d['cases'][0]['trials']
        cell['ms'].append(st.median(t['cuda_event_ms'] / t['decode_iterations'] for t in trials))
        comp = run_dir / 'competitor.jsonl'
        if comp.exists():
            wins = [json.loads(l) for l in comp.read_text().splitlines() if l.startswith('{')]
            hit = [w for w in wins for t in trials if w['t0_unix_ns'] < t['wall_end_unix_ns'] and w['t1_unix_ns'] > t['wall_start_unix_ns']]
            if hit:
                cell['gbps'].append(st.mean(w['gb_per_s'] for w in hit))
                cell['tflops'].append(st.mean(w['tflop_per_s'] for w in hit))
    out = {'classification': 'measured-real-GPU (owned competitor, disjoint CUs)', 'scope': __doc__.strip(), 'cases': {}}
    for case, treats in sorted(rows.items()):
        solo = st.mean(treats['solo']['ms']) if treats.get('solo', {}).get('ms') else None
        out['cases'][case] = {}
        for t, c in sorted(treats.items()):
            mean = st.mean(c['ms']) if c['ms'] else None
            out['cases'][case][t] = {'acquisitions': len(c['ms']), 'statuses': c['statuses'], 'ms_per_step': mean,
                                     'ratio_to_solo': mean / solo if mean and solo else None,
                                     'competitor_gb_per_s_during_trials': st.mean(c['gbps']) if c['gbps'] else None,
                                     'competitor_tflop_per_s_during_trials': st.mean(c['tflops']) if c['tflops'] else None}
            r = out['cases'][case][t]
            print(f"{case:26s} {t:8s} n={r['acquisitions']} ms={mean and round(mean, 4)} x_solo={r['ratio_to_solo'] and round(r['ratio_to_solo'], 3)} "
                  f"compGB/s={r['competitor_gb_per_s_during_trials'] and round(r['competitor_gb_per_s_during_trials'])} "
                  f"compTF={r['competitor_tflop_per_s_during_trials'] and round(r['competitor_tflop_per_s_during_trials'], 1)}")
    (root / 'j6_competitor/summary.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
