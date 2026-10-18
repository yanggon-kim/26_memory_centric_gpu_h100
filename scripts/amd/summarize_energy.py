#!/usr/bin/env python3
"""M8 summary: qualified decode-only J/token per case (amdsmi energy accumulator; whole-GPU socket scope),
raw counter and sampled-power estimates, agreement, window and average power. NA trials stay NA."""
import glob, json, re, statistics as st, sys
from pathlib import Path
root = Path(sys.argv[1]); out = {'classification': 'measured-real-GPU energy (socket/board scope, decode only, no idle subtraction)', 'cases': {}}
for f in sorted(glob.glob(str(root / '*_a[0-9]/run.json'))):
    d = json.loads(Path(f).read_text()); case = re.sub(r'_a\d+$', '', Path(f).parent.name)
    row = out['cases'].setdefault(case, {'acquisitions': [], 'statuses': []}); row['statuses'].append(d['status'])
    if d['status'] != 'complete':
        continue
    c = d['cases'][0]
    for t in c['trials']:
        s = t['telemetry']
        row['acquisitions'].append({'acquisition': Path(f).parent.name, 'jpt': s['joules_per_token'],
            'counter_j': s['counter_j'], 'integrated_j': s['integrated_power_j'], 'diff': s['counter_power_relative_difference'],
            'window_s': s['window_s'], 'avg_w': s['counter_j'] / s['window_s'] if isinstance(s['counter_j'], float) else None,
            'tokens': t['timed_generated_tokens'], 'ms_per_step': t['cuda_event_ms'] / t['decode_iterations']})
for case, row in out['cases'].items():
    q = [a['jpt'] for a in row['acquisitions'] if a['jpt'] != 'NA']
    acq = sorted({a['acquisition'] for a in row['acquisitions']})
    row.update(qualified_trials=len(q), acquisitions_complete=len(acq), mean_j_per_token=st.mean(q) if q else 'NA',
               min_j=min(q) if q else 'NA', max_j=max(q) if q else 'NA',
               mean_avg_w=st.mean(a['avg_w'] for a in row['acquisitions'] if a['avg_w']) if row['acquisitions'] else 'NA',
               max_diff=max((a['diff'] for a in row['acquisitions'] if isinstance(a['diff'], float)), default='NA'))
    print(f"{case:28s} acq={len(acq)} trials={len(q)} J/tok={row['mean_j_per_token'] if q == [] else round(row['mean_j_per_token'], 4)} "
          f"[{row['min_j'] if not q else round(row['min_j'], 4)},{row['max_j'] if not q else round(row['max_j'], 4)}] "
          f"W={row['mean_avg_w'] if row['mean_avg_w'] == 'NA' else round(row['mean_avg_w'])} maxdiff={row['max_diff']}")
(root / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
