#!/usr/bin/env python3
"""TP extension summary: wall ms/step per (TP size, case, variant) over complete acquisitions, speedup vs the
TP1 graph baseline (M1, HIP-event timing; TP uses synchronized wall, so ratios carry that scope difference),
and output-ID identity against TP1 (reduction order can legitimately change BF16 greedy tokens)."""
import glob, json, re, statistics as st, sys
from pathlib import Path
root = Path(sys.argv[1]); m1 = json.loads((root / 'm1/summary.json').read_text())['conditions']
tp1_ids = {}
for f in glob.glob(str(root / 'm1/*_graph_*_a0/run.json')):
    d = json.loads(Path(f).read_text())
    if d['status'] == 'complete':
        tp1_ids[re.sub(r'_a\d+$', '', Path(f).parent.name)] = d['cases'][0]['trials'][0]['output_sha256']
groups = {}
for f in glob.glob(str(root / 'tp/tp*_a[0-9]/run.json')):
    d = json.loads(Path(f).read_text()); m = re.match(r'tp(\d+)_(.+)_(full|matched)_a\d+$', Path(f).parent.name)
    tp, case, var = int(m.group(1)), m.group(2), m.group(3)
    g = groups.setdefault((case, tp, var), {'acq': [], 'statuses': [], 'sha': set()})
    g['statuses'].append(d['status'])
    if d['status'] == 'complete':
        g['acq'].append(st.median(t['ms_per_step_wall'] for t in d['cases'][0]['trials']))
        g['sha'] |= set(d['cases'][0]['output_sha256'])
out = {'classification': 'measured-real-GPU (owner-authorized TP extension)', 'scope': __doc__, 'rows': []}
for (case, tp, var), g in sorted(groups.items()):
    base = m1.get(case, {}).get('mean_ms_per_step')
    mean = st.mean(g['acq']) if g['acq'] else None
    row = {'case': case, 'tp': tp, 'variant': var, 'acquisitions': len(g['acq']), 'statuses': g['statuses'],
           'wall_ms_per_step': mean, 'tp1_ms_per_step': base, 'speedup_vs_tp1': base / mean if base and mean else None,
           'outputs_match_tp1': (g['sha'] == {tp1_ids.get(case)}) if g['sha'] else None}
    out['rows'].append(row)
    print(f"TP{tp} {case:26s} {var:8s} n={row['acquisitions']} ms={mean and round(mean,4)} tp1={base and round(base,4)} "
          f"x={row['speedup_vs_tp1'] and round(row['speedup_vs_tp1'],3)} sameIDs={row['outputs_match_tp1']}")
(root / 'tp/summary.json').write_text(json.dumps(out, indent=1, default=list) + '\n')
