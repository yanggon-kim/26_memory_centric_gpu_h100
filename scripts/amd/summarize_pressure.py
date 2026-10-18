#!/usr/bin/env python3
"""M5 summary: per case, paired normal-vs-pressure step-time ratio per acquisition (interleaved trials)."""
import glob, json, re, statistics as st, sys
from pathlib import Path
root = Path(sys.argv[1]); out = {'classification': 'measured-real-GPU (HF compiled StaticCache path)', 'cases': {}}
for f in sorted(glob.glob(str(root / '*_a[0-9].json'))):
    d = json.load(open(f)); case = re.sub(r'_a\d+\.json$', '', Path(f).name)
    med = lambda t: st.median([x for r in d['trials'] if r['treatment'] == t for x in r['step_ms']])
    row = out['cases'].setdefault(case, {'acquisitions': []})
    row['acquisitions'].append({'acquisition': d['acquisition'], 'outputs_identical': d['outputs_identical'],
        'normal_ms': med('normal'), 'pressure_ms': med('pressure'), 'ratio': med('pressure') / med('normal'),
        'pressure_read_ms': st.median([x for r in d['trials'] for x in r['pressure_ms']])})
for case, row in out['cases'].items():
    r = [a['ratio'] for a in row['acquisitions']]
    row.update(mean_ratio=st.mean(r), min_ratio=min(r), max_ratio=max(r), n=len(r))
    print(f"{case:22s} n={len(r)} normal={st.mean(a['normal_ms'] for a in row['acquisitions']):.4f}ms "
          f"pressure/normal={row['mean_ratio']:.4f} [{row['min_ratio']:.4f},{row['max_ratio']:.4f}] identical={all(a['outputs_identical'] for a in row['acquisitions'])}")
(root / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
