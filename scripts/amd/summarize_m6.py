#!/usr/bin/env python3
"""M6 + prefill: decode and prefill slowdown vs CU restriction (same GPU per case), with median GFX clock.
Ratios are means of acquisition medians; clock medians come from the node monitor windows (summary.json)."""
import glob, json, re, statistics as st, sys
from pathlib import Path
root = Path(sys.argv[1]); m6 = json.loads((root / 'm6_cumask/summary.json').read_text())['conditions']
out = {'classification': 'measured-real-GPU', 'decode': {}, 'prefill': {}, 'scope': __doc__}
for k, v in m6.items():
    m = re.match(r'(.+)_cu(\d+)$', k)
    if not m or not v.get('mean_ms_per_step'):
        continue
    out['decode'].setdefault(m.group(1), {})[int(m.group(2))] = {'ms': v['mean_ms_per_step'], 'ci95': v.get('ci95_ms_per_step'),
        'n': v['qualified_acquisitions'], 'gfx_median': v['gfx_mhz']['median']}
for case, d in out['decode'].items():
    if 304 in d:
        for cu, r in d.items():
            r['slowdown_vs_304'] = r['ms'] / d[304]['ms']
pre = {}
for f in glob.glob(str(root / 'm8_prefill/*_a[0-9]/run.json')):
    d = json.loads(Path(f).read_text())
    if d['status'] != 'complete':
        continue
    m = re.match(r'(.+)_cu(\d+)_a\d+$', Path(f).parent.name)
    pre.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append(st.median(t['prefill_wall_ms'] for t in d['cases'][0]['trials']))
for case, d in pre.items():
    out['prefill'][case] = {cu: {'ms': st.mean(v), 'n': len(v), 'slowdown_vs_304': st.mean(v) / st.mean(d[304]) if 304 in d else None}
                            for cu, v in sorted(d.items(), reverse=True)}
(root / 'analysis/m6_prefill_summary.json').write_text(json.dumps(out, indent=1) + '\n')
for kind in ('decode', 'prefill'):
    print(f'--- {kind}')
    for case, d in sorted(out[kind].items()):
        print(f"{case:28s}", '  '.join(f"cu{cu}:{r['slowdown_vs_304']:.3f}(n{r['n']}{',' + str(int(r['gfx_median'])) + 'MHz' if r.get('gfx_median') else ''})"
                                        for cu, r in sorted(d.items(), reverse=True) if r.get('slowdown_vs_304')))
