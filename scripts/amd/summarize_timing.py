#!/usr/bin/env python3
"""Summarize vLLM timing acquisitions (M1, M6) with clock joins and hierarchical bootstrap CIs.

Per trial: ms per decode step = HIP-event elapsed / decode iterations (63 for 64 tokens).
Clock qualification joins node-monitor samples (gfx/mem MHz) inside each trial's wall window.
A condition is clock-qualified when pooled in-window GFX and memory samples across all
acquisitions each stay within max/min <= 1.05, with no missing readings, as declared in
the common contract before looking at performance. Bootstrap: 10,000 draws, seed 20261018,
resampling acquisitions then trials within each acquisition.
"""
import argparse
import bisect
import collections
import json
from pathlib import Path
import random
import re
import statistics


def load_monitor(path):
    samples = collections.defaultdict(list)
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        samples[d['uuid']].append((d['t_unix_ns'], d.get('gfx_mhz'), d.get('mem_mhz'), d.get('power_mw')))
    for v in samples.values():
        v.sort()
    return samples


def window(samples, uuid, a, b):
    rows = samples.get(uuid, [])
    i = bisect.bisect_left(rows, (a,))
    j = bisect.bisect_right(rows, (b, float('inf')))
    return rows[i:j]


def bootstrap(acqs, draws=10000, seed=20261018):
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pick = [rng.choice(acqs) for _ in acqs]
        means.append(statistics.mean(statistics.mean(rng.choice(a) for _ in a) for a in pick))
    means.sort()
    return means[int(.025 * draws)], means[int(.975 * draws)]


def main(args):
    mon = load_monitor(args.monitor)
    groups = collections.defaultdict(list)
    for run in sorted(args.root.glob(args.pattern)):
        f = run / 'run.json'
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        key = re.sub(r'_a\d+$', '', run.name)
        groups[key].append((run.name, d))
    out = {'classification': 'measured-real-GPU', 'scope': __doc__.strip(), 'conditions': {}}
    for key, runs in sorted(groups.items()):
        acqs, clocks_g, clocks_m, rows, statuses = [], [], [], [], []
        inner_g, inner_m = [], []
        for name, d in runs:
            statuses.append(d['status'])
            if d['status'] != 'complete':
                continue
            uuid = d['hardware']['uuid']
            case = d['cases'][0]
            per = []
            for t in case['trials']:
                ms = t['cuda_event_ms'] / t['decode_iterations']
                per.append(ms)
                w = window(mon, uuid, t['wall_start_unix_ns'], t['wall_end_unix_ns'])
                clocks_g += [s[1] for s in w]
                clocks_m += [s[2] for s in w]
                span = t['wall_end_unix_ns'] - t['wall_start_unix_ns']
                wi = window(mon, uuid, t['wall_start_unix_ns'] + span // 10, t['wall_end_unix_ns'] - span // 10)
                inner_g += [s[1] for s in wi]; inner_m += [s[2] for s in wi]
            acqs.append(per)
            rows.append({'acquisition': name, 'median_ms_per_step': statistics.median(per), 'trials': len(per),
                         'outputs_identical': case['output_digests_identical'],
                         'batch': case['batch_size'], 'context': case['context_length']})
        if not acqs:
            out['conditions'][key] = {'statuses': statuses, 'qualified_acquisitions': 0}
            continue
        valid_g = [c for c in clocks_g if c]
        valid_m = [c for c in clocks_m if c]
        qualified = (len(valid_g) == len(clocks_g) > 0 and len(valid_m) == len(clocks_m) and
                     max(valid_g) / min(valid_g) <= 1.05 and max(valid_m) / min(valid_m) <= 1.05)
        batch = rows[0]['batch']
        mean = statistics.mean(statistics.mean(a) for a in acqs)
        entry = {'statuses': statuses, 'qualified_acquisitions': len(acqs), 'acquisitions': rows,
                 'mean_ms_per_step': mean, 'tokens_per_s': batch * 1000 / mean,
                 'gfx_mhz': {'n': len(clocks_g), 'min': min(valid_g, default=None), 'max': max(valid_g, default=None),
                             'median': statistics.median(valid_g) if valid_g else None},
                 'mem_mhz': {'min': min(valid_m, default=None), 'max': max(valid_m, default=None)},
                 'clock_qualified': qualified,
                 # Post-hoc diagnostic (declared after seeing v1 rejections; not the contract rule):
                 # same 1.05 envelope using only samples in the middle 80% of each trial window.
                 'clock_qualified_interior_posthoc': bool(inner_g and inner_m and all(inner_g) and all(inner_m)
                                                          and max(inner_g) / min(inner_g) <= 1.05 and max(inner_m) / min(inner_m) <= 1.05),
                 'interior_samples': len(inner_g)}
        if len(acqs) >= 3:
            entry['ci95_ms_per_step'] = bootstrap(acqs)
        out['conditions'][key] = entry
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out['conditions'].items():
        if v.get('qualified_acquisitions'):
            print(f"{k:40s} n={v['qualified_acquisitions']} {v['mean_ms_per_step']:.4f} ms "
                  f"CI={tuple(round(x, 4) for x in v.get('ci95_ms_per_step', ()))} tok/s={v['tokens_per_s']:.1f} "
                  f"gfx={v['gfx_mhz']['min']}-{v['gfx_mhz']['max']} clockOK={v['clock_qualified']}")
        else:
            print(f"{k:40s} {v['statuses']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pattern', default='*_a[0-9]')
    p.add_argument('--monitor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
