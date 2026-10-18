#!/usr/bin/env python3
"""Summarize the M4/A-D/M7 service sweep: per condition and CU count, mean of acquisition medians,
min/max across the three acquisitions, physical L2->memory bytes from the counter acquisition,
and the ratio to the full-CU value (M7: does trimming CUs reduce independently attainable service?)."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics


def metric(res):
    for key in ('useful_gb_per_s', 'useful_tflop_per_s', 'ns_per_dependent_load'):
        if key in res:
            return key, res[key]
    return None, None


def main(args):
    rows = [json.loads(l) for l in open(args.input)]
    vals, counters, failures, stream_cus = collections.defaultdict(list), {}, [], {}
    for r in rows:
        base = re.sub(r'_cu\d+$', '', r['condition_id'])
        cus = r['cus_requested']
        if r['acquisition'] == 'counters':
            counters[(base, cus)] = {k: v.get('kb_sum_over_k_kernels') for k, v in r.get('counters', {}).items()}
            continue
        res = r.get('result')
        if not res or not res.get('check_passed', True):
            failures.append({'condition': r['condition_id'], 'acquisition': r['acquisition'],
                             'returncode': r.get('returncode'), 'error': r.get('error'), 'stderr': r.get('stderr', '')[-200:]})
            continue
        key, v = metric(res)
        vals[(base, cus)].append(v)
        vals[(base, cus, 'unit')] = key
    out = {'classification': 'measured-real-GPU (independent microbenchmarks)', 'conditions': [], 'failures': failures}
    bases = sorted({k[0] for k in vals if len(k) == 2})
    for base in bases:
        full = vals.get((base, 304))
        for cus in sorted({k[1] for k in vals if len(k) == 2 and k[0] == base}, reverse=True):
            v = vals[(base, cus)]
            unit = vals[(base, cus, 'unit')]
            mean = statistics.mean(v)
            row = {'condition': base, 'cus': cus, 'unit': unit, 'acquisitions': len(v), 'mean': mean,
                   'min': min(v), 'max': max(v),
                   'ratio_to_full_cu': mean / statistics.mean(full) if full else None,
                   'fetch_kb_counter_acq': counters.get((base, cus), {}).get('FETCH_SIZE'),
                   'write_kb_counter_acq': counters.get((base, cus), {}).get('WRITE_SIZE')}
            out['conditions'].append(row)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    for r in out['conditions']:
        print(f"{r['condition']:52s} cu={r['cus']:3d} n={r['acquisitions']} {r['mean']:10.2f} {r['unit']:22s} "
              f"x_full={r['ratio_to_full_cu']:.3f}" if r['ratio_to_full_cu'] else '')
    print('failures', len(failures))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
