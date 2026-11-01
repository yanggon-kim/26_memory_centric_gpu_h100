#!/usr/bin/env python3
"""E6 step 1 (CPU only): dominant decode kernels from existing graph-mode kernel-trace timelines.

Reads rocprofv3 kernel_trace.csv + marker_api_trace.csv of E5 ktrace runs (cubw production roots), keeps
dispatches whose start lies inside the 'decode' NVTX ranges, and aggregates per kernel name: dispatches
per decode tick, mean duration, share of summed kernel time, grid/workgroup and resource fields. With
two CU levels of the same cohort it also reports each kernel's duration ratio (restricted / 304), i.e.
which kernels carry the CU-restriction penalty. Kernel-trace timings carry the measured trace distortion
(E5 slowdown 1.05-1.09); ratios between levels share it.

  e6_kernels.py --trace n304=<attempt_dir> --trace n200=<attempt_dir> --out e6_kernels.json [--top 25]
"""
import argparse
import bisect
import collections
import csv
import json
from pathlib import Path


def short(name):
    s = name.split('(')[0]
    return s if len(s) <= 120 else s[:117] + '...'


def load(adir):
    d = Path(adir) / 'rocprof'
    mk = next(d.rglob('*marker_api_trace.csv'))
    kt = next(d.rglob('*kernel_trace.csv'))
    ranges, ticks = [], 0
    with mk.open() as f:
        for r in csv.DictReader(f):
            if r['Function'] == 'decode':
                ranges.append((int(r['Start_Timestamp']), int(r['End_Timestamp'])))
    ranges.sort()
    with mk.open() as f:
        for r in csv.DictReader(f):
            if r['Function'].startswith('decode_step_'):
                ticks += 1
    starts = [a for a, _ in ranges]
    agg = collections.defaultdict(lambda: {'n': 0, 'ns': 0})
    meta, total, ndisp = {}, 0, 0
    with kt.open() as f:
        for r in csv.DictReader(f):
            t0 = int(r['Start_Timestamp'])
            i = bisect.bisect_right(starts, t0) - 1
            if i < 0 or t0 > ranges[i][1]:
                continue
            dur = int(r['End_Timestamp']) - t0
            k = short(r['Kernel_Name'])
            agg[k]['n'] += 1; agg[k]['ns'] += dur
            total += dur; ndisp += 1
            meta.setdefault(k, {f: r[f] for f in ('Grid_Size_X', 'Grid_Size_Y', 'Grid_Size_Z', 'Workgroup_Size_X',
                                                  'VGPR_Count', 'Accum_VGPR_Count', 'SGPR_Count', 'LDS_Block_Size',
                                                  'Scratch_Size')})
    return {'decode_ranges': len(ranges), 'decode_step_markers': ticks, 'dispatches': ndisp, 'kernel_ns': total,
            'kernels': {k: dict(v, mean_us=v['ns'] / v['n'] / 1e3, share=v['ns'] / total, **meta[k]) for k, v in agg.items()}}


def main(a):
    runs = {}
    for t in a.trace:
        label, path = t.split('=', 1)
        runs[label] = load(path)
        runs[label]['attempt_dir'] = path
    base = next(iter(runs))
    top = sorted(runs[base]['kernels'].items(), key=lambda kv: -kv[1]['ns'])[:a.top]
    table = []
    for k, v in top:
        row = {'kernel': k, 'share': v['share'], 'mean_us': v['mean_us'],
               'per_tick': v['n'] / max(1, runs[base]['decode_step_markers']),
               'grid': [int(v['Grid_Size_X']), int(v['Grid_Size_Y']), int(v['Grid_Size_Z'])],
               'wg': int(v['Workgroup_Size_X']), 'vgpr': int(v['VGPR_Count']), 'agpr': int(v['Accum_VGPR_Count']),
               'lds': int(v['LDS_Block_Size'])}
        for label, r in runs.items():
            if label != base and k in r['kernels']:
                row[f'mean_us_{label}'] = r['kernels'][k]['mean_us']
                row[f'ratio_{label}_vs_{base}'] = r['kernels'][k]['mean_us'] / v['mean_us']
        table.append(row)
    summary = {label: {x: r[x] for x in ('decode_ranges', 'decode_step_markers', 'dispatches', 'kernel_ns', 'attempt_dir')}
               for label, r in runs.items()}
    out = {'base': base, 'runs': summary, 'top_kernels': table,
           'note': 'kernel-trace timing (trace distortion 1.05-1.09 measured in E5); dispatches inside decode ranges only'}
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for row in table:
        rs = ' '.join(f"{k.split('_vs_')[0][6:]}={v:.3f}" for k, v in row.items() if k.startswith('ratio_'))
        print(f"{row['share']:6.1%} {row['per_tick']:6.2f}/tick {row['mean_us']:8.2f} us grid={row['grid'][0]:>7} "
              f"wg={row['wg']:>4} vgpr={row['vgpr']:>3} lds={row['lds']:>6} {rs:>10}  {row['kernel'][:90]}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trace', action='append', required=True, help='label=attempt_dir (first is the base)')
    p.add_argument('--out', required=True)
    p.add_argument('--top', type=int, default=25)
    main(p.parse_args())
