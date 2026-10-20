#!/usr/bin/env python3
"""Per-kernel bottleneck classification of decode steps from existing rocprofv3 counter runs.

For each case under m2_counters/, the FETCH_SIZE, WRITE_SIZE and MFMA counter runs (separate processes of
the same program) are aligned per decode step by kernel position, with kernel names checked to match.
Per kernel (decode ticks 4-6):
  bytes  = FETCH_SIZE + WRITE_SIZE (L2 <-> memory side; upper bound on HBM bytes, Infinity Cache invisible)
  flops  = SQ_INSTS_VALU_MFMA_MOPS_BF16 x 512 (issued MFMA work; unit per the rocprof-compute convention)
  t      = kernel duration from the FETCH run's kernel trace (profiled: counter collection serializes
           dispatches, so durations and especially gaps are perturbed; graph-mode durations come from
           kernels replayed inside the profiled graph)
  ideal  = max(bytes / BETA, flops / P) using the measured service rates (4.55 TB/s, 1071 TFLOP/s)
  excess = t - ideal (time a kernel spends beyond what bandwidth or MFMA peak would need)
Labels (declared before inspection, with threshold sensitivity reported):
  memory-bound   achieved bytes/t >= THR x BETA
  compute-bound  achieved flops/t >= THR x P
  overhead       t < 10 us and not memory/compute-bound (launch/latency-dominated tiny kernels)
  latency-bound  everything else: substantial time at low bandwidth and low MFMA rate
  uncounted      kernels excluded from counter collection (runtime Triton kernels); time only
Per step: busy time by label, ideal vs excess, plus GPU idle gaps inside the step and the host span
outside GPU work. This splits the residual O of the C/M/X/O bounds into below-peak memory kernels,
other kernels and idle gaps. Classification is analytical over measured, profiled inputs.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_trace import classify as op_class

BETA, P = 4.5485e12, 1.0711e15


def load_run(rocprof_dir):
    d = Path(rocprof_dir)
    kt = next(d.rglob('*kernel_trace.csv'))
    mk = next(d.rglob('*marker_api_trace.csv'))
    cc = next(d.rglob('*counter_collection.csv'), None)
    steps = {int(r['Function'].rsplit('_', 1)[1]): (int(r['Start_Timestamp']), int(r['End_Timestamp']))
             for r in csv.DictReader(open(mk)) if re.fullmatch(r'decode_step_\d+', r['Function'])}
    counters = {}
    if cc:
        for r in csv.DictReader(open(cc)):
            counters.setdefault(int(r['Dispatch_Id']), {})[r['Counter_Name']] = float(r['Counter_Value'])
    kernels = sorted(({'name': r['Kernel_Name'], 's': int(r['Start_Timestamp']), 'e': int(r['End_Timestamp']),
                       'dispatch': int(r['Dispatch_Id'])} for r in csv.DictReader(open(kt))), key=lambda k: k['s'])
    out = {}
    for k, (a, b) in steps.items():
        ks = [x for x in kernels if a <= x['s'] <= b]
        for x in ks:
            x['counters'] = counters.get(x['dispatch'], {})
        out[k] = {'window': (a, b), 'kernels': ks}
    return out


def label(t, nbytes, flops, counted, thr):
    if not counted:
        return 'uncounted'
    if nbytes / t >= thr * BETA:
        return 'memory-bound'
    if flops / t >= thr * P:
        return 'compute-bound'
    if t < 10e-6:
        return 'overhead'
    return 'latency-bound'


def union_ns(ks):
    total, end = 0, None
    for x in sorted(ks, key=lambda k: k['s']):
        if end is None or x['s'] > end:
            total += x['e'] - x['s']; end = x['e']
        elif x['e'] > end:
            total += x['e'] - end; end = x['e']
    return total


def analyse_case(case_dir, steps, thr):
    groups = {g.name: g for g in case_dir.iterdir() if g.is_dir() and (g / 'done').exists()}
    need = ['FETCH_SIZE', 'WRITE_SIZE', 'SQ_INSTS_VALU_MFMA_MOPS_BF16+SQ_WAVES+GRBM_GUI_ACTIVE']
    if not all(n in groups for n in need):
        return {'status': 'missing counter groups', 'groups': sorted(groups)}
    runs = {n: load_run(groups[n] / 'rocprof') for n in need}
    base = runs['FETCH_SIZE']
    result = {'status': 'ok', 'threshold': thr, 'steps': [], 'alignment_mismatches': 0}
    per_kernel = {}
    for k in steps:
        if not all(k in r for r in runs.values()):
            continue
        ks = base[k]['kernels']
        others = {n: runs[n][k]['kernels'] for n in need[1:]}
        if any(len(o) != len(ks) for o in others.values()):
            result['alignment_mismatches'] += 1
            continue
        if any(o[i]['name'] != ks[i]['name'] for o in others.values() for i in range(len(ks))):
            result['alignment_mismatches'] += 1
            continue
        agg = {}
        for i, x in enumerate(ks):
            t = (x['e'] - x['s']) * 1e-9
            fkb = x['counters'].get('FETCH_SIZE')
            wkb = others['WRITE_SIZE'][i]['counters'].get('WRITE_SIZE')
            mops = others[need[2]][i]['counters'].get('SQ_INSTS_VALU_MFMA_MOPS_BF16')
            counted = fkb is not None and wkb is not None and mops is not None
            nbytes = ((fkb or 0) + (wkb or 0)) * 1024
            flops = (mops or 0) * 512
            lab = label(t, nbytes, flops, counted, thr) if t > 0 else 'overhead'
            ideal = max(nbytes / BETA, flops / P) if counted else 0.0
            cell = agg.setdefault(lab, {'kernels': 0, 'time_s': 0.0, 'ideal_s': 0.0, 'bytes': 0.0, 'flops': 0.0})
            cell['kernels'] += 1; cell['time_s'] += t; cell['ideal_s'] += min(ideal, t); cell['bytes'] += nbytes; cell['flops'] += flops
            key = (op_class(x['name']), lab)
            pk = per_kernel.setdefault(key, {'kernels': 0, 'time_s': 0.0, 'bytes': 0.0, 'flops': 0.0})
            pk['kernels'] += 1; pk['time_s'] += t; pk['bytes'] += nbytes; pk['flops'] += flops
        a, b = base[k]['window']
        busy = union_ns(ks) * 1e-9
        span = (ks[-1]['e'] - ks[0]['s']) * 1e-9 if ks else 0.0
        result['steps'].append({'step': k, 'labels': agg, 'busy_union_s': busy, 'gpu_span_s': span,
                                'idle_gaps_in_span_s': span - busy, 'marker_wall_s': (b - a) * 1e-9,
                                'kernel_sum_s': sum(v['time_s'] for v in agg.values())})
    n = len(result['steps'])
    if not n:
        result['status'] = 'no aligned steps'
        return result
    labels = sorted({l for s in result['steps'] for l in s['labels']})
    mean = lambda f: sum(f(s) for s in result['steps']) / n
    result['mean_step'] = {
        'kernel_sum_s': mean(lambda s: s['kernel_sum_s']), 'busy_union_s': mean(lambda s: s['busy_union_s']),
        'idle_gaps_in_span_s': mean(lambda s: s['idle_gaps_in_span_s']), 'marker_wall_s': mean(lambda s: s['marker_wall_s']),
        'by_label': {l: {q: mean(lambda s: s['labels'].get(l, {}).get(q, 0.0)) for q in ('kernels', 'time_s', 'ideal_s', 'bytes', 'flops')}
                     for l in labels}}
    for l, v in result['mean_step']['by_label'].items():
        v['excess_s'] = v['time_s'] - v['ideal_s']
        v['achieved_tb_per_s'] = v['bytes'] / v['time_s'] / 1e12 if v['time_s'] else None
    result['by_op_and_label'] = [{'op': o, 'label': l, 'kernels_per_step': v['kernels'] / n, 'time_s_per_step': v['time_s'] / n,
                                  'achieved_tb_per_s': v['bytes'] / v['time_s'] / 1e12 if v['time_s'] else None,
                                  'achieved_tflop_per_s': v['flops'] / v['time_s'] / 1e12 if v['time_s'] else None}
                                 for (o, l), v in sorted(per_kernel.items(), key=lambda kv: -kv[1]['time_s'])]
    return result


def main(args):
    out = {'classification': 'analytical classification over measured (profiled) per-kernel inputs',
           'scope': __doc__.strip(), 'beta_bytes_per_s': BETA, 'p_flops_per_s': P, 'cases': {}}
    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        out['cases'][case_dir.name] = {f'thr{int(t * 100)}': analyse_case(case_dir, args.steps, t) for t in (0.5, 0.6, 0.7)}
    # Decomposition of the clean (unprofiled) step time T from M1: kernel durations are taken from the
    # profiled runs, idle/launch time = T - busy union (profiled gaps are inflated by serialization).
    m1 = json.loads((args.root.parent / 'm1/summary.json').read_text())['conditions'] if (args.root.parent / 'm1/summary.json').exists() else {}
    out['decomposition'] = {}
    for name, by_thr in out['cases'].items():
        r = by_thr['thr60']
        T = m1.get(name, {}).get('mean_ms_per_step')
        if r.get('status') != 'ok' or not T:
            continue
        T *= 1e-3
        L = r['mean_step']['by_label']
        get = lambda l, q: L.get(l, {}).get(q, 0.0)
        row = {'T_clean_s': T, 'busy_union_profiled_s': r['mean_step']['busy_union_s'],
               'memory_kernels_ideal_s': get('memory-bound', 'ideal_s'),
               'memory_kernels_excess_s': get('memory-bound', 'excess_s'),
               'latency_bound_kernels_s': get('latency-bound', 'time_s'),
               'compute_bound_kernels_s': get('compute-bound', 'time_s'),
               'small_overhead_kernels_s': get('overhead', 'time_s'),
               'uncounted_kernels_s': get('uncounted', 'time_s'),
               'other_kernels_ideal_s': sum(get(l, 'ideal_s') for l in ('latency-bound', 'overhead', 'compute-bound')),
               'idle_launch_s': T - r['mean_step']['busy_union_s']}
        row['shares_of_T'] = {k[:-2]: v / T for k, v in row.items() if k.endswith('_s') and k not in ('T_clean_s', 'busy_union_profiled_s')}
        out['decomposition'][name] = row
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    print('--- decomposition of clean T (ms): ideal-memory | memory excess | latency-bound | small kernels | uncounted | idle/launch')
    for name, d in out['decomposition'].items():
        print(f"{name:26s} T={d['T_clean_s']*1e3:6.3f} | {d['memory_kernels_ideal_s']*1e3:6.3f} | {d['memory_kernels_excess_s']*1e3:5.3f} | "
              f"{d['latency_bound_kernels_s']*1e3:5.3f} | {d['small_overhead_kernels_s']*1e3:5.3f} | {d['uncounted_kernels_s']*1e3:5.3f} | {d['idle_launch_s']*1e3:6.3f}")
    for name, by_thr in out['cases'].items():
        r = by_thr['thr60']
        if r.get('status') != 'ok':
            print(f'{name:26s} {r.get("status")} {r.get("groups", "")}'); continue
        m = r['mean_step']
        parts = '  '.join(f"{l}:{v['time_s'] * 1e3:.3f}ms(ideal {v['ideal_s'] * 1e3:.3f}, {v['achieved_tb_per_s'] or 0:.2f}TB/s)"
                          for l, v in sorted(m['by_label'].items(), key=lambda kv: -kv[1]['time_s']))
        print(f"{name:26s} busy={m['busy_union_s'] * 1e3:.3f}ms gaps={m['idle_gaps_in_span_s'] * 1e3:.3f}ms "
              f"mism={r['alignment_mismatches']}  {parts}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='results/.../m2_counters')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[4, 5, 6])
    main(p.parse_args())
