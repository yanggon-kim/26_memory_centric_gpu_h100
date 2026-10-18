#!/usr/bin/env python3
"""Attribute rocprofv3 kernel dispatches to vLLM decode steps and operation classes (M1/M3).

Eager vLLM synchronizes at the end of every engine step, so a kernel whose GPU start
lies inside a decode_step_N marker belongs to that step (host and GPU timestamps share
one clock in rocprofv3). Kernel time sums can exceed wall time only with concurrency;
busy-union and wall shares use separate denominators. Profiled timing is not baseline timing.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

CLASSES = [  # first match wins; kept explicit so misclassification is reviewable
    ('attention_forward', r'paged_attention|_fwd_kernel|flash|fmha|attn_fwd|unified_attention|mla'),
    ('attention_reduce', r'reduce_kernel|combine'),
    ('kv_cache_update', r'reshape_and_cache|cache_kernel|concat_and_cache'),
    ('linear_gemv_gemm', r'wvSplitK|Cijk_|gemm|gemv|matmul|hipblaslt|skinny|wvSpltK|LLGemm'),
    ('norm', r'rms_norm|layernorm|layer_norm'),
    ('activation', r'act_and_mul|silu|gelu'),
    ('rotary', r'rotary'),
    ('sampling_logits', r'argmax|topk|softmax|sampl|log_softmax|gather|scatter|index'),
    ('copy_fill', r'copyBuffer|fillBuffer|FillFunctor|elementwise_kernel|copy_kernel|CatArrayBatched'),
]


def classify(name):
    for label, pattern in CLASSES:
        if re.search(pattern, name):
            return label
    return 'other'


def load(directory):
    d = Path(directory)
    kernels = [dict(r) for r in csv.DictReader(open(next(d.rglob('*kernel_trace.csv'))))]
    markers = [dict(r) for r in csv.DictReader(open(next(d.rglob('*marker_api_trace.csv'))))]
    for k in kernels:
        k['s'], k['e'] = int(k['Start_Timestamp']), int(k['End_Timestamp'])
    for m in markers:
        m['s'], m['e'] = int(m['Start_Timestamp']), int(m['End_Timestamp'])
    kernels.sort(key=lambda k: k['s'])
    return kernels, markers


def union(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def main(args):
    kernels, markers = load(args.trace)
    steps = sorted([m for m in markers if re.fullmatch(r'decode_step_\d+', m['Function'])],
                   key=lambda m: int(m['Function'].rsplit('_', 1)[1]))
    layers = [m for m in markers if re.fullmatch(r'layer_\d+', m['Function'])]
    out = {'classification': 'measured-real-GPU (profiled; timings perturbed)', 'trace': str(args.trace),
           'steps': [], 'kernel_inventory': {}, 'class_patterns': CLASSES}
    inventory = {}
    for m in steps:
        ks = [k for k in kernels if m['s'] <= k['s'] <= m['e']]
        step = {'step': int(m['Function'].rsplit('_', 1)[1]), 'wall_ns': m['e'] - m['s'], 'launches': len(ks),
                'kernel_sum_ns': sum(k['e'] - k['s'] for k in ks), 'busy_union_ns': union([(k['s'], k['e']) for k in ks])}
        gaps = [b['s'] - a['e'] for a, b in zip(ks, ks[1:])]
        step['gap_ns'] = {'median': statistics.median(gaps) if gaps else None, 'sum_positive': sum(g for g in gaps if g > 0)}
        if ks:
            step['gpu_span_ns'] = ks[-1]['e'] - ks[0]['s']
        by_class = {}
        for k in ks:
            c = classify(k['Kernel_Name'])
            by_class.setdefault(c, {'ns': 0, 'launches': 0})
            by_class[c]['ns'] += k['e'] - k['s']
            by_class[c]['launches'] += 1
            if step['step'] in args.detail_steps:
                key = k['Kernel_Name'][:200]
                row = inventory.setdefault(key, {'class': c, 'calls': 0, 'ns': 0, 'vgpr': k['VGPR_Count'],
                                                 'agpr': k['Accum_VGPR_Count'], 'sgpr': k['SGPR_Count'],
                                                 'lds_bytes': k['LDS_Block_Size'], 'scratch': k['Scratch_Size'],
                                                 'workgroup': [k['Workgroup_Size_X'], k['Workgroup_Size_Y'], k['Workgroup_Size_Z']],
                                                 'grids': set()})
                row['calls'] += 1
                row['ns'] += k['e'] - k['s']
                row['grids'].add(f"{k['Grid_Size_X']}x{k['Grid_Size_Y']}x{k['Grid_Size_Z']}")
        step['by_class'] = by_class
        step_layers = [l for l in layers if m['s'] <= l['s'] <= m['e']]
        if step['step'] in args.detail_steps and step_layers:
            per = []
            for l in sorted(step_layers, key=lambda l: l['s']):
                lk = [k for k in ks if l['s'] <= k['s'] <= l['e']]
                per.append({'layer': int(l['Function'].split('_')[1]), 'host_ns': l['e'] - l['s'], 'launches': len(lk),
                            'kernel_sum_ns': sum(k['e'] - k['s'] for k in lk)})
            step['layers'] = per
        out['steps'].append(step)
    for row in inventory.values():
        row['grids'] = sorted(row['grids'])
    out['kernel_inventory'] = dict(sorted(inventory.items(), key=lambda kv: -kv[1]['ns']))
    out['detail_steps'] = args.detail_steps
    sel = [s for s in out['steps'] if s['step'] in args.detail_steps]
    if sel:
        tot = sum(s['kernel_sum_ns'] for s in sel)
        classes = {}
        for s in sel:
            for c, v in s['by_class'].items():
                classes[c] = classes.get(c, 0) + v['ns']
        out['detail_summary'] = {'mean_wall_ns': statistics.mean(s['wall_ns'] for s in sel),
                                 'mean_busy_union_ns': statistics.mean(s['busy_union_ns'] for s in sel),
                                 'mean_launches': statistics.mean(s['launches'] for s in sel),
                                 'kernel_time_share_by_class': {c: v / tot for c, v in sorted(classes.items(), key=lambda kv: -kv[1])},
                                 'busy_union_over_wall': statistics.mean(s['busy_union_ns'] / s['wall_ns'] for s in sel)}
    args.output.write_text(json.dumps(out, indent=1, default=list) + '\n')
    print(json.dumps(out.get('detail_summary'), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--detail-steps', type=int, nargs='+', default=[4, 5, 6])
    main(p.parse_args())
