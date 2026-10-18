#!/usr/bin/env python3
"""Build the common-contract deliverables for an AMD result root from the per-family summaries.

Writes, under <root>/analysis/:
  shoreline_case_manifest.json   every declared condition with disposition and acquisition counts
  shoreline_measurements.jsonl   one row per condition and quantity (value, unit, class, source, denominator)
  shoreline_service_calibration.json   independent service controls (M4/A-D, M7 under CU masks)
Latency bounds and c/r/f sensitivity come from latency_bounds.py (shoreline_latency_bounds.json).
Missing quantities are null with a reason, never zero.
"""
import argparse
import glob
import json
from pathlib import Path
import re


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def statuses(root, pattern):
    out = {}
    for d in sorted(glob.glob(str(root / pattern))):
        name = Path(d).name
        if not Path(d).is_dir():
            continue
        base = re.sub(r'_failed_\d+$', '', name)
        key = re.sub(r'_a\d+$', '', base)
        run = load(Path(d) / 'run.json')
        st = 'failed' if '_failed_' in name else (run or {}).get('status', 'missing_run_json')
        out.setdefault(key, []).append(st)
    return out


def main(args):
    root = args.root
    an = root / 'analysis'
    an.mkdir(exist_ok=True)
    manifest = {'result_root': str(root), 'families': {}}
    rows = []
    fam = {
        'M1_baselines': ('m1', '*_a[0-9]*'),
        'M1_representative_text': ('m1_text', '*_a[0-9]*'),
        'M6_cu_mask_decode': ('m6_cumask', '*_cu*_a[0-9]*'),
        'M8_energy': ('m8_energy', '*_a[0-9]*'),
        'M8_energy_under_cu_mask': ('m8_energy_cu', '*_a[0-9]*'),
        'M8_prefill_vs_cu': ('m8_prefill', '*_a[0-9]*'),
        'TP_extension': ('tp', 'tp*_a[0-9]*'),
    }
    for family, (sub, pat) in fam.items():
        st = statuses(root / sub, pat)
        manifest['families'][family] = {k: {'complete_acquisitions': v.count('complete'), 'attempts': len(v),
                                            'disposition': 'complete' if v.count('complete') >= 3 else
                                            ('partial' if v.count('complete') else 'failed')}
                                        for k, v in st.items()}
    m1 = load(root / 'm1/summary.json')
    for k, v in (m1 or {}).get('conditions', {}).items():
        if v.get('mean_ms_per_step'):
            rows.append({'family': 'M1', 'condition': k, 'quantity': 'decode_ms_per_step', 'value': v['mean_ms_per_step'],
                         'unit': 'ms', 'classification': 'measured-real-GPU', 'ci95': v.get('ci95_ms_per_step'),
                         'acquisitions': v['qualified_acquisitions'], 'clock_qualified_v1': v['clock_qualified'],
                         'denominator': 'HIP-event decode interval / 63 decode ticks; 10 trials x acquisitions'})
    cnt = load(root / 'm2_counters/summary.json')
    for k, v in (cnt or {}).get('cases', {}).items():
        for s in v.get('steps', []):
            for q, unit in (('read_bytes', 'bytes'), ('write_bytes', 'bytes'), ('D_read_over_U_tick', 'ratio'),
                            ('tcc_hit_fraction_of_requests', 'fraction')):
                rows.append({'family': 'M2', 'condition': k, 'tick': s['step'], 'quantity': q, 'value': s.get(q),
                             'unit': unit, 'classification': 'measured-real-GPU (profiled)',
                             'null_reason': None if q in s else 'counter group not completed (see manifest)',
                             'denominator': 'one decode step; L2->memory-side bytes (pre-Infinity-Cache), not HBM bytes'})
        manifest['families'].setdefault('M2_counters', {})[k] = {'groups_done': v['groups_done']}
    svc = load(root / 'm4_service/summary.json')
    if svc:
        (an / 'shoreline_service_calibration.json').write_text(json.dumps(
            {'classification': 'measured-real-GPU independent microbenchmarks', 'conditions': svc['conditions'],
             'failures': svc['failures'],
             'notes': ['useful bytes/flops are logical; counter acquisitions give L2->memory bytes',
                       'CU restriction: HSA_CU_MASK contiguous ranges (XCD-interleaved), multiples of 8',
                       'no held-out predictive model was fitted (too few distinct decoder families); see latency bounds']},
            indent=1) + '\n')
        for r in svc['conditions']:
            rows.append({'family': 'M4/M7', 'condition': r['condition'], 'cus': r['cus'], 'quantity': r['unit'],
                         'value': r['mean'], 'min': r['min'], 'max': r['max'], 'acquisitions': r['acquisitions'],
                         'classification': 'measured-real-GPU'})
    for fname, family in (('m5_pressure/summary.json', 'M5'), ('m8_energy/summary.json', 'M8')):
        s = load(root / fname)
        for k, v in (s or {}).get('cases', {}).items():
            q = 'pressure_over_normal_step_time' if family == 'M5' else 'joules_per_token'
            val = v.get('mean_ratio') if family == 'M5' else v.get('mean_j_per_token')
            rows.append({'family': family, 'condition': k, 'quantity': q, 'value': val,
                         'classification': 'measured-real-GPU', 'acquisitions': v.get('n') or v.get('acquisitions_complete')})
    with (an / 'shoreline_measurements.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    (an / 'shoreline_case_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f"rows={len(rows)} families={list(manifest['families'])}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args())
