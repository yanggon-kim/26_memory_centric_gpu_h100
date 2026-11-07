#!/usr/bin/env python3
"""model_inputs.json for the area-allocation model V0 (docs/area_allocation_mathematical_model_v0_20261105.md §10-11).

Reads one published goal folder (goal_tables.json, gates.json) and writes the G1-G8 observables with units, scope,
estimate/CI, measured-vs-fitted status and source hashes, plus explicit unknowns. Values are copied, never refitted.

  model_inputs.py --folder results_published/<goal folder> [--g8-judge docs/amd/goal/g8_judge_final.json]
                  [--oc-frozen docs/amd/oc/oc6_frozen_<cohort>.json]
"""
import argparse
import hashlib
import json
from pathlib import Path

ARCH = {'mi300x': ('gfx942', 304), 'mi325x': ('gfx942', 304), 'mi350x': ('gfx950', 256)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ratio(r):
    return None if not r else {'estimate': r['ratio'], 'ci95': r.get('ci95'), 'blocks': r.get('units', r.get('blocks'))}


def main(a):
    t = json.loads((a.folder / 'goal_tables.json').read_text())
    coh = t['cohort']
    arch, cus = ARCH[coh]
    src = {'goal_tables.json': sha(a.folder / 'goal_tables.json'), 'gates.json': sha(a.folder / 'gates.json')}
    clk = 'observational_clock_confounded (controlled clocks: unavailable, not authorized)'
    rows = []

    def add(name, units, est, scope, status='measured', **kw):
        rows.append({'name': name, 'units': units, 'estimate': est, 'scope': scope, 'status': status, **kw})

    wl = 'Qwen2.5-32B BF16 vLLM graph decode, batch 1, context 4096'
    for c in t['g1']['cells']:
        add(f'G1.R_restrict[mask={c["mask"]},grid={c["grid"]}]', 'ratio (decode step time / stock full device)',
            {'estimate': c['ratio'], 'ci95': c['ci95'], 'blocks': c['blocks']},
            {'workload': wl, 'cus_active': c['mask'], 'wvsplitk_grid': c['grid'], 'clock': c['clock']},
            qualification=c['disposition'], outputs_match_stock=c['outputs_match_stock'],
            applicability='CU mask restricts issue resources; it does not identify mm2 per CU')
    for case, d in t['g2_isolated'].items():
        for k, v in d.items():
            if v.get('ratio_vs_A'):
                add(f'G2.isolated[{case},{k}]', f'ratio ({v["metric"]} vs A = full, stock)', ratio(v['ratio_vs_A']),
                    {'workload': case, 'condition': k, 'clock': v.get('clock')}, qualification=clk)
    for tab, d in t['serving'].items():
        for k, v in d.items():
            for m in ('tok_s', 'ttft_p99', 'itl_p99', 'e2e_p99', 'j_per_tok'):
                r = (v.get(m) or {}).get('ratio_vs_A') if isinstance(v.get(m), dict) else None
                if r:
                    add(f'G2.serving[{tab},{k},{m}]', 'ratio vs A', ratio(r),
                        {'workload': 'mixed prefill+decode serving, Qwen2.5-32B', 'table': tab, 'condition': k}, qualification=clk)
    for case, d in t['g3_energy'].items():
        for k, v in d.items():
            add(f'G3.decode_energy[{case},{k}]', 'J/token (board accumulator, decode window)',
                {'estimate': v['joules_per_token_mean'], 'avg_power_w': v['avg_power_w_mean'],
                 'ratio_vs_A': ratio(v.get('jpt_ratio_vs_A'))}, {'workload': case, 'condition': k, 'scope': v.get('scope')},
                qualification=clk, applicability='board energy; no component (CU/SRAM/HBM/PHY) split')
    for n, d in t['g5'].items():
        for k, v in d.items():
            if isinstance(v, dict) and v.get('ratio_vs_absent'):
                add(f'G5.decoder_slowdown[n={n},{k}]', 'ratio (decoder time with competitor / absent)', ratio(v['ratio_vs_absent']),
                    {'cus_active': int(n), 'competitor': k, 'competitor_gb_s': v.get('competitor_gb_s'), 'clock': v.get('clock')},
                    qualification=clk)
    for k, v in t['g6_service'].items():
        if isinstance(v, dict) and 'value' in v:
            add(f'G6.service[{k}]', v['metric'], v['value'], {'cus_active': v.get('level'), 'kernel': 'native control'},
                qualification=clk)
    g7 = t.get('g7_logical') or {}
    if g7:
        add('G7.logical_demand', 'bytes/token', {k: v for k, v in g7.items() if k != 'scope'}, {'scope': g7.get('scope')},
            status='derived (model shapes, BF16)')
    for k, v in (t.get('g7_counters') or {}).items():
        if v.get('per_tick') is not None:
            add(f'G7.counter[{k}]', f'{v["counter"]} per decode tick (unit per G4 audit)', v['per_tick'],
                {'cus_active': v['n'], 'grid': v['grid'], 'mode': 'eager'}, qualification=t['g4_counters'].get('classification', {}).get(v['counter']))
    g8 = None
    if a.g8_judge:
        j = json.loads(a.g8_judge.read_text())
        g8 = {'verdict': j['verdict'], 'mae': j['mae'], 'max_abs_error': j['max_abs_error'], 'source_sha256': sha(a.g8_judge),
              'rows': [r for r in j['rows'] if r['cohort'] == coh]}
    oc = None
    if a.oc_frozen:
        oc = {'frozen_file_sha256': sha(a.oc_frozen), 'verdict': 'not accepted on held-out (see overlap_calibration_results_20261105.md)',
              'issued_mfma_wvsplitk': '4*m*ceil(k/1024) per call (exact, validated on held-out shapes)',
              'eta': 'not identifiable from kernel timings; ATT per-CU X/A_C 0.94-1.00'}
    out = {'cohort': coh, 'arch': arch, 'cus_full': cus, 'source_sha256': src,
           'gates': {k: t['gates'].get(k) for k in ('placement_ok', 'lifecycle_ok', 'outputs_ok', 'leak_ok_v4', 'leak_ok_v2')},
           'inputs': rows, 'g8_frozen_model': g8, 'oc6': oc,
           'unknowns': ['mm2 per CU / per MFMA engine (CU masks do not identify area)',
                        'SRAM, memory-controller and PHY areas (not measured; none invented)',
                        'physical HBM read traffic: ' + ('no counter validated as physical HBM by G4'
                                                         if not t['g4_counters'].get('physical_hbm_read_counter') else 'see G4'),
                        'component energy split (only board energy measured)',
                        'clock-controlled restriction curves (clock changes not authorized)',
                        'G5 decoder x competitor on this cohort' if not t['gates'].get('leak_ok_v4') else None]}
    out['unknowns'] = [u for u in out['unknowns'] if u]
    (a.folder / 'model_inputs.json').write_text(json.dumps(out, indent=1) + '\n')
    print(coh, len(rows), 'inputs')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--folder', type=Path, required=True)
    q.add_argument('--g8-judge', type=Path)
    q.add_argument('--oc-frozen', type=Path)
    main(q.parse_args())
