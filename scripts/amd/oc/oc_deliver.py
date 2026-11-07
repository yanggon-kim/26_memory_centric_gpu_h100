#!/usr/bin/env python3
"""OC-v0 per-cohort delivery files (assignment §10): observability, overlap_bounds, service_references,
calibration_frozen, model_inputs and oc_status, written next to validation.json in the held-out folder.

Everything is copied from already-published OC artifacts (pilot oc_tables/att_activity, frozen file, held-out
validation and transfer) and from the cohort's G1-G8 goal_tables (OC5 reuse); nothing is refitted.

  oc_deliver.py --cohort mi300x --pilot <folder> --heldout <folder> --frozen docs/amd/oc/oc6_frozen_<c>.json
                --goal <goal folder>
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return json.loads(Path(p).read_text())


def dump(folder, name, obj):
    (folder / name).write_text(json.dumps(obj, indent=1) + '\n')


def main(a):
    pt, fz, val, tr, gt = (jl(a.pilot / 'oc_tables.json'), jl(a.frozen), jl(a.heldout / 'validation.json'),
                           jl(a.heldout / 'transfer_qwen7b.json'), jl(a.goal / 'goal_tables.json'))
    srcs = {'pilot/oc_tables.json': sha(a.pilot / 'oc_tables.json'), 'frozen': sha(a.frozen),
            'heldout/validation.json': sha(a.heldout / 'validation.json'),
            'heldout/transfer_qwen7b.json': sha(a.heldout / 'transfer_qwen7b.json'),
            'goal/goal_tables.json': sha(a.goal / 'goal_tables.json')}
    tool = {'rocprofv3': 'container 1.3.2 (vLLM routes, ATT projection), host 1.1.0 (native routes)', 'time_base': 'gfx clock (ATT)'}

    obs = {'cohort': a.cohort, 'tools': tool, 'metrics': {}}
    for k, v in pt['counter_validation']['tests'].items():
        obs['metrics'][k] = {'status': v.get('status'), 'status_isa': v.get('status_isa'), 'ratio': v.get('ratio'),
                             'ratio_isa': v.get('ratio_isa'), 'note': v.get('note'),
                             'scope': 'dispatch aggregate, device; replay per counter group (rocprofv3 --pmc)'}
    for k, v in pt['counter_validation']['activity'].items():
        if not k.startswith('_'):
            obs['metrics'][k] = {'status': 'unavailable as time', 'class': v.get('class'), 'scope': 'dispatch aggregate'}
    obs['metrics']['ATT_c_m_X'] = {'status': 'measured (checks ii, iv pass; iii degenerate)',
                                   'definition': 'c(t): any MFMA executing on the traced CU; m(t): any VMEM load outstanding '
                                                 '(issue to end of retiring s_waitcnt vmcnt); X = |c and m|',
                                   'units': 'gfx cycles', 'scope': 'one CU per SE, all SIMDs; ATT serialization as recorded'}
    obs['metrics']['device_HBM_activity'] = {'status': 'unavailable', 'reason': 'no counter validated as a time-domain or physical HBM observable'}
    obs['oc4_candidate_A'] = pt['oc4_candidate_A']['verdict']
    obs['oc4_candidate_B'] = {'verdict': pt['oc4_candidate_B']['verdict'], 'caveat': pt['oc4_candidate_B'].get('caveat')}
    obs['pc_sampling'] = {k: v.get('returncode') for k, v in pt['probe'].items() if k.startswith('pc_')}

    xs = [r['X_over_A_C'] for r in fz['att_per_shape'] if r.get('X_over_A_C') is not None]
    bounds = {'cohort': a.cohort, 'estimator': 'ATT candidate B (scripts/amd/oc/oc_att.py)',
              'per_shape': fz['att_per_shape'], 'pilot_routes': pt['oc4_candidate_B'].get('checks'),
              'X_over_A_C_range': [min(xs), max(xs)] if xs else None,
              'coupled_bounds': 'max(0, A_C + A_M - T) <= X <= min(A_C, A_M) per CU (in att_activity.json)',
              'validation': 'mfma4 A_M/T = 0, read A_C/T = 0, fused X > 0; split control degenerate',
              'eta_from_timings': {'status': 'not identifiable', 'eta_fit': fz['eta'], 'profile_95': fz['eta_profile_95'],
                                   'o_scale_tradeoff': fz['o_scale_tradeoff']},
              'note': 'physical per-CU activity intersection; not the fitted effective overlap eta'}

    refs = {'cohort': a.cohort, 'frozen_cells': fz['references'], 'frozen_acquisitions': fz['references_acq'],
            'pilot': {k: {x: v.get(x) for x in ('mode', 'mask', 'blocks', 'threads', 'footprint_bytes', 'median_ms', 'cv',
                                                  'useful_gb_per_s', 'useful_tflop_per_s')} for k, v in pt['references'].items()},
            'g6_service_goal': gt.get('g6_service'),
            'units': {'R_mfma4': 'issued 4x4x4 bf16 MFMA wave-instructions/s', 'B_ref': 'B/s (matched 1024-thread read, same mask and grid)',
                      'O_ref': 's (64x256 wvSplitK call)', 'useful_gb_per_s': 'GB/s useful bytes', 'value': 'per metric field'},
            'history': 'stream (cold weights per call); MALL-resident read reported separately in pilot'}

    shutil.copyfile(a.frozen, a.heldout / 'calibration_frozen.json')

    mi = {'cohort': a.cohort, 'source_sha256': srcs, 'family': fz['family'],
          'inputs': [
              {'name': 'issued_MFMA_wvSplitK', 'units': 'MFMA instructions per call', 'estimate': '4*m*ceil(k/1024)',
               'status': 'measured, validated on held-out shapes', 'check': val.get('issued_check'),
               'applicability': 'wvSplitK_hf_sml_<bf16,64,2,16,8,2,1>, N = 1; independent of grid and mask'},
              {'name': 'R_mfma4, B_ref, O_ref', 'units': refs['units'], 'estimate': fz['references'], 'status': 'measured (3 acquisitions)'},
              {'name': 'eta (fitted overlap)', 'status': 'not identifiable from timings', 'range': fz['eta_profile_95']},
              {'name': 'X/A_C per CU (ATT)', 'units': 'fraction', 'range': bounds['X_over_A_C_range'], 'status': 'measured, per-CU scope'},
              {'name': 'M6 kappa', 'estimate': fz['kappa_m6'], 'status': 'fitted on development; held-out not accepted',
               'heldout': val.get('m6')},
              {'name': 'M1 eta', 'estimate': fz['eta'], 'status': 'fitted on development; held-out not accepted',
               'heldout': {k: val.get(k) for k in ('n_pass', 'n_cells', 'median_signed_err', 'accepted')}},
              {'name': 'transfer (Qwen2.5-7B graph decode)', 'per_tick_projection_rel_err': tr['per_tick_projection_rel_err'],
               'status': 'secondary check, not an acceptance criterion'}],
          'unknowns': ['device HBM activity in time', 'physical overlap at device scope', 'eta as an architectural constant',
                       'arithmetic/SRAM/NoC/MC/PHY areas (not derivable from utilization)']}

    sel = [c for c in gt['g1']['cells']]
    st = {'cohort': a.cohort, 'source_sha256': srcs, 'goals': {
        'OC0': {'status': 'complete', 'evidence': 'pilot probe', 'result': 'PC sampling and ATT decode on both tool routes'},
        'OC1': {'status': 'complete (reused G-campaign Qwen32B graph and eager traces)', 'evidence': 'pilot oc1_*'},
        'OC2': {'status': 'complete', 'evidence': 'service_references.json'},
        'OC3': {'status': 'complete', 'evidence': 'pilot/dev/held-out references and kernel controls'},
        'OC4': {'status': 'complete: candidate A unavailable, candidate B bounded (per-CU)', 'evidence': 'overlap_bounds.json',
                'missing': 'device-scope activity; non-degenerate split control'},
        'OC5': {'status': 'complete (mechanism table from existing contrasts, no new GPU steps)',
                'evidence': 'docs/amd/overlap_calibration_results_20261107.md section OC5', 'g1_cells': len(sel)},
        'OC6': {'status': 'complete: neither frozen model accepted on held-out', 'evidence': 'calibration_frozen.json, validation.json'},
        'OC7': {'status': 'complete (reused G2/G3 scopes; no new transfer cases justified)', 'evidence': 'model_inputs.json, transfer_qwen7b.json'}},
        'next_action': 'none queued; a new memory reference (wvSplitK-shaped stream) would be needed before refitting OC6'}

    for n, o in (('observability.json', obs), ('overlap_bounds.json', bounds), ('service_references.json', refs),
                 ('model_inputs.json', mi), ('oc_status.json', st)):
        dump(a.heldout, n, o)
    print(a.cohort, 'written to', a.heldout)


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('pilot', 'heldout', 'frozen', 'goal'):
        q.add_argument(f'--{k}', type=Path, required=True)
    q.add_argument('--cohort', required=True)
    main(q.parse_args())
