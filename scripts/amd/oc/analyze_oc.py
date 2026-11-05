#!/usr/bin/env python3
"""OC-v0 reader (CPU only): applies the pre-declared rules of docs/amd/overlap_calibration_execution_plan_20261105.md
to one OC run root and writes <root>/analysis_oc/{oc_tables.json, observability.json, service_references.json,
overlap_bounds.json, oc_status.json, oc_summary.md}. Raw files are never modified.

  analyze_oc.py --root <run root>
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics as st

WAVE = 64
MFMA4_FLOPS = 2048  # 2*4*4*4*16 per wave-level v_mfma_f32_4x4x4_16b_bf16


def jl(p):
    lines = [x for x in Path(p).read_text().splitlines() if x.strip().startswith('{')]
    return json.loads(lines[-1]) if lines else None


def final_attempts(root):
    out = {}
    for f in sorted((root / 'journal').glob('*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get('step'):
                out[r['step']] = r  # last attempt wins
    return out


def family(name):
    n = name.lower()
    if 'wvsplitk' in n or 'llmm1' in n:
        return 'projection_skinny'
    if 'cijk' in n or 'gemm' in n or 'hipblaslt' in n:
        return 'projection_gemm'
    if 'attention' in n or 'paged' in n or 'attn' in n or 'fmha' in n:
        return 'attention'
    if 'rms' in n or 'norm' in n:
        return 'norm'
    if 'rotary' in n or 'silu' in n or 'act_and_mul' in n or 'elementwise' in n or 'copy' in n or 'cache' in n:
        return 'elementwise'
    if 'reduce' in n or 'softmax' in n or 'argmax' in n or 'sample' in n:
        return 'reduction'
    return 'other'


# ---------------------------------------------------------------- counter validation (section 2)
def validation(root, plan):
    files = sorted(root.glob('oc0_counters/**/oc_counters.json'))
    if not files:
        return {'status': 'missing'}
    d = json.loads(files[0].read_text())
    cus = d['cus']
    it = 4096
    n4 = (1 << 30) // 16
    runs = {(r['load'], r['group']): r for r in d['runs']}

    def val(load, group, ctr, kprefix='k_'):
        r = runs.get((load, group))
        if not r or r.get('returncode') != 0:
            return None
        ks = [v for k, v in r.get('kernels', {}).items() if k.startswith(kprefix)]
        if not ks:
            return None
        tot = sum(v['counters'].get(ctr, 0) for v in ks)
        disp = sum(v['dispatches'] for v in ks)
        return tot / disp if disp else None

    out = {'groups_ok': {f'{l}:{g}': r.get('returncode') for (l, g), r in runs.items()}, 'tests': {}}
    t = out['tests']
    waves1, waves4 = cus * 256 // WAVE, cus * 1024 // WAVE
    x = val('read', 'inst', 'SQ_INSTS_VMEM_RD') or val('read', 'vmem_level', 'SQ_INSTS_VMEM_RD')
    exp = n4 / WAVE
    t['SQ_INSTS_VMEM_RD'] = {'measured': x, 'expected': exp, 'ratio': x / exp if x else None,
                             'status': 'validated' if x and abs(x / exp - 1) <= 0.02 else ('invalid' if x else 'unavailable')}
    for w, waves in (('w1', waves1), ('w4', waves4)):
        x = val(f'mfma4_{w}', 'mfma', 'SQ_INSTS_MFMA')
        exp = 8 * it * waves
        t[f'SQ_INSTS_MFMA_{w}'] = {'measured': x, 'expected': exp, 'ratio': x / exp if x else None,
                                   'status': 'validated' if x and abs(x / exp - 1) <= 0.02 else ('invalid' if x else 'unavailable')}
        m = val(f'mfma4_{w}', 'mfma', 'SQ_INSTS_VALU_MFMA_MOPS_BF16')
        t[f'SQ_INSTS_VALU_MFMA_MOPS_BF16_{w}'] = {'measured': m, 'per_mfma4_instruction': m / (8 * it * waves) if m else None,
                                                  'status': 'measured-unvalidated' if m else 'unavailable'}
        x = val(f'fma_{w}', 'inst', 'SQ_INSTS_VALU')
        exp = 8 * it * waves
        t[f'SQ_INSTS_VALU_{w}'] = {'measured': x, 'expected_min': exp, 'ratio': x / exp if x else None,
                                   'status': 'validated' if x and 1.0 <= x / exp <= 1.25 else ('invalid' if x else 'unavailable')}
    for ctr in ('FETCH_SIZE',):
        x = val('read', 'fetch', ctr)
        t[ctr] = {'measured': x, 'useful_bytes': 1 << 30, 'ratio_KiB': x * 1024 / (1 << 30) if x else None}
        t[ctr]['status'] = 'validated' if x and abs(t[ctr]['ratio_KiB'] - 1) <= 0.05 else ('invalid' if x else 'unavailable')
    for ctr in ('TCC_EA0_RDREQ_32B', 'TCC_EA0_RDREQ_64B', 'TCC_EA0_RDREQ_128B', 'TCC_EA0_RDREQ_DRAM'):
        x = val('read', 'req_size', ctr)
        t[ctr] = {'measured': x, 'ratio_req': {u: x * u / (1 << 30) for u in (32, 64, 128)} if x else None}
        ok = x and any(abs(v - 1) <= 0.05 for v in t[ctr]['ratio_req'].values())
        t[ctr]['status'] = 'validated' if ok else ('measured-unvalidated' if x else 'unavailable')
    x = val('read', 'ea_level', 'TCC_EA0_RDREQ')
    t['TCC_EA0_RDREQ'] = {'measured': x, 'ratio_req': {u: x * u / (1 << 30) for u in (32, 64, 128)} if x else None}
    t['TCC_EA0_RDREQ']['status'] = 'validated' if x and any(abs(v - 1) <= 0.05 for v in t['TCC_EA0_RDREQ']['ratio_req'].values()) \
        else ('measured-unvalidated' if x else 'unavailable')
    # in-flight levels: Little's law latency (cycles) = level / requests
    lv = {}
    for load in ('ilp1', 'ilp8', 'read'):
        a, b = val(load, 'vmem_level', 'SQ_INST_LEVEL_VMEM'), val(load, 'vmem_level', 'SQ_INSTS_VMEM_RD')
        c, e, g = val(load, 'ea_level', 'TCC_EA0_RDREQ_LEVEL'), val(load, 'ea_level', 'TCC_EA0_RDREQ'), val(load, 'ea_level', 'GRBM_GUI_ACTIVE')
        lv[load] = {'vmem_latency_cycles': a / b if a and b else None, 'vmem_mean_in_flight_per_gui_cycle': a / g if a and g else None,
                    'ea_latency_cycles': c / e if c and e else None, 'ea_mean_in_flight_per_gui_cycle': c / g if c and g else None}
    rises = all(lv[k][f] is not None for k in ('ilp1', 'ilp8') for f in ('ea_mean_in_flight_per_gui_cycle',)) and \
        lv['ilp8']['ea_mean_in_flight_per_gui_cycle'] > lv['ilp1']['ea_mean_in_flight_per_gui_cycle']
    lat = lv['read']['ea_latency_cycles']
    t['TCC_EA0_RDREQ_LEVEL'] = {'by_load': lv, 'in_flight_rises_with_ilp': rises,
                                'latency_cycles_read': lat,
                                'status': 'validated' if rises and lat and 1000 <= lat <= 10000 else ('measured-unvalidated' if lat else 'unavailable')}
    # activity semantics: busy fraction (counter / SQ_BUSY_CYCLES of the same pass) at 4 vs 1 waves per SIMD
    act = {}
    for ctr, group, loads in (('SQ_ACTIVE_INST_VALU', 'active', ('fma', 'mfma4')), ('SQ_ACTIVE_INST_VMEM', 'active', None),
                              ('SQ_ACTIVE_INST_ANY', 'active2', ('fma', 'mfma4'))):
        fr = {}
        for load in ('fma_w1', 'fma_w4', 'mfma4_w1', 'mfma4_w4', 'read', 'ilp1', 'ilp8', 'split', 'fused'):
            a = val(load, group, ctr)
            # plan section 7: T = SQ_BUSY_CU_CYCLES (per-CU domain); SQ_BUSY_CYCLES recorded as an alternative
            den = val(load, group, 'SQ_BUSY_CU_CYCLES') if group == 'active' else val(load, group, 'GRBM_GUI_ACTIVE')
            fr[load] = a / den if a and den else None
            if group == 'active':
                alt = val(load, group, 'SQ_BUSY_CYCLES')
                fr_alt = act.setdefault('_alt_' + ctr, {})
                fr_alt[load] = a / alt if a and alt else None
        cls = 'unidentified'
        if loads:
            r = [fr[f'{l}_w4'] / fr[f'{l}_w1'] for l in loads if fr.get(f'{l}_w1') and fr.get(f'{l}_w4')]
            if r:
                cls = 'per-unit any-active' if max(r) <= 1.2 else ('per-wave sum' if min(r) >= 2.5 else 'ambiguous')
        act[ctr] = {'fraction_of_denominator': fr, 'w4_over_w1': r if loads else None, 'class': cls}
    # VMEM: no occupancy pair of pure memory with equal work; classify with ilp8 vs ilp1 (same bytes, more in flight)
    f = act['SQ_ACTIVE_INST_VMEM']['fraction_of_denominator']
    if f.get('ilp1') and f.get('ilp8'):
        r = f['ilp8'] / f['ilp1']
        act['SQ_ACTIVE_INST_VMEM']['ilp8_over_ilp1'] = r
    out['activity'] = act
    out['busy_cu_cycles'] = {l: val(l, 'active', 'SQ_BUSY_CU_CYCLES') for l in ('mfma4_w4', 'read', 'split', 'fused')}
    return out


def oc4(valid):
    """Candidate A rule (plan section 7), per-CU cycle domain."""
    if 'activity' not in valid:
        return {'verdict': 'unavailable', 'reason': 'counter validation missing'}
    a = valid['activity']
    fv, fm = a['SQ_ACTIVE_INST_VALU']['fraction_of_denominator'], a['SQ_ACTIVE_INST_VMEM']['fraction_of_denominator']
    res = {'candidate': 'A (SQ_ACTIVE_INST_VALU / SQ_ACTIVE_INST_VMEM over SQ_BUSY_CU_CYCLES, dispatch aggregate)', 'checks': {}}
    c = res['checks']
    c['i_valu_any_active'] = a['SQ_ACTIVE_INST_VALU']['class'] == 'per-unit any-active'
    c['i_vmem_ilp8_over_ilp1'] = a['SQ_ACTIVE_INST_VMEM'].get('ilp8_over_ilp1')
    c['i_vmem_any_active'] = c['i_vmem_ilp8_over_ilp1'] is not None and c['i_vmem_ilp8_over_ilp1'] <= 1.2
    c['ii_mfma4_vmem_frac'] = fm.get('mfma4_w4')
    c['ii_read_valu_frac'] = fv.get('read')
    c['ii'] = c['ii_mfma4_vmem_frac'] is not None and c['ii_mfma4_vmem_frac'] <= 0.05 and c['ii_read_valu_frac'] is not None \
        and c['ii_read_valu_frac'] <= 0.2
    lb = lambda l: max(0.0, (fv.get(l) or 0) + (fm.get(l) or 0) - 1) if fv.get(l) is not None and fm.get(l) is not None else None
    ub = lambda l: min(fv.get(l), fm.get(l)) if fv.get(l) is not None and fm.get(l) is not None else None
    c['iii_split_lower'] = lb('split')
    c['iv_fused_lower'] = lb('fused')
    c['iii'] = c['iii_split_lower'] is not None and c['iii_split_lower'] <= 0.05
    c['iv'] = c['iv_fused_lower'] is not None and c['iv_fused_lower'] > 0
    c['fractions_in_unit_interval'] = all(0 <= v <= 1.05 for v in list(fv.values()) + list(fm.values()) if v is not None)
    ok = c['i_valu_any_active'] and c['i_vmem_any_active'] and c['ii'] and c['iii'] and c['iv'] and c['fractions_in_unit_interval']
    res['verdict'] = 'bounded (per-CU scope)' if ok else 'unavailable'
    res['bounds_on_controls'] = {l: {'lower': lb(l), 'upper': ub(l)} for l in ('split', 'fused', 'mfma4_w4', 'read')}
    res['note'] = 'never device-level physical HBM activity; that observable remains unavailable'
    return res


# ---------------------------------------------------------------- references and kernels
def refs(root, plan, att):
    out = {}
    for s in plan['steps']:
        if s['experiment'] != 'oc23_refs':
            continue
        rec = att.get(s['id'])
        f = root / s['out'] / 'attempt1' / 'out.json'
        if rec and Path(rec['attempt_dir'], 'out.json').exists():
            f = Path(rec['attempt_dir']) / 'out.json'
        r = jl(f) if f.exists() else None
        if not r:
            out[s['cell']] = {'status': 'missing', 'returncode': rec.get('returncode') if rec else None}
            continue
        ms = r.get('trial_ms') or []
        e = {'mode': r.get('mode'), 'mask': s.get('level'), 'blocks': r.get('blocks'), 'threads': r.get('threads'),
             'footprint_bytes': r.get('footprint_bytes'), 'median_ms': r.get('median_ms'), 'trial_ms': ms,
             'cv': st.pstdev(ms) / st.mean(ms) if len(ms) > 1 else None, 'check_passed': r.get('check_passed'),
             'useful_gb_per_s': r.get('useful_gb_per_s'), 'useful_tflop_per_s': r.get('useful_tflop_per_s'),
             'stream_cus': r.get('stream_cus')}
        if r.get('mode') in ('mfma4', 'mfma4_lds') and r.get('useful_tflop_per_s'):
            e['mfma4_wave_instr_per_s'] = r['useful_tflop_per_s'] * 1e12 / MFMA4_FLOPS
        out[s['cell']] = e
    return out


def kernels(root, plan, att):
    rows = []
    for s in plan['steps']:
        if s['experiment'] != 'oc16_kernels':
            continue
        rec = att.get(s['id'])
        f = Path(rec['attempt_dir']) / 'k.json' if rec else None
        if not f or not f.exists():
            rows.append({'cell': s['cell'], 'status': 'missing'})
            continue
        d = json.loads(f.read_text())
        for r in d['rows']:
            rows.append(dict(r, cell=s['cell'], mask=s.get('level'), device_cus=d['device_cus']))
    return rows


# ---------------------------------------------------------------- OC1 eager demand and graph timeline
def eager_demand(root, plan, att):
    out = {}
    for s in plan['steps']:
        if s['experiment'] != 'oc1_eager':
            continue
        rec = att.get(s['id'])
        if not rec:
            continue
        d = Path(rec['attempt_dir'])
        mk = next(d.rglob('*marker_api_trace.csv'), None)
        cc = next(d.rglob('*counter_collection.csv'), None)
        run = d / 'run' / 'run.json'
        if not (mk and cc and run.exists()):
            out[s['cell']] = {'status': 'missing files', 'returncode': rec.get('returncode')}
            continue
        dec = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in csv.DictReader(mk.open()) if r['Function'] == 'decode')
        starts = [lo for lo, hi in dec]
        rj = json.loads(run.read_text())
        its = {t['decode_iterations'] for c in rj.get('cases', []) for t in c['trials']}
        ticks = its.pop() if len(its) == 1 else None
        fam = collections.defaultdict(lambda: collections.defaultdict(float))
        seen = set()
        import bisect
        for r in csv.DictReader(cc.open()):
            t = int(r['Start_Timestamp'])
            i = bisect.bisect_right(starts, t) - 1
            if i < 0 or t > dec[i][1]:
                continue
            f = family(r['Kernel_Name'])
            fam[f][r['Counter_Name']] += float(r['Counter_Value'])
            key = (r.get('Dispatch_Id') or r.get('Correlation_Id'))
            if key not in seen:
                seen.add(key)
                fam[f]['_dispatches'] += 1
                fam[f]['_duration_ns'] += int(r['End_Timestamp']) - t
        n = len(dec) * ticks if ticks else None
        out[s['cell']] = {'decode_ranges': len(dec), 'ticks_per_range': ticks,
                          'per_tick': {f: {k: v / n for k, v in c.items()} for f, c in fam.items()} if n else None,
                          'note': 'eager, profiled (replay per pass); per tick = sum over decode marker ranges / (ranges x ticks)'}
    return out


def graph_timeline(root, plan, att):
    out = {}
    for s in plan['steps']:
        if s['experiment'] != 'oc1_graph' or not s.get('profile'):
            continue
        rec = att.get(s['id'])
        if not rec:
            continue
        d = Path(rec['attempt_dir'])
        kt = next(d.rglob('*kernel_trace.csv'), None)
        mk = next(d.rglob('*marker_api_trace.csv'), None)
        if not kt:
            out[s['cell']] = {'status': 'missing kernel trace'}
            continue
        ks = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name']) for r in csv.DictReader(kt.open()))
        dec = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in csv.DictReader(mk.open()) if r['Function'] == 'decode') if mk else []
        fam = collections.defaultdict(lambda: {'n': 0, 'busy_ns': 0})
        union = gaps = 0
        span = 0
        for lo, hi in dec:
            sel = [k for k in ks if lo <= k[0] <= hi]
            if not sel:
                continue
            span += sel[-1][1] - sel[0][0]
            end = sel[0][0]
            for a, b, name in sel:
                f = family(name)
                fam[f]['n'] += 1
                fam[f]['busy_ns'] += b - a
                if a > end:
                    gaps += a - end
                union += max(0, b - max(a, end))
                end = max(end, b)
        out[s['cell']] = {'decode_ranges': len(dec), 'family': dict(fam), 'kernel_union_ns': union, 'gap_ns': gaps,
                          'first_to_last_kernel_ns': span,
                          'note': 'graph mode, kernel trace only; union and gaps over decode marker ranges; sums of durations '
                                  'are not the execution interval when kernels overlap'}
    return out


def probe(root):
    f = next(root.glob('oc0_probe/**/oc_probe.json'), None)
    if not f:
        return {'status': 'missing'}
    d = json.loads(f.read_text())
    return {r['name']: {'returncode': r['returncode'], 'n_files': r['n_files'], 'wall_s': r['wall_s'],
                        'outputs': [x['path'] for x in r['files']][:12], 'stderr_tail': r['stderr_tail'][-300:]} for r in d['routes']}


def main(a):
    root = a.root
    plan = json.loads((root / 'launch_plan.json').read_text())
    att = final_attempts(root)
    out = root / 'analysis_oc'
    out.mkdir(exist_ok=True)
    v = validation(root, plan)
    tables = {'version': 'oc_v0 reader 1', 'cohort': plan['cohort'], 'kind': plan['kind'], 'plan_sha256': plan.get('plan_sha256'),
              'probe': probe(root), 'counter_validation': v, 'oc4_candidate_A': oc4(v), 'references': refs(root, plan, att),
              'kernels': kernels(root, plan, att), 'oc1_eager_demand': eager_demand(root, plan, att),
              'oc1_graph_timeline': graph_timeline(root, plan, att)}
    (out / 'oc_tables.json').write_text(json.dumps(tables, indent=1, default=str) + '\n')
    lines = [f"# OC-v0 {plan['kind']} — {plan['cohort']}", '', '## Probe (OC0)', '']
    for k, r in tables['probe'].items() if isinstance(tables['probe'], dict) else []:
        if isinstance(r, dict):
            lines.append(f"- {k}: rc {r.get('returncode')}, {r.get('n_files')} files")
    lines += ['', '## Counter validation', '']
    for k, r in v.get('tests', {}).items():
        lines.append(f"- {k}: {r.get('status')} (ratio {r.get('ratio') or r.get('ratio_KiB') or r.get('ratio_req')})")
    for k, r in v.get('activity', {}).items():
        if k.startswith('_alt_'):
            continue
        lines.append(f"- {k}: {r['class']}; w4/w1 {r.get('w4_over_w1')}; ilp8/ilp1 {r.get('ilp8_over_ilp1')}")
    o = tables['oc4_candidate_A']
    lines += ['', f"## OC4 candidate A: **{o['verdict']}**", '', '```', json.dumps(o.get('checks'), indent=1, default=str), '```', '',
              '## References', '', '| cell | mode | median ms | GB/s | TFLOP/s | cv |', '|---|---|---:|---:|---:|---:|']
    for k, r in sorted(tables['references'].items()):
        lines.append(f"| {k} | {r.get('mode')} | {r.get('median_ms')} | {r.get('useful_gb_per_s')} | {r.get('useful_tflop_per_s')} | "
                     f"{round(r['cv'], 4) if r.get('cv') else ''} |")
    lines += ['', '## Kernel controls (median µs)', '', '| cell | model | shape | route | history | grid | µs |', '|---|---|---|---|---|---:|---:|']
    for r in tables['kernels']:
        lines.append(f"| {r.get('cell')} | {r.get('model')} | {r.get('shape')} | {r.get('route')} | {r.get('history')} | {r.get('grid')} | "
                     f"{round(r['median_us'], 2) if r.get('median_us') else r.get('error', r.get('status'))} |")
    (out / 'oc_summary.md').write_text('\n'.join(lines) + '\n')
    print(f'oc tables written to {out}')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--root', type=Path, required=True)
    main(q.parse_args())
