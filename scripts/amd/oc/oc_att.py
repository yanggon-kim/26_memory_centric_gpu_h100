#!/usr/bin/env python3
"""OC4 candidate B: per-CU compute / memory activity and overlap from decoded thread traces (ATT).

Input: the rocprofv3 --att output directories written by oc_probe.py (decoded `ui_output_agent_*_dispatch_*/` folders:
code.json for the instruction text, one se*_sm*_sl*_wv*.json per traced wave with [time, type, stall, latency, code
index] per issued instruction, gfx clock). One CU (all four SIMDs) is traced per shader engine.

Per traced CU of each dispatch, on the CU window T = [first wave begin, last wave end]:

  c(t) = 1 while any MFMA on the CU is executing: union of [issue, issue + latency] of v_mfma* instructions
  m(t) = 1 while any VMEM load on the CU is outstanding: a load is outstanding from its issue until the end of the
         first s_waitcnt whose vmcnt(N) bound retires it (in-order vmcnt; stores count toward vmcnt but not toward m).
         This is an upper estimate: a load whose waitcnt did not stall may have returned before the waitcnt issued.
         The fraction of loads retired by a non-stalling waitcnt is reported (`loads_retired_without_stall`).
  A_C = |c|, A_M = |m|, X = |c and m|; also the coupled bounds max(0, A_C + A_M - T) <= X <= min(A_C, A_M).

The section 7 checks (execution plan) are evaluated by analyze_oc.py on these numbers. Scope: one CU per SE, replay
and serialization as recorded by the tool; this is never device-level HBM activity.

  oc_att.py --probe-dir <oc0_probe attempt dir> --out <att_activity.json>
"""
import argparse
import json
from pathlib import Path
import re

VMCNT = re.compile(r'vmcnt\((\d+)\)')
VMEM = re.compile(r'^(global|buffer|flat)_(load|store|atomic)')


def union(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def measure(iv):
    return sum(b - a for a, b in iv)


def intersect(x, y):
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b:
            out.append([a, b])
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def wave_intervals(w, code):
    """MFMA execution intervals and VMEM-load outstanding intervals of one wave."""
    mf, ld, queue = [], [], []  # queue: issue times of in-flight VMEM ops (None for stores)
    n_loads = n_nostall = 0
    for t, _typ, stall, lat, idx in w['instructions']:
        text = code[idx][0] if idx < len(code) else ''
        op = text.split(' ', 1)[0]
        if op.startswith('v_mfma'):
            mf.append((t, t + max(lat, 1)))
        elif VMEM.match(op):
            queue.append(t if '_load' in op else None)
        elif op == 's_waitcnt':
            m = VMCNT.search(text)
            if m:
                keep = int(m.group(1))
                end = t + max(lat, 1)
                while len(queue) > keep:
                    t0 = queue.pop(0)
                    if t0 is not None:
                        ld.append((t0, end))
                        n_loads += 1
                        n_nostall += stall == 0
    for t0 in queue:  # never waited on: outstanding at most until the wave ends
        if t0 is not None:
            ld.append((t0, w['end']))
    return mf, ld, n_loads, n_nostall


def dispatch(d):
    code = json.loads((d / 'code.json').read_text())['code']
    name = next((c[3] for c in code if c[0].startswith(';') and c[3]), '')
    cus = {}
    for f in sorted(d.glob('se*_sm*_sl*_wv*.json')):
        w = json.loads(f.read_text())['wave']
        se = f.name.split('_')[0]
        k = cus.setdefault(f'{se}_cu{w["cu"]}', {'mf': [], 'ld': [], 'begin': w['begin'], 'end': w['end'], 'waves': 0,
                                                  'loads': 0, 'nostall': 0, 'mfma': 0})
        mf, ld, nl, ns = wave_intervals(w, code)
        k['mf'] += mf
        k['ld'] += ld
        k['loads'] += nl
        k['nostall'] += ns
        k['mfma'] += len(mf)
        k['waves'] += 1
        k['begin'], k['end'] = min(k['begin'], w['begin']), max(k['end'], w['end'])
    out = {}
    for cu, k in cus.items():
        T = k['end'] - k['begin']
        c, m = union(k['mf']), union(k['ld'])
        AC, AM = measure(c), measure(m)
        X = measure(intersect(c, m))
        out[cu] = {'T_cycles': T, 'waves': k['waves'], 'mfma_instructions': k['mfma'], 'vmem_loads': k['loads'],
                   'A_C': AC, 'A_M': AM, 'X': X, 'A_C_over_T': AC / T if T else None, 'A_M_over_T': AM / T if T else None,
                   'X_over_T': X / T if T else None, 'X_lower': max(0, AC + AM - T), 'X_upper': min(AC, AM),
                   'loads_retired_without_stall': k['nostall'] / k['loads'] if k['loads'] else None}
    return {'kernel': name, 'cus': out}


def main(a):
    res = {'scope': __doc__.strip(), 'routes': {}}
    for route in sorted(a.probe_dir.glob('att_*')):
        disp = {}
        for d in sorted(route.glob('ui_output_agent_*_dispatch_*')):
            r = dispatch(d)
            if r['cus'] and any(v['waves'] for v in r['cus'].values()):
                disp[d.name.rsplit('_', 1)[1]] = r
        res['routes'][route.name] = disp
        for k, r in disp.items():
            for cu, v in r['cus'].items():
                print(route.name, k, r['kernel'][:40], cu, {x: (round(v[x], 3) if isinstance(v[x], float) else v[x])
                                                            for x in ('T_cycles', 'A_C_over_T', 'A_M_over_T', 'X_over_T')}, flush=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--probe-dir', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
