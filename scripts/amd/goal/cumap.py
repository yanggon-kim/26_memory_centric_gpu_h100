#!/usr/bin/env python3
"""Leak test v6 CU map (docs/amd/goal_g5y_plan_20261109.md): physical CU of every HSA_CU_MASK index on this GPU, and the
shader-engine-balanced probe lists of each leak level. Runs as a cubw 'shell' step in its lane (HIP_VISIBLE_DEVICES =
the lane's GPU, CUBW_STEP_DIR = the attempt directory) and writes <root>/cumap/<lane>.json, which scripts/amd/cubw/cubw.py
reads to resolve '@name' masks.

Map: mask indices are interleaved over the 8 XCDs (index i on XCD i mod 8; contiguous masks give equal per-XCD counts in
every placement proof). A mask that leaves an XCD empty is unsafe (cubw pilot review 2026-10-28: work spilled onto the
empty XCD), so indices are mapped 8 at a time: one svc_bench_v3 placement run on HSA_CU_MASK=<gpu>:j-(j+7) (one index per
XCD) must expose exactly 8 CUs, one per XCD; index j+x is the CU on XCD x. Keys are xcc<<8 | se<<5 | sh<<4 | cu. All
mapped CUs must be distinct. Each balanced list is then re-proven by a placement run on the list itself (comma-list
masks are accepted on this runtime, same review): it must expose exactly the CUs the map predicts.

Balanced lists, per level n (probe indices n..FULL-1): group the probe indices by (XCD, shader engine), c_g = group size,
k = min c_g. The dispatcher sends a launch's workgroups round-robin over XCDs and, within an XCD, over shader engines from
a starting engine that moves between launches (jobs 507396/507397: probe 256-303 has 1,2,2,1 CUs per engine on every
XCD). A list with exactly k indices from every group therefore gets one workgroup per CU in every launch whatever the
starting engine. Two lists: s0 = the first k indices of each group, s1 = the last k. Together they cover every probe
index when 2k >= max c_g; otherwise the level is marked infeasible and gets no masks (its leak v6 steps fail to start and
the level is 'unavailable'). Positive overlap set of list s: from each XCD the index of that list in the lowest shader
engine (8 CUs); positive mask = 0..n-1 plus those 8.

  cumap.py --lane L0 --levels 256 192 128 --full 304
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess


def placement_keys(bench, gpu, mask, expect):
    env = dict(os.environ, HIP_VISIBLE_DEVICES=gpu, HSA_CU_MASK=f'{gpu}:{mask}')
    r = subprocess.run([bench, '--mode', 'placement', '--blocks', str(8 * 304), '--trials', '3', '--expect-cus', str(expect)],
                       env=env, capture_output=True, text=True, timeout=120)
    lines = r.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1]).get('cu_keys', []) if lines else []
    except ValueError:
        return []


def lists_for(level, full, key):
    groups = collections.defaultdict(list)
    for i in range(level, full):
        k = key[i]
        groups[(k >> 8, (k >> 5) & 7)].append(i)
    c = {g: len(v) for g, v in groups.items()}
    k = min(c.values())
    out = {'groups': {f'xcd{g[0]}_se{g[1]}': v for g, v in sorted(groups.items())}, 'k': k,
           'feasible': 2 * k >= max(c.values())}
    if not out['feasible']:
        return out
    subsets = [sorted(i for v in groups.values() for i in v[:k]), sorted(i for v in groups.values() for i in v[-k:])]
    pos = []
    for s in subsets:
        by_xcd = {}
        for g in sorted(groups):  # lowest shader engine first
            pick = [i for i in groups[g] if i in s]
            if pick and g[0] not in by_xcd:
                by_xcd[g[0]] = pick[0]
        pos.append(sorted(by_xcd.values()))
    out.update(subsets=subsets, positive_overlap=pos, covered=sorted(set(subsets[0]) | set(subsets[1])) == list(range(level, full)))
    return out


def main(a):
    adir = Path(os.environ['CUBW_STEP_DIR'])
    root = next(p for p in adir.parents if (p / 'launch_plan.json').exists())
    gpu = os.environ['HIP_VISIBLE_DEVICES']
    bench = os.environ.get('CUBW_BENCH', str(Path(os.environ.get('WORK', '')) / 'bin/svc_bench_gfx942_v3'))
    lo = min(a.levels)
    assert lo % 8 == 0 and a.full % 8 == 0
    raw, key = {}, {}
    for j in range(lo, a.full, 8):
        ks = placement_keys(bench, gpu, f'{j}-{j + 7}', 8)
        raw[f'{j}-{j + 7}'] = ks
        by_xcd = {k >> 8: k for k in ks}
        if len(ks) == 8 and sorted(by_xcd) == list(range(8)):
            key.update({j + x: by_xcd[x] for x in range(8)})
    map_ok = len(key) == a.full - lo and len(set(key.values())) == len(key)
    doc = {'lane': a.lane, 'gpu_index': int(gpu), 'indices': [lo, a.full - 1], 'map_ok': map_ok,
           'placement_runs': raw, 'keys': {str(i): v for i, v in sorted(key.items())}, 'levels': {}, 'masks': {}}
    if map_ok:
        for n in a.levels:
            L = lists_for(n, a.full, key)
            doc['levels'][str(n)] = L
            if not (L['feasible'] and L['covered']):
                continue
            L['list_proof'] = []
            for s in L['subsets']:
                ks = placement_keys(bench, gpu, ','.join(map(str, s)), len(s))
                L['list_proof'].append({'cus': len(ks), 'match': sorted(ks) == sorted(key[i] for i in s)})
            if all(x['match'] for x in L['list_proof']):
                for j, (s, p) in enumerate(zip(L['subsets'], L['positive_overlap'])):
                    doc['masks'][f'probe_n{n}_s{j}'] = ','.join(map(str, s))
                    doc['masks'][f'pos_n{n}_s{j}'] = f'0-{n - 1},' + ','.join(map(str, p))
    (root / 'cumap').mkdir(exist_ok=True)
    text = json.dumps(doc, indent=1) + '\n'
    (root / 'cumap' / f'{a.lane}.json').write_text(text)
    (adir / 'cumap.json').write_text(text)
    print(json.dumps({'map_ok': map_ok, 'levels': {n: {k: v for k, v in L.items() if k in ('k', 'feasible', 'covered')}
                                                    for n, L in doc['levels'].items()}, 'masks': sorted(doc['masks'])}))
    raise SystemExit(0 if map_ok else 1)


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--lane', required=True)
    q.add_argument('--levels', type=int, nargs='+', required=True)
    q.add_argument('--full', type=int, default=304)
    main(q.parse_args())
