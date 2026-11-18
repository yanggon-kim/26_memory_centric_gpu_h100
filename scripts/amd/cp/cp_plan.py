#!/usr/bin/env python3
"""Chunked-prefill campaign plans (docs/amd/cp_campaign_design_20261118.md; handoff
docs/amd/chunked_prefill_campaign_20261118.md). One plan = one persistent run root (scripts/amd/cp/cp.sbatch,
cp_run.py). Steps are assigned to lanes by longest-estimate-first so lanes finish together.

Plans:
  s1    experiment 1 (step replay, both models, full / intermediate / reduced masks, stock and tuned grids, pure-prefill
        anchors), instrumented kernel-profile replays, experiment 3 step level (prefix length, warm vs cold), the SLO and
        capacity pilot (stock full allocation, development seeds), experiment 4 (mixed-step competitor matrix with
        placement proofs) and the KV link benchmark.
  s2    experiment 2 development stage and experiment 3 serving (rates and SLOs from s1; written later)
  s3    experiment 2 held-out stage and experiment 5 (written later)

  cp_plan.py s1 --cohort mi300x --out plan.json [--smoke]
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

COHORTS = {
    'mi300x': {'full': 304, 'arch': 'gfx942', 'gpu_name': 'MI300X', 'partition': 'mi3008x', 'rate': 1.0, 'pool': '256-303',
               'mid': 192, 'low': 128, 'comp_levels': (256, 192, 128)},
    'mi325x': {'full': 304, 'arch': 'gfx942', 'gpu_name': 'MI325X', 'partition': 'mi3258x', 'rate': 1.2, 'pool': '256-303',
               'mid': 192, 'low': 128, 'comp_levels': (256, 192, 128)},
    'mi350x': {'full': 256, 'arch': 'gfx950', 'gpu_name': 'MI350X', 'partition': 'mi3508x', 'rate': 1.4, 'pool': '224-255',
               'mid': 192, 'low': 128, 'comp_levels': (224, 192, 128)},
}
CFG = {'q32': 'scripts/amd/cp/configs/qwen32b.json', 'q7': 'scripts/amd/cp/configs/qwen7b.json'}
SEED = 20261118
E1_CHUNKS = '0,16,32,64,128,256,512,768,1024,1536,2048'
E3_PREFIX = '0,1024,2048,4096,8192'
TREATMENTS = ('absent', 'inactive', 'mem25', 'mem50', 'mem100', 'arith100')
# SLO / capacity pilot (stock full allocation, mixed policy, budget 2048, development seed 101 for the SLO pilot at low
# load and 102 for the capacity ladder). Rates in requests/s.
PILOT = {'q32': {'dheavy': (0.2, [0.5, 1, 1.5, 2, 2.5, 3, 4]), 'balanced': (0.2, [0.5, 1, 1.5, 2, 3, 4, 5]),
                 'pheavy': (0.1, [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2])},
         'q7': {'dheavy': (0.5, [2, 4, 6, 8, 10, 12, 16]), 'balanced': (0.5, [2, 4, 6, 8, 10, 14, 18]),
                'pheavy': (0.25, [1, 2, 3, 4, 5, 6, 8])}}


class Plan:
    def __init__(self, name, cohort):
        self.c = COHORTS[cohort]
        self.name, self.cohort = name, cohort
        self.steps = []

    def add(self, stage, exp, sid, script, args, est, cus=None, grid=None, treatment='absent', kind='engine', pin=None, **k):
        self.steps.append(dict(id=sid, stage=stage, experiment=exp, kind=kind, script=script, args=[str(x) for x in args],
                               cus=cus, grid=grid, treatment=treatment, est_s=est, timeout_s=int(est * 3 + 900), pin=pin, **k))

    def allocs(self, which):
        c = self.c
        a = {'full': (c['full'], None), f"{c['mid']}s": (c['mid'], None), f"{c['mid']}t": (c['mid'], c['mid']),
             f"{c['low']}s": (c['low'], None), f"{c['low']}t": (c['low'], c['low'])}
        return {k: a[k] for k in which}

    def assign(self, stage, lanes):
        """Longest-estimate-first onto lanes (pinned steps keep their lane); returns per-lane seconds."""
        load = {ln: 0 for ln in lanes}
        todo = [s for s in self.steps if s['stage'] == stage]
        for s in todo:
            if s.get('pin'):
                s['lane'] = s['pin']
                load[s['lane']] += s['est_s']
        for s in sorted((s for s in todo if not s.get('pin')), key=lambda s: -s['est_s']):
            ln = min(load, key=lambda x: (load[x], x))
            s['lane'] = ln
            load[ln] += s['est_s']
        rng = random.Random(f'{SEED}:{self.cohort}:{self.name}:{stage}:order')
        for ln in lanes:  # randomized execution order within a lane
            mine = [s for s in self.steps if s['stage'] == stage and s['lane'] == ln]
            rng.shuffle(mine)
            for i, s in enumerate(mine):
                s['order'] = i
        return load


def s1(cohort, smoke=False):
    p = Plan('s1_smoke' if smoke else 's1', cohort)
    c = p.c
    full, low, mid = c['full'], c['low'], c['mid']
    R = 'scripts/amd/cp/cp_replay.py'
    S = 'scripts/amd/cp/cp_serve.py'
    reps = 2 if smoke else 10
    # experiment 1
    for m, allocs in (('q32', ('full', f'{mid}s', f'{mid}t', f'{low}s', f'{low}t')), ('q7', ('full', f'{low}s', f'{low}t'))):
        if smoke and m == 'q7':
            allocs = ('full',)
        for name, (cus, grid) in p.allocs(allocs).items():
            if smoke and name not in ('full', f'{low}t'):
                continue
            p.add('a', 'e1', f'e1_{m}_{name}', R, ['--config', CFG[m], '--gpu-uuid', '@UUID', '--contexts', '1024,8192',
                  '--widths', '1,8,32', '--chunks', E1_CHUNKS if not smoke else '0,64,512,2048', '--reps', reps,
                  '--budget', 4096, '--anchors', '--seed-offset', 1], est=900 if m == 'q32' else 600,
                  cus=cus, grid=grid, model=m, alloc=name)
    # instrumented kernel profile (rocprofv3 kernel + marker trace; timings not used for headroom)
    for name, (cus, grid) in p.allocs(('full', f'{low}t')).items():
        p.add('a', 'e1p', f'e1p_q32_{name}', R, ['--config', CFG['q32'], '--gpu-uuid', '@UUID', '--contexts', '8192', '--widths', '8',
              '--chunks', '0,256,1024', '--reps', 3, '--budget', 4096, '--markers', '--seed-offset', 2], est=900, cus=cus,
              grid=grid, profile=True, model='q32', alloc=name)
    # experiment 3, step level
    for m in ('q32', 'q7'):
        for name, (cus, grid) in p.allocs(('full', f'{low}t')).items():
            if smoke and (m == 'q7' or name != 'full'):
                continue
            p.add('a', 'e3s', f'e3s_{m}_{name}', R, ['--config', CFG[m], '--gpu-uuid', '@UUID', '--contexts', '1024',
                  '--widths', '8,32', '--prefix', E3_PREFIX, '--suffix', 256, '--reps', 2 if smoke else 8, '--budget', 8704,
                  '--seed-offset', 3], est=900 if m == 'q32' else 600, cus=cus, grid=grid, model=m, alloc=name)
    # SLO and capacity pilot (stock full allocation; mixed, budget 2048; development seeds)
    for m in ('q32', 'q7'):
        for wl, (lo, ladder) in PILOT[m].items():
            if smoke and (m == 'q7' or wl != 'balanced'):
                continue
            p.add('a', 'pilot', f'pilot_slo_{m}_{wl}', S, ['--config', CFG[m], '--gpu-uuid', '@UUID', '--policy', 'mixed',
                  '--budget', 2048, '--workload', wl, '--rates', lo, '--seeds', 101, '--requests', 40, '--drain-s', 120],
                  est=int(40 / lo + 400), cus=full, model=m, alloc='full')
            lad = ladder if not smoke else ladder[2:4]
            n = 120 if not smoke else 40
            for name, (cus, grid) in p.allocs(('full', f'{low}t')).items():
                p.add('a', 'pilot', f'pilot_cap_{m}_{wl}_{name}', S, ['--config', CFG[m], '--gpu-uuid', '@UUID', '--policy', 'mixed',
                      '--budget', 2048, '--workload', wl, '--rates', ','.join(map(str, lad)), '--seeds', 102,
                      '--requests', n, '--drain-s', 60, '--energy'], est=int(sum(n / r + 70 for r in lad) + 400),
                      cus=cus, grid=grid, model=m, alloc=name)
    # experiment 4: blocks of the decoder-mask x competitor matrix, mixed iterations held fixed
    blocks = 1 if smoke else 4
    for b in range(blocks):
        lane = f'L{b}'
        for n in c['comp_levels']:
            p.add('a', 'e4', f'e4_place_{n}_b{b}', None, ['--mode', 'placement', '--blocks', 8 * full, '--trials', 5,
                  '--expect-cus', n], est=10, kind='native', mask=f'0-{n - 1}', pin=lane)
        p.add('a', 'e4', f'e4_place_pool_b{b}', None, ['--mode', 'placement', '--blocks', 8 * full, '--trials', 5,
              '--expect-cus', full - c['comp_levels'][0]], est=10, kind='native', mask=c['pool'], pin=lane)
        cells = [(n, t) for n in c['comp_levels'] for t in TREATMENTS]
        if smoke:
            cells = [(c['comp_levels'][0], t) for t in ('absent', 'mem100')]
        for n, t in cells:
            p.add('a', 'e4', f'e4_q32_n{n}_{t}_b{b}', R, ['--config', CFG['q32'], '--gpu-uuid', '@UUID', '--contexts', '1024,8192',
                  '--widths', '32', '--chunks', '0,256,1024', '--reps', 2 if smoke else 8, '--budget', 4096,
                  '--seed-offset', 4], est=420, cus=n, grid=n, treatment=t, pin=lane, model='q32', alloc=f'{n}t', block=b)
    # KV link benchmark needs two GPUs: stage b on lane pairs
    p.add('b', 'e5link', 'e5link_q32', 'scripts/amd/cp/cp_link.py', ['--config', CFG['q32'], '--gpu-indices', '@IDX'], est=120,
          pin='P01', model='q32')
    p.add('b', 'e5link', 'e5link_q7', 'scripts/amd/cp/cp_link.py', ['--config', CFG['q7'], '--gpu-indices', '@IDX'], est=120,
          pin='P23', model='q7')
    lanes = {'a': {f'L{i}': [i] for i in range(8)}, 'b': {'P01': [0, 1], 'P23': [2, 3]}}
    if smoke:
        lanes['a'] = {'L0': [0]}
        for s in p.steps:
            if s['stage'] == 'a':
                s['pin'] = 'L0'
    load = {st: p.assign(st, list(lanes[st])) for st in lanes}
    return p, lanes, load


def finalize(p, lanes, load):
    c = p.c
    plan = {'campaign': 'cp_20261118', 'plan': p.name, 'cohort': p.cohort, 'arch': c['arch'], 'full_cus': c['full'],
            'gpu_name': c['gpu_name'], 'gpus': 8, 'partition': c['partition'], 'charge_rate': c['rate'],
            'competitor_mask': c['pool'], 'stages': sorted(lanes), 'lanes': lanes, 'reserve_s': 900,
            'design': 'docs/amd/cp_campaign_design_20261118.md', 'handoff': 'docs/amd/chunked_prefill_campaign_20261118.md',
            'steps': sorted(p.steps, key=lambda s: (s['stage'], s['lane'], s['order']))}
    longest = {st: max(v.values()) for st, v in load.items()}
    plan['budget'] = {'lane_seconds': load, 'longest_lane_s': longest,
                      'node_hours_est': round(sum(longest.values()) / 3600 + 0.3, 2),
                      'allocation_units_est': round((sum(longest.values()) / 3600 + 0.3) * c['rate'], 2)}
    text = json.dumps(plan, indent=1) + '\n'
    plan['plan_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return plan


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('which', choices=('s1',))
    q.add_argument('--cohort', choices=sorted(COHORTS), required=True)
    q.add_argument('--smoke', action='store_true')
    q.add_argument('--out', type=Path, required=True)
    a = q.parse_args()
    p, lanes, load = {'s1': s1}[a.which](a.cohort, a.smoke)
    plan = finalize(p, lanes, load)
    a.out.write_text(json.dumps(plan, indent=1) + '\n')
    print(f"{plan['plan']}/{a.cohort}: {len(plan['steps'])} steps; longest lane per stage {plan['budget']['longest_lane_s']}; "
          f"~{plan['budget']['node_hours_est']} node-h ~{plan['budget']['allocation_units_est']} units")
