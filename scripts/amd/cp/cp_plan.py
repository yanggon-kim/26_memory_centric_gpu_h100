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


GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)  # amendment 2026-11-18 (design section 11)
POLICIES = {'whole': ('separate', 8192, True), 'sep2048': ('separate', 2048, False), 'mix2048': ('mixed', 2048, False)}
TUNE = {'mix256': ('mixed', 256, False), 'mix512': ('mixed', 512, False), 'mix1024': ('mixed', 1024, False),
        'mix4096': ('mixed', 4096, False), 'sep512': ('separate', 512, False), 'sep4096': ('separate', 4096, False)}


def rates(lam):
    return ','.join(f'{x * lam:.3g}' for x in GRID)


def serve_step(p, stage, exp, sid, m, wl, pol, cg, lam, seeds, arrivals, n, **k):
    policy, budget, whole = pol
    cus, grid = cg
    rs = [x * lam for x in GRID]
    est = int(len(seeds) * len(arrivals) * sum(n / r + 40 for r in rs) + 420)
    args = ['--config', CFG[m], '--gpu-uuid', '@UUID', '--policy', policy, '--budget', budget, '--workload', wl,
            '--rates', rates(lam), '--seeds', ','.join(map(str, seeds)), '--arrivals', ','.join(arrivals),
            '--requests', n, '--drain-s', 60, '--energy'] + (['--no-chunk'] if whole else [])
    p.add(stage, exp, sid, 'scripts/amd/cp/cp_serve.py', args, est=est, cus=cus, grid=grid, model=m, workload=wl,
          policy=policy, budget=budget, whole=whole, lam=lam, **k)


def s2(cohort, params):
    """Experiment 2 development stage (common budget + tuning) and experiment 3 serving. params = analysis s1 output
    (lambda_star per model/workload at full allocation)."""
    p = Plan('s2', cohort)
    c = p.c
    al = p.allocs(('full', f"{c['mid']}t", f"{c['low']}t"))
    lam = params['lambda_star']
    for m in ('q32', 'q7'):
        for wl in ('dheavy', 'balanced', 'pheavy'):
            L = lam[f'{m}_{wl}']
            for an, a_ in al.items():
                if m == 'q7' and an == f"{c['mid']}t":
                    continue
                for pn, pol in POLICIES.items():
                    serve_step(p, 'a', 'e2c', f'e2c_{m}_{wl}_{an}_{pn}', m, wl, pol, a_, L, [101],
                               ['poisson', 'bursty'], 150 if m == 'q32' else 300, alloc=an, pname=pn)
                if m == 'q32' and an != f"{c['mid']}t":
                    for pn, pol in TUNE.items():
                        serve_step(p, 'a', 'e2t', f'e2t_{m}_{wl}_{an}_{pn}', m, wl, pol, a_, L, [101, 102], ['poisson'],
                                   150, alloc=an, pname=pn)
    # experiment 3 serving: own prefix per request + 256 fresh tokens, output 128, fixed rate, warm vs cold
    for an, (cus, grid) in p.allocs(('full', f"{c['low']}t")).items():
        for P_ in (0, 1024, 4096, 8192):
            for mode in (('warm', 'cold') if P_ else ('none',)):
                p.add('a', 'e3v', f'e3v_q32_{an}_P{P_}_{mode}', 'scripts/amd/cp/cp_serve.py',
                      ['--config', CFG['q32'], '--gpu-uuid', '@UUID', '--policy', 'mixed', '--budget', 2048, '--workload', 'prefix',
                       '--prefix-len', P_, '--prefix-mode', mode, '--suffix', 256, '--output-len', 128,
                       '--rates', params['e3_rate'], '--seeds', 101, '--requests', 40, '--drain-s', 120, '--energy'],
                      est=900, cus=cus, grid=grid, model='q32', alloc=an, P=P_, mode=mode)
    lanes = {'a': {f'L{i}': [i] for i in range(8)}}
    load = {'a': p.assign('a', list(lanes['a']))}
    return p, lanes, load


def s3(cohort, params):
    """Experiment 2 held-out judging and experiment 5. params: lambda_star, tuned[{m}_{wl}_{alloc}_{policy}] = config name
    (from TUNE or POLICIES), best_colocated[{wl}] = {'policy', 'budget', 'whole', 'alloc'}."""
    p = Plan('s3', cohort)
    c = p.c
    al = p.allocs(('full', f"{c['low']}t"))
    lam = params['lambda_star']
    cfgs = dict(POLICIES, **TUNE)
    for wl in ('dheavy', 'balanced', 'pheavy'):
        L = lam[f'q32_{wl}']
        for an, a_ in al.items():
            for pol in ('whole', 'separate', 'mixed'):
                pn = params['tuned'][f'q32_{wl}_{an}_{pol}']
                serve_step(p, 'a', 'e2h', f'e2h_q32_{wl}_{an}_{pol}_{pn}', 'q32', wl, cfgs[pn], a_, L, [201, 202, 203],
                           ['poisson', 'bursty'], 150, alloc=an, pname=pn, family=pol)
    # experiment 5: two-GPU lanes
    D = 'scripts/amd/cp/cp_disagg.py'
    for wl in ('dheavy', 'balanced', 'pheavy'):
        L2 = 2 * lam[f'q32_{wl}']
        rs = ','.join(f'{x * L2:.3g}' for x in GRID)
        best = params['best_colocated'][wl]
        deps = {'single': ['--deployment', 'single', '--rates', ','.join(f'{x * L2 / 2:.3g}' for x in GRID)],
                'colo_stock': ['--deployment', 'colocated', '--rates', rs],
                'colo_tuned': ['--deployment', 'colocated', '--rates', rs, '--policy', best['policy'], '--budget', best['budget']]
                + (['--cus', best['cus'], '--grid', best['grid']] if best.get('cus') else [])
                + (['--no-chunk'] if best.get('whole') else []),
                'disagg_full': ['--deployment', 'disagg', '--rates', rs],
                'disagg_r128': ['--deployment', 'disagg', '--rates', rs, '--decode-cus', c['low'], '--decode-grid', c['low']]}
        for dn, extra in deps.items():
            est = int(sum(150 / (x * L2) + 40 for x in GRID) + 600)
            p.add('b', 'e5', f'e5_q32_{wl}_{dn}', D, ['--config', CFG['q32'], '--gpu-uuids', '@UUIDS', '--gpu-indices', '@IDX',
                  '--workload', wl, '--seeds', 201, '--requests', 150, '--drain-s', 60] + extra, est=est, model='q32',
                  workload=wl, deployment=dn)
    lanes = {'a': {f'L{i}': [i] for i in range(8)}, 'b': {'P01': [0, 1], 'P23': [2, 3], 'P45': [4, 5], 'P67': [6, 7]}}
    load = {st: p.assign(st, list(lanes[st])) for st in lanes}
    return p, lanes, load


def s3r(cohort, params):
    """Experiment 2 held-out stock reference (design section 4 stage iii: mixed, B = 2048, full allocation) on held-out
    seeds 201-203, Poisson and bursty; omitted from s3 and added afterwards (results report, deviations). One step per
    (workload, seed, arrival) so the eight lanes share the work."""
    p = Plan('s3r', cohort)
    al = p.allocs(('full',))
    lam = params['lambda_star']
    for wl in ('dheavy', 'balanced', 'pheavy'):
        for seed in (201, 202, 203):
            for arr in ('poisson', 'bursty'):
                serve_step(p, 'a', 'e2h', f'e2h_q32_{wl}_full_stock_mix2048_s{seed}_{arr}', 'q32', wl, POLICIES['mix2048'],
                           al['full'], lam[f'q32_{wl}'], [seed], [arr], 150, alloc='full', pname='mix2048', family='stock')
    lanes = {'a': {f'L{i}': [i] for i in range(8)}}
    load = {'a': p.assign('a', list(lanes['a']))}
    return p, lanes, load


def x1(cohort, params, max_capture=None, low=None):
    """Cross-cohort reproduction (design section 8): experiment 1 (both models, full and reduced stock/tuned) and
    experiment-2 common-budget points around the MI300X transitions (balanced, pheavy; rates = MI300X lambda_q x GRID;
    SLOs = the MI300X limits as common application targets)."""
    p = Plan('x1' + (f'_cap{max_capture}' if max_capture else '') + (f'_low{low}' if low else ''), cohort)
    if low:  # reduced allocation override (MI350X: 128 CUs blocked for Qwen2.5-32B, design amendment 13)
        p.c = dict(p.c, low=low)
    c = p.c
    R = 'scripts/amd/cp/cp_replay.py'
    low = c['low']
    for m, allocs in (('q32', ('full', f'{low}s', f'{low}t')), ('q7', ('full', f'{low}t'))):
        for name, (cus, grid) in p.allocs(allocs).items():
            p.add('a', 'e1', f'e1_{m}_{name}', R, ['--config', CFG[m], '--gpu-uuid', '@UUID', '--contexts', '1024,8192',
                  '--widths', '1,8,32', '--chunks', E1_CHUNKS, '--reps', 10, '--budget', 4096, '--anchors', '--seed-offset', 1],
                  est=900 if m == 'q32' else 600, cus=cus, grid=grid, model=m, alloc=name)
    lam = params['lambda_star']
    for wl in ('balanced', 'pheavy'):
        for an, a_ in p.allocs(('full', f'{low}t')).items():
            for pn, pol in POLICIES.items():
                serve_step(p, 'a', 'e2c', f'e2c_q32_{wl}_{an}_{pn}', 'q32', wl, pol, a_, lam[f'q32_{wl}'], [101], ['poisson'],
                           150, alloc=an, pname=pn)
    if max_capture:  # platform fallback (MI350X: 32B at 128 CUs hangs capturing mid-size piecewise graphs)
        for st_ in p.steps:
            st_['env'] = {'CP_MAX_CAPTURE': max_capture}
    lanes = {'a': {f'L{i}': [i] for i in range(8)}}
    load = {'a': p.assign('a', list(lanes['a']))}
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
    q.add_argument('which', choices=('s1', 's2', 's3', 's3r', 'x1'))
    q.add_argument('--params', type=Path, help='stage parameters from the s1 analysis')
    q.add_argument('--cohort', choices=sorted(COHORTS), required=True)
    q.add_argument('--smoke', action='store_true')
    q.add_argument('--max-capture', type=int)
    q.add_argument('--low', type=int, help='x1: reduced allocation override')
    q.add_argument('--out', type=Path, required=True)
    a = q.parse_args()
    if a.which == 's1':
        p, lanes, load = s1(a.cohort, a.smoke)
    else:
        prm = json.loads(a.params.read_text())
        p, lanes, load = (x1(a.cohort, prm, a.max_capture, a.low) if a.which == 'x1'
                          else {'s2': s2, 's3': s3, 's3r': s3r}[a.which](a.cohort, prm))
    plan = finalize(p, lanes, load)
    a.out.write_text(json.dumps(plan, indent=1) + '\n')
    print(f"{plan['plan']}/{a.cohort}: {len(plan['steps'])} steps; longest lane per stage {plan['budget']['longest_lane_s']}; "
          f"~{plan['budget']['node_hours_est']} node-h ~{plan['budget']['allocation_units_est']} units")
