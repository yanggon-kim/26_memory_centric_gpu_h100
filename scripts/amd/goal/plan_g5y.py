#!/usr/bin/env python3
"""goal G5 on gfx942 with leak test v6 (MI300X, MI325X): shader-engine-balanced start-delay probe, then the G5 surface.
Pre-registration: docs/amd/goal_g5y_plan_20261109.md. Steps run with scripts/amd/cubw/cubw.py under
scripts/amd/goal/goal.sbatch (kind 'g5y' = auto: pilot stage, gate, production stage).

  pilot       placement at every mask used; CU map (scripts/amd/goal/cumap.py: physical CU of every probe index and two
              shader-engine-balanced probe lists s0, s1 per level); leak test v6 on every GPU at n = 256, 192 and 128:
              for each list s, the start-delay probe (svc_bench_v6 gapprobe, one full-LDS workgroup per CU of the list)
              alone, beside four non-overlapping natives on 0..n-1 (mem100, arith100, storm_lds, storm_small), beside the
              positive (full-LDS cuprobe on 0..n-1 plus 8 CUs of the list), and beside the tuned decoder on 0..n-1
  production  8 blocks (one per GPU) of G5: Qwen32B graph B1/C4096, tuned decoder (grid = n) at n = 256, 192, 128 x
              {absent, inactive, mem25, mem50, mem100, arith100} with the competitor on 256-303; randomized per block.
              Gated by placement only (owner order 2026-11-07); leak_ok_v6 (scripts/amd/goal/analyze_g5y.py) sets the
              disposition of the G5 rows, not whether they run.
--smoke: one GPU (mi3001x): placement, CU map, leak v6 at all three levels, and G5 absent/mem100/arith100 at 256;
pilot-labelled.

  plan_g5y.py --cohort mi300x|mi325x --out plan.json [--smoke]
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import plan_goal as G

P = G.P
VERSION = 'goal_g5y'
TREATMENTS = ('absent', 'inactive', 'mem25', 'mem50', 'mem100', 'arith100')
LEAK_LEVELS = (256, 192, 128)
NATIVES = ('mem100', 'arith100', 'storm_lds', 'storm_small')


def leak_v6(p, lane, stage, rng, levels):
    """Per level and balanced list s: probe alone, beside natives on 0..n-1, beside the positive, beside the decoder."""
    for n in levels:
        steps = []
        for j in (0, 1):
            probe = f'@probe_n{n}_s{j}'
            meta = lambda role, j=j, n=n: {'n': n, 'subset': j, 'role': role}
            steps.append(lambda probe=probe, j=j, n=n, meta=meta: p.competitor_only(
                lane, stage, 'e1_leak_v6', f'alone_n{n}_s{j}_{lane}', 'gapprobe', 30, comp_mask=probe).update(leak6=meta('alone')))
            for co in NATIVES:
                steps.append(lambda co=co, probe=probe, j=j, n=n, meta=meta: p.competitor_only(
                    lane, stage, 'e1_leak_v6', f'corun_{co}_0-{n - 1}_n{n}_s{j}_{lane}', 'gapprobe', 30, comp_mask=probe,
                    corunner={'treatment': co, 'mask': f'0-{n - 1}'}).update(leak6=dict(meta('native'), treatment=co)))
            steps.append(lambda probe=probe, j=j, n=n, meta=meta: p.competitor_only(
                lane, stage, 'e1_leak_v6', f'corun_cuprobe_pos_n{n}_s{j}_{lane}', 'gapprobe', 30, comp_mask=probe,
                corunner={'treatment': 'cuprobe', 'mask': f'@pos_n{n}_s{j}'}).update(leak6=meta('positive')))
            steps.append(lambda probe=probe, j=j, n=n, meta=meta: p.vllm(
                lane, stage, 'e1_leak_v6', 'qwen32b', n, 'gapprobe', comp_mask=probe, tag=f'_s{j}_{lane}',
                extra_env=G.ov(n)).update(leak6=meta('decoder')))
        rng.shuffle(steps)
        for f in steps:
            f()


def build(cohort, smoke=False):
    assert G.COHORTS[cohort]['arch'] == 'gfx942', 'leak v6 is for the gfx942 cohorts'
    c = G.configure(cohort)
    f = c['full']
    lanes = ['L0'] if smoke else [f'L{i}' for i in range(8)]
    p = P.Plan('g5y_smoke' if smoke else 'g5y', cohort)
    levels = sorted({f, *c['g5_levels'], *LEAK_LEVELS}, reverse=True)
    P.placement_core(p, 'pilot', lanes=lanes, levels=levels)
    for lane in lanes:
        p.shell(lane, 'pilot', 'e1_cumap', f'cumap_{lane}',
                f"python3.12 scripts/amd/goal/cumap.py --lane {lane} --full {f} --levels {' '.join(map(str, LEAK_LEVELS))}",
                est=150, timeout=900)
        leak_v6(p, lane, 'pilot', random.Random(f'{G.SEED}:{cohort}:g5y:leak6:{lane}'), LEAK_LEVELS)
    if smoke:
        n = c['g5_levels'][0]
        for t in ('absent', 'mem100', 'arith100'):
            G.cell(p, 'L0', 'pilot', 'g5', 'qwen32b', n, n, treatment=t, comp_mask=c['pool'])
        p.n_lanes, p.partition, p.rate = 1, c['single'], c['single_rate']
    else:
        for b, lane in enumerate(lanes):
            work = [(t, n) for n in c['g5_levels'] for t in TREATMENTS]
            random.Random(f'{G.SEED}:{cohort}:g5y:block{b}').shuffle(work)
            for t, n in work:
                G.cell(p, lane, 'production', 'g5', 'qwen32b', n, n, acq=b, block=b, treatment=t, comp_mask=c['pool'],
                       requires='release_g1')
    p.levels = levels
    return p


def main(a):
    p = build(a.cohort, a.smoke)
    plan = G.finalize(p, a.cohort, Path(a.config_source))
    plan.update({
        'version': VERSION, 'smoke': a.smoke, 'execution_plan': 'docs/amd/goal_g5y_plan_20261109.md',
        'statistics': dict(plan['statistics'], acquisitions='one per block; 8 blocks (G5)'),
        'gates': {'release_g1': 'placement_ok (every mask: exact distinct CUs, per-XCD counts within 1)',
                  'leak_ok_v6': 'see docs/amd/goal_g5y_plan_20261109.md section 3; sets the G5 disposition, not execution',
                  'clock_ok': 'unchanged: pooled in-window gfx and memory clocks max/min <= 1.05, coverage >= 0.5'},
        'binaries': {'placement/cumap': 'svc_bench_gfx942_v3', 'gapprobe': 'svc_bench_gfx942_v6',
                     'cuprobe': 'svc_bench_gfx942_v4', 'storm_lds/storm_small': 'svc_bench_gfx942_v8'},
    })
    plan.pop('plan_sha256', None)
    text = json.dumps(plan, indent=1) + '\n'
    plan['plan_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    a.out.write_text(json.dumps(plan, indent=1) + '\n')
    b = plan['budget']
    print(f"{plan['kind']}/{a.cohort}: {len(plan['steps'])} steps; stages {b['per_stage_longest_lane_s']}; "
          f"~{b['node_hours_est']} node-h ~{b['allocation_units_est']} units")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--cohort', choices=('mi300x', 'mi325x'), required=True)
    q.add_argument('--smoke', action='store_true')
    q.add_argument('--config-source', default='/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/amd_20261018_mi300x')
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
