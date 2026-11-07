#!/usr/bin/env python3
"""goal G5 extension on gfx942 (MI300X, MI325X): leak test v5 and the G5 tuned-decoder x memory-service surface.
Pre-registration: docs/amd/goal_g5x_plan_20261107.md. Steps run with scripts/amd/cubw/cubw.py under
scripts/amd/goal/goal.sbatch (kind 'g5x' = auto: pilot stage, gate, production stage).

  pilot       placement at every mask used; leak test v5 on every GPU at n = 256 and 128 = the goal_v1 leak v4 suite
              (tuned decoder on 0..n-1 beside the start-delay probe on n..303; natives mem100/arith100 on 0..n-1;
              positives cuprobe/arith100/mem100 on 0..n+7 and the stock decoder on 0..n+7) plus two launch-storm
              negative controls on 0..n-1 (storm_lds, storm_small; svc_bench_v8)
  production  8 blocks (one per GPU) of G5: Qwen32B graph B1/C4096, tuned decoder (grid = n) at n = 256, 192, 128 x
              {absent, inactive, mem25, mem50, mem100, arith100} with the competitor on 256-303; randomized per block.
              Gated by placement only (owner order 2026-11-07); the isolation verdict (leak_ok_v5) sets the disposition
              of the G5 rows in analysis (scripts/amd/goal/analyze_g5x.py), not whether they run.
--smoke: one GPU (mi3001x): placement, leak v5 at 256, and G5 absent/mem100/arith100 at 256; pilot-labelled.

  plan_g5x.py --cohort mi300x|mi325x --out plan.json [--smoke]
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import plan_goal as G

P = G.P
VERSION = 'goal_g5x'
TREATMENTS = ('absent', 'inactive', 'mem25', 'mem50', 'mem100', 'arith100')


def leak_v5(p, lane, stage, rng, levels, full):
    """leak v4 suite with the tuned decoder plus launch-storm negative controls on the decoder mask."""
    for n in levels:
        probe = f'{n}-{full - 1}'
        steps = [lambda: p.competitor_only(lane, stage, 'e1_leak_v4', f'alone_gapprobe_{probe}_{lane}', 'gapprobe', 30, comp_mask=probe)]
        for co, mask in (('mem100', f'0-{n - 1}'), ('arith100', f'0-{n - 1}'), ('storm_lds', f'0-{n - 1}'),
                         ('storm_small', f'0-{n - 1}'), ('cuprobe', f'0-{n + 7}'), ('arith100', f'0-{n + 7}'),
                         ('mem100', f'0-{n + 7}')):
            steps.append(lambda co=co, mask=mask: p.competitor_only(lane, stage, 'e1_leak_v4', f'corun_{co}_{mask}_gapprobe_{probe}_{lane}',
                                                                    'gapprobe', 30, comp_mask=probe, corunner={'treatment': co, 'mask': mask}))
        steps.append(lambda: p.vllm(lane, stage, 'e1_leak_v4', 'qwen32b', n, 'gapprobe', comp_mask=probe, tag=f'_{lane}', extra_env=G.ov(n)))
        steps.append(lambda: p.vllm(lane, stage, 'e1_leak_v4', 'qwen32b', n + 8, 'gapprobe', comp_mask=probe, tag=f'_positive_{lane}'))
        rng.shuffle(steps)
        for f in steps:
            f()


def build(cohort, smoke=False):
    assert G.COHORTS[cohort]['arch'] == 'gfx942', 'G5 extension is for the gfx942 cohorts'
    c = G.configure(cohort)
    f = c['full']
    lanes = ['L0'] if smoke else [f'L{i}' for i in range(8)]
    p = P.Plan('g5x_smoke' if smoke else 'g5x', cohort)
    levels = sorted({f, *c['g5_levels'], *(n + 8 for n in c['leak_levels'])}, reverse=True)
    P.placement_core(p, 'pilot', lanes=lanes, levels=levels)
    for lane in lanes:
        leak_v5(p, lane, 'pilot', random.Random(f'{G.SEED}:{cohort}:g5x:leak5:{lane}'),
                c['leak_levels'][:1] if smoke else c['leak_levels'], f)
    if smoke:
        n = c['g5_levels'][0]
        for t in ('absent', 'mem100', 'arith100'):
            G.cell(p, 'L0', 'pilot', 'g5', 'qwen32b', n, n, treatment=t, comp_mask=c['pool'])
        p.n_lanes, p.partition, p.rate = 1, c['single'], c['single_rate']
    else:
        for b, lane in enumerate(lanes):
            work = [(t, n) for n in c['g5_levels'] for t in TREATMENTS]
            random.Random(f'{G.SEED}:{cohort}:g5x:block{b}').shuffle(work)
            for t, n in work:
                G.cell(p, lane, 'production', 'g5', 'qwen32b', n, n, acq=b, block=b, treatment=t, comp_mask=c['pool'],
                       requires='release_g1')
    p.levels = levels
    return p


def main(a):
    p = build(a.cohort, a.smoke)
    plan = G.finalize(p, a.cohort, Path(a.config_source))
    plan.update({
        'version': VERSION, 'smoke': a.smoke, 'execution_plan': 'docs/amd/goal_g5x_plan_20261107.md',
        'statistics': dict(plan['statistics'], acquisitions='one per block; 8 blocks (G5)'),
        'gates': {'release_g1': 'placement_ok (every mask: exact distinct CUs, per-XCD counts within 1)',
                  'leak_ok_v5': 'see docs/amd/goal_g5x_plan_20261107.md section 3; sets the G5 disposition, not execution',
                  'clock_ok': 'unchanged: pooled in-window gfx and memory clocks max/min <= 1.05, coverage >= 0.5'},
        'binaries': {'gapprobe': 'svc_bench_gfx942_v6', 'storm_lds/storm_small': 'svc_bench_gfx942_v8'},
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
