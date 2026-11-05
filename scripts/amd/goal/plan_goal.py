#!/usr/bin/env python3
"""Generate an immutable goal_v1 launch plan (docs/amd/goal_based_followup_20261105.md, G1-G8; execution plan
docs/amd/goal_followup_execution_plan_20261105.md). Steps run with scripts/amd/cubw/cubw.py (kinds vllm, native,
shell, competitor_only, serve) under scripts/amd/goal/goal.sbatch.

Per cohort (MI300X, MI325X, MI350X), one 8-GPU 'auto' allocation:
  pilot stage   placement at every mask used; leak test v4 with the tuned decoder on every GPU (gfx942: also leak v2 on
                two GPUs, the pre-registered fallback); dispatch probe at the G1 levels; G4 counter audit; G6 static ISA
                metadata, service curves and the CU x grid kernel-shape sweep (3 acquisitions); G2 kernel-route traces;
                G7 eager-decode counters
  gate          scripts/amd/goal/analyze_goal.py gate: release_g1 (placement), release_g5 (placement, lifecycle, tuned
                isolation)
  production    8 blocks (one per GPU) of G1 (Qwen32B graph B1/C4096: stock full first and last, tuned full, full mask
                with the reduced grid, tuned reduced levels, stock at the promising and boundary masks); blocks 0-3: G2
                isolated cases and G5 tuned-decoder x memory-service surface; blocks 4-7: G2 mixed serving (C=2, 8),
                G3 decode energy and serving energy
--smoke: the one-GPU version (mi3001x / mi3501x), pilot-labelled, judges nothing.

The CU-count override (CUBW_CU_COUNT_OVERRIDE, scripts/sitecustomize.py) changes only vLLM's ROCm skinny GEMMs:
wvSplitK when the token count n <= 5 (gfx9), wvSplitKrc when 10 <= n <= 128 (gfx950 only). Prefill and batch-8 decode
use hipBLASLt; for those cases the override is recorded as no intervention unless the kernel-route trace shows
otherwise.

  plan_goal.py --cohort mi300x --out plan.json [--smoke]
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cubw'))
import plan as P  # noqa: E402  (cubw plan helpers: Plan, placement_core, dispatch_suite, e4_suite, e6_cmd, derived_configs)

SEED = 20261105
VERSION = 'goal_v1'
COHORTS = {
    'mi300x': {'full': 304, 'arch': 'gfx942', 'gpu_name': 'MI300X', 'partition': 'mi3008x', 'single': 'mi3001x', 'rate': 1.0,
               'single_rate': 0.125, 'n_p': 128, 'n_b': 96, 'tuned': [256, 224, 192, 160, 128, 96], 'full_tuned': 288,
               'pool': '256-303', 'g5_levels': [256, 192, 128], 'leak_levels': (256, 128), 'leak_v2': True,
               'g6_masks': [304, 272, 256, 232, 192, 152, 128, 96], 'g6_counts': [64, 96, 128, 160, 192, 224, 256, 288, 304],
               'e4_masks': [304, 192, 128, 96], 'g7': True},
    'mi325x': {'full': 304, 'arch': 'gfx942', 'gpu_name': 'MI325X', 'partition': 'mi3258x', 'single': None, 'rate': 1.2,
               'n_p': 160, 'n_b': 128, 'tuned': [256, 224, 192, 160, 128, 96], 'full_tuned': 288,
               'pool': '256-303', 'g5_levels': [256, 192, 128], 'leak_levels': (256, 128), 'leak_v2': True,
               'g6_masks': [304, 272, 256, 232, 192, 152, 128, 96], 'g6_counts': [64, 96, 128, 160, 192, 224, 256, 288, 304],
               'e4_masks': [304, 192, 128, 96], 'g7': False},  # counters refused on MI325X (CP firmware 177 < 186, 492957)
    'mi350x': {'full': 256, 'arch': 'gfx950', 'gpu_name': 'MI350X', 'partition': 'mi3508x', 'single': 'mi3501x', 'rate': 1.4,
               'single_rate': 0.175, 'n_p': 192, 'n_b': 128, 'tuned': [224, 192, 160, 128, 96], 'full_tuned': None,
               'pool': '224-255', 'g5_levels': [224, 192, 128], 'leak_levels': (224, 128), 'leak_v2': False,
               'g6_masks': [256, 232, 224, 208, 192, 160, 128, 96], 'g6_counts': [64, 96, 128, 160, 192, 224, 256],
               'e4_masks': [256, 192, 128, 96], 'g7': True},
}
P.CASE.update({'qwen7b_b4': 'qwen7b_graph_b4_c1024', 'qwen32b_serve': 'qwen32b_serve'})
P.EST.update({'qwen7b_b4': 110, 'qwen32b_serve': 400})
G2_CASES = ['qwen32b_c16k', 'qwen7b_b8', 'qwen7b_b4', 'qwen14b', 'mistral7b', 'prefill']  # + qwen32b B1/C4096 from G1
SERVE = {'requests': 32, 'trials': 10, 'prompt_lens': '256,1024,4096,512,2048', 'output_lens': '64,256,128'}
EAGER_COUNTERS = ['FETCH_SIZE', 'WRITE_SIZE']


def ov(g):
    return [f'CUBW_CU_COUNT_OVERRIDE={g}']


def configure(cohort):
    c = COHORTS[cohort]
    P.FULL = c['full']
    P.COMPETITOR_MASK = c['pool']
    return c


def derived(src):
    out = P.derived_configs(src)
    base = json.loads((src / 'configs/qwen32b_graph_b1_c4096.json').read_text())
    serve = dict(base, stage='goal_v1_derived', derived_from='configs/qwen32b_graph_b1_c4096.json',
                 purpose='G2/G3 mixed serving: closed-loop concurrency up to 8 (scripts/amd/goal/serve_mix.py)')
    serve['engine'] = dict(base['engine'], max_num_seqs=8, max_num_batched_tokens=8192, max_model_len=8192,
                           gpu_memory_utilization=0.9)
    serve.pop('config_sha256', None)
    serve['config_sha256'] = hashlib.sha256(json.dumps(serve, sort_keys=True).encode()).hexdigest()
    out['configs/qwen32b_serve.json'] = serve
    return out


def cell(p, lane, stage, exp, model, n, grid, acq=0, block=None, tag='', treatment='absent', **k):
    """A decoder step; grid=None is the stock kernel (device CU count), else CUBW_CU_COUNT_OVERRIDE=grid."""
    gtag = '__stock' if grid is None else f'__g{grid}'
    return p.vllm(lane, stage, exp, model, n, treatment, acq=acq, block=block, tag=gtag + tag,
                  extra_env=None if grid is None else ov(grid), **k)


def serve(p, lane, stage, exp, n, grid, conc, acq, block, mode='timing', trials=None, requests=None, requires=None):
    s = cell(p, lane, stage, exp, 'qwen32b_serve', n, grid, acq=acq, block=block, tag=f'__c{conc}' + ('__energy' if mode == 'energy' else ''),
             config='configs/qwen32b_serve.json', requires=requires)
    s['kind'] = 'serve'
    s['mode'] = mode
    s['serve_args'] = ['--concurrency', conc, '--requests', requests or SERVE['requests'], '--trials', trials or SERVE['trials'],
                       '--prompt-lens', SERVE['prompt_lens'], '--output-lens', SERVE['output_lens']]
    est = {2: 760, 8: 400}[conc] * (requests or SERVE['requests']) / SERVE['requests'] * (trials or SERVE['trials']) / SERVE['trials']
    s['est_s'] = int(est + 150)
    s['timeout_s'] = max(1200, 3 * s['est_s'])
    return s


def leak_v4_tuned(p, lane, stage, rng, levels, full):
    """leak test v4 (v1.6 rule) with the tuned decoder (grid = n) at n; the required decoder positive on 0..n+7 is the
    stock decoder (device-sized grid, so it occupies every CU of its mask, as in 499332)."""
    for n in levels:
        probe = f'{n}-{full - 1}'
        steps = [lambda: p.competitor_only(lane, stage, 'e1_leak_v4', f'alone_gapprobe_{probe}_{lane}', 'gapprobe', 30, comp_mask=probe)]
        for co, mask in (('mem100', f'0-{n - 1}'), ('arith100', f'0-{n - 1}'), ('cuprobe', f'0-{n + 7}'),
                         ('arith100', f'0-{n + 7}'), ('mem100', f'0-{n + 7}')):
            steps.append(lambda co=co, mask=mask: p.competitor_only(lane, stage, 'e1_leak_v4', f'corun_{co}_{mask}_gapprobe_{probe}_{lane}',
                                                                    'gapprobe', 30, comp_mask=probe, corunner={'treatment': co, 'mask': mask}))
        steps.append(lambda: p.vllm(lane, stage, 'e1_leak_v4', 'qwen32b', n, 'gapprobe', comp_mask=probe, tag=f'_{lane}', extra_env=ov(n)))
        steps.append(lambda: p.vllm(lane, stage, 'e1_leak_v4', 'qwen32b', n + 8, 'gapprobe', comp_mask=probe, tag=f'_positive_{lane}'))
        rng.shuffle(steps)
        for f in steps:
            f()


def leak_v2_tuned(p, lane, stage, rng, levels, full):
    """gfx942 fallback (pre-registered): leak test v2 (v1.3 rule) with the tuned decoder."""
    for n in levels:
        probe = f'{n}-{full - 1}'
        steps = [lambda: p.competitor_only(lane, stage, 'e1_leak_v2', f'alone_cuprobe_{probe}_{lane}', 'cuprobe', 30, comp_mask=probe)]
        for co, mask in (('mem100', f'0-{n - 1}'), ('arith100', f'0-{n - 1}'), ('cuprobe', f'0-{n + 7}')):
            steps.append(lambda co=co, mask=mask: p.competitor_only(lane, stage, 'e1_leak_v2', f'corun_{co}_{mask}_cuprobe_{probe}_{lane}',
                                                                    'cuprobe', 30, comp_mask=probe, corunner={'treatment': co, 'mask': mask}))
        steps.append(lambda: p.vllm(lane, stage, 'e1_leak_v2', 'qwen32b', n, 'cuprobe', comp_mask=probe, tag=f'_{lane}', extra_env=ov(n)))
        rng.shuffle(steps)
        for f in steps:
            f()


def g2_isolated(p, lane, stage, case, n, grid, acq, block, requires=None, profile=None, tag='', exp='g2'):
    if case == 'prefill':
        return cell(p, lane, stage, exp, 'qwen32b', n, grid, acq=acq, block=block, tokens=2, tag='__prefill' + tag,
                    requires=requires, profile=profile)
    cfg = 'configs/qwen32b_graph_b1_c16384.json' if case == 'qwen32b_c16k' else None
    return cell(p, lane, stage, exp, case, n, grid, acq=acq, block=block, config=cfg, requires=requires, profile=profile, tag=tag)


def conditions(c):
    """G2 conditions: (label, mask n, grid). A full/stock; B full/reduced grid; C reduced/reduced grid; D reduced/stock;
    Cb boundary reduced/its grid."""
    f, np_, nb = c['full'], c['n_p'], c['n_b']
    return [('A', f, None), ('B', f, np_), ('C', np_, np_), ('D', np_, None), ('Cb', nb, nb)]


def g1_cells(c):
    f = c['full']
    mid = [(f, c['full_tuned'])] if c['full_tuned'] else []
    mid += [(f, c['n_p'])] if c['n_p'] != c['full_tuned'] else []
    mid += [(n, n) for n in c['tuned']] + [(c['n_p'], None), (c['n_b'], None)]
    return mid


def placement_levels(c):
    lv = {c['full'], c['n_p'], c['n_b'], *c['tuned'], *c['g5_levels'], *c['g6_masks'], *c['e4_masks']}
    lv |= {n + 8 for n in c['leak_levels']}
    return sorted(lv, reverse=True)


def build(cohort, smoke=False):
    c = configure(cohort)
    f = c['full']
    lanes = ['L0'] if smoke else [f'L{i}' for i in range(8)]
    p = P.Plan('pilot' if smoke else 'auto', cohort)
    st = 'pilot'
    levels = placement_levels(c) if not smoke else [f, c['n_p'], c['n_b'], c['leak_levels'][0] + 8]
    P.placement_core(p, st, lanes=lanes, levels=levels)
    rng = random.Random(f'{SEED}:{cohort}:pilot')
    # isolation with the tuned decoder (G5 precondition)
    for lane in lanes:
        leak_v4_tuned(p, lane, st, random.Random(f'{SEED}:{cohort}:leak4:{lane}'),
                      c['leak_levels'][:1] if smoke else c['leak_levels'], f)
    if c['leak_v2'] and not smoke:
        for lane in ('L0', 'L1'):
            leak_v2_tuned(p, lane, st, random.Random(f'{SEED}:{cohort}:leak2:{lane}'), c['leak_levels'], f)
    # G1 placement of the tuned grid (dispatch probe, grid = n / balanced / device size)
    for lane in (['L0'] if smoke else ['L0', 'L1']):
        P.dispatch_suite(p, lane, st, [c['n_p']] if smoke else [f] + c['tuned'], tag=f'_{lane}')
    # G4 counter audit, G6 static metadata
    L = (lambda i: 'L0') if smoke else (lambda i: f'L{i}')
    p.shell(L(2), st, 'g4_counters', 'counter_audit',
            f'$PY scripts/amd/goal/counter_audit.py --out-dir "$CUBW_STEP_DIR" --bench "$CUBW_BENCH"'
            + (' --max-counters 3 --reps 1' if smoke else ''), est=300 if smoke else 2700, timeout=1200 if smoke else 6000)
    p.shell(L(3), st, 'g6_isa', 'isa_meta',
            f'scripts/amd/vllm_in_container.sh scripts/amd/goal/isa_meta.py --arch {c["arch"]} --out "$CUBW_STEP_DIR/isa"',
            est=120, timeout=1800)
    # G6 service curves (native E4 suite) and the CU x grid kernel-shape sweep
    for n in ([c['n_p']] if smoke else c['e4_masks']):
        for name, args in P.e4_suite(n):
            p.native(L(4), st, 'g6_service', name, args, P.mask_for(n), level=n)
    for acq, lane in enumerate(['L0'] if smoke else ['L5', 'L6', 'L7']):
        order = [c['n_p']] if smoke else c['g6_masks'][:]
        rng.shuffle(order)
        for n in order:
            p.shell(lane, st, 'g6_shapes', f'n{n}_a{acq}', P.e6_cmd(n, 'e6.json', light=smoke, cu_counts=c['g6_counts']),
                    est=300, timeout=1500, level=n, acquisition=acq)
    # G2 kernel routes: kernel-trace timelines of every isolated case (and the G1 case) at A and C
    routes = ['qwen32b'] + G2_CASES if not smoke else ['qwen32b', 'qwen7b_b8']
    for i, case in enumerate(routes):
        lane = 'L0' if smoke else ('L3', 'L4')[i % 2]
        for lab, n, g in [x for x in conditions(c) if x[0] in ('A', 'C')]:
            if case == 'qwen32b':
                cell(p, lane, st, 'g2_route', 'qwen32b', n, g, profile={'kind': 'kernel_trace'}, tag='__ktrace')
            else:
                g2_isolated(p, lane, st, case, n, g, 0, None, profile={'kind': 'kernel_trace'}, tag='__ktrace', exp='g2_route')
    # G7: eager decode counters (pre-MALL unless G4 shows otherwise)
    if c['g7']:
        for ctr in (EAGER_COUNTERS[:1] if smoke else EAGER_COUNTERS):
            for lab, n, g in [x for x in conditions(c) if x[0] in ('A', 'C')]:
                s = cell(p, L(1), st, 'g7_counters', 'qwen32b', n, g, config='configs/qwen32b_eager_b1_c4096_profile.json',
                         profile={'kind': 'counters', 'counters': [ctr], 'exclude_regex': '^(_|triton)'}, tag=f'__eager__{ctr}')
                s['timeout_s'] = 1500
    if smoke:
        # one acquisition of each production step kind, unjudged
        for n, g in [(f, None), (c['n_p'], c['n_p'])]:
            cell(p, 'L0', st, 'g1', 'qwen32b', n, g)
        g2_isolated(p, 'L0', st, 'prefill', f, None, 0, None)
        serve(p, 'L0', st, 'g2_serve', f, None, 8, 0, None, trials=2, requests=16)
        serve(p, 'L0', st, 'g3_serve_energy', c['n_p'], c['n_p'], 8, 0, None, mode='energy', trials=2, requests=16)
        cell(p, 'L0', st, 'g3_energy', 'qwen32b', c['n_p'], c['n_p'], mode='energy', tokens=512,
             config='configs_energy/qwen32b_graph_b1_c4096_t10.json')
        n5 = c['g5_levels'][0]
        cell(p, 'L0', st, 'g5', 'qwen32b', n5, n5, treatment='mem100', comp_mask=c['pool'])
        p.n_lanes, p.partition, p.rate = 1, c['single'], c['single_rate']
        return p
    # production: 8 blocks
    pr = 'production'
    for b, lane in enumerate(lanes):
        r = random.Random(f'{SEED}:{cohort}:block{b}')
        cell(p, lane, pr, 'g1', 'qwen32b', f, None, acq=b, block=b, tag='__first', requires='release_g1')
        mid = g1_cells(c)
        r.shuffle(mid)
        for n, g in mid:
            cell(p, lane, pr, 'g1', 'qwen32b', n, g, acq=b, block=b, requires='release_g1')
        if b < 4:
            work = [('iso', case, lab, n, g) for case in G2_CASES for lab, n, g in conditions(c)]
            work += [('g5', None, t, n, n) for n in c['g5_levels'] for t in ('absent', 'inactive', 'mem25', 'mem50', 'mem100', 'arith100')]
        else:
            work = [('serve', conc, lab, n, g) for conc in (2, 8) for lab, n, g in conditions(c)]
            work += [('energy', None, lab, n, g) for lab, n, g in conditions(c) if lab != 'D']
            work += [('serve_energy', 8, lab, n, g) for lab, n, g in conditions(c) if lab in ('A', 'C')]
        r.shuffle(work)
        for kind, x, lab, n, g in work:
            if kind == 'iso':
                g2_isolated(p, lane, pr, x, n, g, b, b, requires='release_g1')
            elif kind == 'g5':
                cell(p, lane, pr, 'g5', 'qwen32b', n, g, acq=b, block=b, treatment=lab, comp_mask=c['pool'], requires='release_g5')
            elif kind == 'serve':
                serve(p, lane, pr, 'g2_serve', n, g, x, b, b, requires='release_g1')
            elif kind == 'energy':
                cell(p, lane, pr, 'g3_energy', 'qwen32b', n, g, acq=b, block=b, mode='energy',
                     config='configs_energy/qwen32b_graph_b1_c4096_t10.json', requires='release_g1')
            else:
                serve(p, lane, pr, 'g3_serve_energy', n, g, x, b, b, mode='energy', requires='release_g1')
        cell(p, lane, pr, 'g1', 'qwen32b', f, None, acq=b, block=b, tag='__last', requires='release_g1')
    p.levels = levels
    return p


def finalize(p, cohort, src):
    c = COHORTS[cohort]
    stages, budget = {}, {}
    for s in p.steps:
        stages.setdefault(s['stage'], {}).setdefault(s['lane'], 0)
        stages[s['stage']][s['lane']] += s['est_s']
        b = budget.setdefault(s['lane'], {'steps': 0, 'est_s': 0})
        b['steps'] += 1; b['est_s'] += s['est_s']
    longest = sum(max(v.values()) for v in stages.values())
    rate = getattr(p, 'rate', c['rate'])
    ids = [s['id'] + s['lane'] for s in p.steps]
    outs = [s['out'] for s in p.steps]
    assert len(set(outs)) == len(outs), 'two steps share an attempt directory'
    return {
        'version': VERSION, 'kind': p.kind, 'cohort': cohort, 'seed': SEED, 'smoke': p.kind == 'pilot',
        'assignment': 'docs/amd/goal_based_followup_20261105.md',
        'execution_plan': 'docs/amd/goal_followup_execution_plan_20261105.md',
        'full_cus': c['full'], 'competitor_mask': c['pool'], 'arch': c['arch'], 'gpu_name': c['gpu_name'],
        'partition': getattr(p, 'partition', c['partition']), 'levels': p.levels if hasattr(p, 'levels') else placement_levels(c),
        'cohort_design': {k: v for k, v in c.items() if k not in ('rate', 'single_rate')},
        'conditions': conditions(c), 'g1_cells': g1_cells(c), 'g2_cases': ['qwen32b'] + G2_CASES, 'serve': SERVE,
        'lanes': {f'L{i}': {'gpu_index': i} for i in range(getattr(p, 'n_lanes', 8))},
        'derived_configs': derived(src), 'config_source': str(src),
        'statistics': {'warmups': 3, 'trials_per_acquisition': 10, 'acquisitions': 'one per block; 8 blocks (G1), 4 blocks (G2/G3/G5)',
                       'bootstrap': '10,000 draws seed 20261018, blocks resampled, trials nested (scripts/amd/cubw/analyze.py ratio_ci)',
                       'noninferiority': f'upper 95% CI of R = T(cell)/T(stock {c["full"]}, same block) <= 1.05 AND clock_ok AND '
                                         'placement_ok AND identical outputs; otherwise observational'},
        'gates': {'release_g1': 'placement_ok (every mask: exact distinct CUs, per-XCD counts within 1)',
                  'release_g5': 'placement_ok and lifecycle_ok and isolation: leak_ok_v4 (v1.6 rule, tuned decoder, every GPU '
                                'and level) if leak v4 is sensitive on every GPU/level; on gfx942 only, if leak v4 is '
                                'insensitive anywhere, leak_ok_v2 (v1.3 rule, tuned decoder) instead; else false',
                  'clock_ok': 'unchanged: pooled in-window gfx and memory clocks max/min <= 1.05, coverage >= 0.5'},
        'unavailable': {'controlled_clocks': 'no authorization to change clocks/power (owner, 2026-11-05): not attempted',
                        'mi325x_counters': 'rocprofv3 refused counters on MI325X (CP firmware 177 < 186, job 492957); the G4 audit '
                                           're-tests once; G7 counters are not planned on MI325X'},
        'retry_policy': 'two attempts maximum, only for execution/tool failures; correctness failures never retried',
        'reserve_s': 2400, 'stages': list(stages),
        'budget': {'per_lane': budget, 'per_stage_longest_lane_s': {k: max(v.values()) for k, v in stages.items()},
                   'longest_lane_s': longest, 'node_hours_est': round(longest / 3600 + 0.4, 2),
                   'allocation_units_est': round((longest / 3600 + 0.4) * rate, 2), 'rate_units_per_node_hour': rate,
                   'rate_source': 'cluster charge multiplier (billing_total = node hours x multiplier)'},
        'steps': p.steps,
    }


def main(a):
    p = build(a.cohort, a.smoke)
    plan = finalize(p, a.cohort, Path(a.config_source))
    text = json.dumps(plan, indent=1) + '\n'
    plan['plan_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    a.out.write_text(json.dumps(plan, indent=1) + '\n')
    b = plan['budget']
    print(f"{plan['kind']}/{a.cohort}: {len(plan['steps'])} steps; stages {b['per_stage_longest_lane_s']}; "
          f"~{b['node_hours_est']} node-h ~{b['allocation_units_est']} units")
    for lane, v in sorted(b['per_lane'].items()):
        print(f"  {lane}: {v['steps']} steps, {v['est_s'] / 60:.0f} min")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--cohort', choices=sorted(COHORTS), required=True)
    q.add_argument('--smoke', action='store_true')
    q.add_argument('--config-source', default='/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/amd_20261018_mi300x')
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
