#!/usr/bin/env python3
"""Generate an immutable cubw_v1 launch plan (E1-E8 of docs/amd/compute_bandwidth_followup_20261026.md).

Plans are declarative: every step names its lane (one GPU), stage, experiment, cell, acquisition, CU
mask, treatment, config, time estimate and timeout. Orders inside a lane are randomized with a recorded
seed. Kinds:

  pilot        bounded E1 pilots plus capability audits (one acquisition per pilot cell, labelled pilot)
  production   randomized complete blocks: each block (one GPU) holds one acquisition of every E2 level
               and every released E3 cell of one model, so comparisons are paired within a GPU; then E4
               service curves, E5 graph traffic and E8 guardrails. E3 cells require a gate from gates.json
               (full grid only if the pilot clock gate passed; otherwise the predeclared minimal subset).
  auto         one allocation: a reduced pilot stage, then `analyze.py gate` writes gates.json, then the
               production stage (used for the MI325X cohort, whose job cannot be reviewed between stages)

  plan.py --kind pilot --cohort mi300x --out plan.json [--boundary 152]
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

FULL = 304
LEVELS = [304, 232, 200, 152, 96]          # E2 nominal decoder CU levels
E3_LEVELS = [232, 200, 152, 96]             # decoder masks inside CUs 0-231
E3_TREATMENTS = ['absent', 'mem25', 'mem50', 'mem75', 'mem100', 'arith100']
E3_MINIMAL = {'levels': [232, 152], 'treatments': ['absent', 'mem50', 'mem100', 'arith100'], 'model': 'qwen32b'}
COMPETITOR_MASK = '232-303'
CASE = {'qwen32b': 'qwen32b_graph_b1_c4096', 'qwen14b': 'qwen14b_graph_b1_c4096', 'bridge': 'bridge_graph_b1_c1024',
        'qwen7b_b8': 'qwen7b_graph_b8_c4096', 'qwen32b_c16k': 'qwen32b_graph_b1_c16384', 'mistral7b': 'mistral7b_graph_b1_c4096'}
EST = {'qwen32b': 170, 'qwen14b': 125, 'bridge': 95, 'qwen7b_b8': 120, 'mistral7b': 110, 'qwen32b_c16k': 330}
SEED = 20261026


def derived_configs(src):
    """Configs that do not exist in the MI300X set: long-context B1 and a 10-trial energy window."""
    base = json.loads((src / 'configs/qwen32b_graph_b1_c4096.json').read_text())
    c16 = dict(base, context_lengths=[16384], stage='cubw_v1_derived', derived_from='configs/qwen32b_graph_b1_c4096.json',
               purpose='E8 longer-context B1 guardrail')
    c16['engine'] = dict(base['engine'], max_model_len=16640, max_num_batched_tokens=16640)
    en = json.loads((src / 'configs_energy/qwen32b_graph_b1_c4096.json').read_text())
    en10 = dict(en, trials=10, energy_generated_tokens=2048, energy_cases=[[1, 4096]], stage='cubw_v1_derived',
                derived_from='configs_energy/qwen32b_graph_b1_c4096.json',
                purpose='E8 matched long-window energy: 10 trials x 2047 decode ticks at B1/C4096')
    out = {}
    for name, cfg in {'configs/qwen32b_graph_b1_c16384.json': c16, 'configs_energy/qwen32b_graph_b1_c4096_t10.json': en10}.items():
        cfg.pop('config_sha256', None)
        cfg['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        out[name] = cfg
    return out


class Plan:
    def __init__(self, kind, cohort):
        self.steps, self.kind, self.cohort = [], kind, cohort

    def add(self, lane, stage, exp, **k):
        k.update(lane=lane, stage=stage, experiment=exp)
        self.steps.append(k)
        return k

    def vllm(self, lane, stage, exp, model, n, treatment='absent', acq=0, block=None, mode='timing', tokens=None,
             profile=None, comp_mask=None, requires=None, config_dir='configs', config=None, tag=''):
        case = CASE[model]
        cfg = config or f'{config_dir}/{case}.json'
        cell = f'{case}__n{n}__{treatment}{tag}'
        est = EST[model] + (10 if treatment != 'absent' else 0)
        if mode == 'energy':
            est += 900
        if profile:
            est = est * 3 + 120
        if tokens == 2:
            est -= 40
        s = self.add(lane, stage, exp, kind='vllm', id=f'{exp}:{cell}:a{acq}', cell=cell, acquisition=acq, block=block,
                     model=model, case=case, decoder_cus=n, treatment=treatment, mode=mode, config=cfg,
                     out=f'{exp}/{cell}/a{acq}', est_s=est, timeout_s=max(900, 3 * est))
        if tokens:
            s['generated_tokens'] = tokens
        if profile:  # rocprofv3 has hung before; bound the cost of a hang
            s['profile'] = profile
            s['no_retry'] = True
            s['timeout_s'] = 1200
        if comp_mask:
            s['competitor_mask'] = comp_mask
        if requires:
            s['requires'] = requires
        return s

    def native(self, lane, stage, exp, name, args, mask, acq=0, est=8, timeout=120, expect_failure=False, requires=None,
               level=None):
        s = self.add(lane, stage, exp, kind='native', id=f'{exp}:{name}:a{acq}', cell=name, acquisition=acq, mask=mask,
                     level=level, args=args, out=f'{exp}/{name}/a{acq}', est_s=est, timeout_s=timeout)
        if expect_failure:
            s.update(expect_failure=True, no_retry=True)
        if requires:
            s['requires'] = requires
        return s

    def shell(self, lane, stage, exp, name, cmd, est=60, timeout=300):
        return self.add(lane, stage, exp, kind='shell', id=f'{exp}:{name}', cell=name, cmd=cmd,
                        out=f'{exp}/{name}', est_s=est, timeout_s=timeout, no_retry=True)

    def competitor_only(self, lane, stage, exp, name, treatment, seconds, comp_mask=None, acq=0):
        s = self.add(lane, stage, exp, kind='competitor_only', id=f'{exp}:{name}:a{acq}', cell=name, acquisition=acq,
                     treatment=treatment, seconds=seconds, out=f'{exp}/{name}/a{acq}', est_s=seconds + 20,
                     timeout_s=seconds + 120, no_retry=True)
        if comp_mask:
            s['competitor_mask'] = comp_mask
        return s


def mask_for(n):
    return None if n >= FULL else f'0-{n - 1}'


def e4_suite(n):
    """E4 controls at one CU allocation: (name, args). Footprints >= 4 GiB are far above the 256 MiB MALL."""
    g4 = str(4 << 30)
    rows = [('seqread4g', ['--mode', 'read', '--bytes', g4, '--reps', '5'])]
    rows += [(f'ilp{k}', ['--mode', 'read_ilp', '--ilp', str(k), '--bytes', g4, '--reps', '5']) for k in (1, 2, 4, 8, 16)]
    rows += [(f'grid{b}', ['--mode', 'read', '--bytes', g4, '--reps', '5', '--blocks', str(b)]) for b in (304, 1216, 9728)]
    rows += [(f'stride{k}', ['--mode', 'strided', '--stride', str(k), '--bytes', g4, '--reps', '5']) for k in (4, 16)]
    rows += [('gather1g', ['--mode', 'gather', '--bytes', str(1 << 30), '--reps', '5'])]
    rows += [(f'chase_c{c}', ['--mode', 'chase', '--chains', str(c), '--steps', '20000', '--bytes', str(1 << 30),
                              '--trials', '5']) for c in (1, 304, 4864, 19456)]
    rows += [(f'reuse{m}m', ['--mode', 'read', '--bytes', str(m << 20), '--passes', str(p), '--reps', '5'])
             for m, p in ((8, 256), (32, 64), (128, 16), (512, 4))]
    rows += [('mfma', ['--mode', 'mfma', '--iters', '4096']), ('fma', ['--mode', 'fma', '--iters', '20000', '--reps', '5'])]
    return [(f'n{n}_{name}', args) for name, args in rows]


def placement_core(plan, stage, lanes=None):
    """Per-GPU placement proof for every decoder level and the competitor partition (first steps of each lane)."""
    for lane in lanes or [f'L{i}' for i in range(8)]:
        for n in LEVELS:
            plan.native(lane, stage, 'e1_placement', f'{lane}_n{n}', ['--mode', 'placement', '--blocks', '2432',
                        '--trials', '5', '--expect-cus', str(n)], mask_for(n), level=n)
        plan.native(lane, stage, 'e1_placement', f'{lane}_competitor_232-303', ['--mode', 'placement', '--blocks', '2432',
                    '--trials', '5', '--expect-cus', '72'], COMPETITOR_MASK, level='competitor')


def placement_negative(plan, lane, stage):
    """Historical observations to revalidate (recorded outcomes either way): comma lists were rejected; a mask
    naming another GPU's index did not restrict this GPU."""
    plan.native(lane, stage, 'e1_placement', 'comma_list_0-7,16-23', ['--mode', 'placement', '--blocks', '2432',
                '--trials', '3', '--expect-cus', '16'], '0-7,16-23', expect_failure=True)
    plan.native(lane, stage, 'e1_placement', 'wrong_index_mask', ['--mode', 'placement', '--blocks', '2432', '--trials', '3',
                '--expect-cus', '304'], None, expect_failure=True)['mask_other_index'] = '0-95'


def caps(plan, lane, stage):
    plan.shell(lane, stage, 'e0_capability', 'host_rocprofv3_list', 'rocprofv3 --version; rocprofv3 -L')
    plan.shell(lane, stage, 'e0_capability', 'container_rocprofv3_list',
               'AMD_ENTRY=rocprofv3 scripts/amd/vllm_in_container.sh --version; AMD_ENTRY=rocprofv3 scripts/amd/vllm_in_container.sh -L')
    plan.shell(lane, stage, 'e0_capability', 'amd_smi_static', 'amd-smi static -g $HIP_VISIBLE_DEVICES; amd-smi metric -g $HIP_VISIBLE_DEVICES',
               est=30)
    plan.shell(lane, stage, 'e5_counter_probe', 'native_counter_validation',
               '$PY scripts/amd/cubw/counter_probe.py --out-dir "$CUBW_STEP_DIR"', est=400, timeout=1500)


def build_pilot(cohort, reduced=False):
    p = Plan('pilot', cohort)
    st = 'pilot'
    placement_core(p, st)
    caps(p, 'L0', st)
    placement_negative(p, 'L0', st)
    for n in LEVELS:
        for name, args in e4_suite(n):
            p.native('L0', st, 'e4_pilot', name, args, mask_for(n), level=n)
    for name, args in [('comp_seqread4g', ['--mode', 'read', '--bytes', str(4 << 30), '--reps', '5']),
                       ('comp_read1g', ['--mode', 'read', '--bytes', str(1 << 30), '--reps', '10']),
                       ('comp_fma', ['--mode', 'fma', '--iters', '2000', '--reps', '5'])]:
        p.native('L0', st, 'e4_pilot', name, args, COMPETITOR_MASK, level='competitor')
    # empty-XCD mask last: historically unsafe; a hang here can only cost the end of this lane
    p.native('L0', st, 'e1_placement', 'empty_xcd_0-6', ['--mode', 'placement', '--blocks', '2432', '--trials', '3',
             '--expect-cus', '7'], '0-6', expect_failure=True, timeout=90)
    lanes = {'L1': [], 'L2': [], 'L3': [], 'L4': [], 'L5': [], 'L6': [], 'L7': []}
    rng = random.Random(f'{SEED}:pilot:{cohort}')
    # E1 smoke then primary: bridge then Qwen32B over all E2 levels (one pilot acquisition each)
    order = LEVELS[:]; rng.shuffle(order)
    if not reduced:
        for n in order:
            p.vllm('L1', st, 'e1_e2_pilot', 'bridge', n)
    order = LEVELS[:]; rng.shuffle(order)
    for n in order:
        p.vllm('L1', st, 'e1_e2_pilot', 'qwen32b', n)
    # E3 lifecycle and partition pilots: n=232 with absent/inactive/memory/arithmetic, alone references
    p.competitor_only('L2', st, 'e1_e3_pilot', 'alone_fma_232-303', 'arith100', 30)
    p.competitor_only('L2', st, 'e1_e3_pilot', 'alone_read100_232-303', 'mem100', 30)
    p.competitor_only('L2', st, 'e1_e3_pilot', 'alone_idle_232-303', 'inactive', 10)
    tr = ['absent', 'inactive', 'mem50', 'mem100', 'arith100'] if not reduced else ['absent', 'inactive', 'mem100', 'arith100']
    rng.shuffle(tr)
    for t in tr:
        p.vllm('L2', st, 'e1_e3_pilot', 'qwen32b', 232, t)
    tr = ['absent', 'mem50', 'mem100', 'arith100'] if not reduced else ['absent', 'mem100', 'arith100']
    rng.shuffle(tr)
    for t in tr:
        p.vllm('L3', st, 'e1_e3_pilot', 'qwen32b', 152, t)
    # graph-mask leak probes: decoder on 0..n-1, FMA probe on n..303; compare with the probe alone
    for n in (152, 96):
        p.competitor_only('L3', st, 'e1_leak', f'alone_fma_{n}-303', 'leakprobe', 30, comp_mask=f'{n}-303')
        p.vllm('L3', st, 'e1_leak', 'qwen32b', n, 'leakprobe', comp_mask=f'{n}-303')
    order = LEVELS[:]; rng.shuffle(order)
    for n in order:
        p.vllm('L4', st, 'e1_e2_pilot', 'qwen14b', n)
    # instrumentation distortion: ordinary vs kernel-trace timeline at 304 and 152; graph-mode counters
    for n in (304, 152):
        p.vllm('L5', st, 'e1_timeline', 'qwen32b', n)
        p.vllm('L5', st, 'e1_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace')
    p.vllm('L5', st, 'e5_pilot', 'qwen32b', 304, profile={'kind': 'pmc', 'counters': ['FETCH_SIZE']}, tag='__pmc_FETCH_SIZE')
    p.vllm('L7', st, 'e5_pilot', 'qwen32b', 304, profile={'kind': 'pmc', 'counters': ['FETCH_SIZE'],
           'exclude_regex': '^(_|triton)'}, tag='__pmc_FETCH_SIZE_excl')
    # E8 guardrail pilots
    for n in (304, 152):
        p.vllm('L6', st, 'e8_pilot', 'qwen7b_b8', n)
        p.vllm('L6', st, 'e8_pilot', 'qwen32b', n, tokens=2, tag='__prefill')
    p.vllm('L6', st, 'e8_pilot', 'qwen32b_c16k', 304)
    if not reduced:
        p.vllm('L6', st, 'e8_pilot', 'qwen32b', 304, mode='energy', config='configs_energy/qwen32b_graph_b1_c4096_t10.json',
               tag='__energy')
    # second pilot block on another GPU (GPU-to-GPU spread and clocks)
    order = LEVELS[:]; rng.shuffle(order)
    for n in order:
        p.vllm('L7', st, 'e1_e2_pilot', 'qwen32b', n, tag='__gpu7')
    return p


def production(p, st, models_blocks, boundary, energy_levels, e3_full_requires='release_e3_full',
               e3_min_requires='release_e3_minimal'):
    """Randomized complete blocks: lane i holds block i of one model (E2 levels + E3 cells), then E4/E5/E8."""
    lane_ids = [f'L{i}' for i in range(8)]
    if p.kind == 'production':  # the auto kind already proved placement in its pilot stage
        placement_core(p, st)
    li = 0
    blocks = []
    for model, nblocks in models_blocks:
        for b in range(nblocks):
            blocks.append((lane_ids[li], model, b)); li += 1
    for lane, model, b in blocks:
        rng = random.Random(f'{SEED}:{p.cohort}:{model}:block{b}')
        cells = [('e2', n, 'absent', None) for n in LEVELS]
        for n in E3_LEVELS:
            for t in E3_TREATMENTS:
                minimal = model == E3_MINIMAL['model'] and n in E3_MINIMAL['levels'] and t in E3_MINIMAL['treatments']
                cells.append(('e3', n, t, e3_min_requires if minimal else e3_full_requires))
        if b == 0 and model == 'qwen32b':
            cells.append(('e3', 232, 'inactive', 'release_e3_minimal'))  # named validation control, not a grid cell
        rng.shuffle(cells)
        for exp, n, t, req in cells:
            p.vllm(lane, st, exp, model, n, t, acq=b, block=b, requires=req if exp == 'e3' else 'release_e2')
    # after the blocks: E4 (3 acquisitions, one per lane triple), E5 and E8 spread over all lanes
    tail = []
    for acq in range(3):
        for n in LEVELS + ['competitor']:
            suite = e4_suite(n) if n != 'competitor' else [
                ('comp_seqread4g', ['--mode', 'read', '--bytes', str(4 << 30), '--reps', '5']),
                ('comp_read1g', ['--mode', 'read', '--bytes', str(1 << 30), '--reps', '10']),
                ('comp_fma', ['--mode', 'fma', '--iters', '2000', '--reps', '5'])]
            tail.append(('e4', acq, n, suite))
    lane_cycle = 0
    for exp, acq, n, suite in tail:
        lane = lane_ids[lane_cycle % 8]; lane_cycle += 1
        rng = random.Random(f'{SEED}:{p.cohort}:e4:{acq}:{n}')
        suite = suite[:]; rng.shuffle(suite)
        for name, args in suite:
            p.native(lane, st, 'e4', name, args, COMPETITOR_MASK if n == 'competitor' else mask_for(n), acq=acq,
                     level=n)
    # E5: graph-mode traffic for baseline and boundary cells (one counter group per run) plus a timeline
    e5_lanes = iter(lane_ids * 4)
    for n in sorted({304, boundary}):
        for counters in (['FETCH_SIZE'], ['WRITE_SIZE'], ['TCC_HIT', 'TCC_MISS']):
            p.vllm(next(e5_lanes), st, 'e5', 'qwen32b', n, profile={'kind': 'pmc', 'counters': counters,
                   'exclude_regex': '^(_|triton)'}, tag='__pmc_' + '+'.join(counters), requires='release_e5e8')
        p.vllm(next(e5_lanes), st, 'e5', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace',
               requires='release_e5e8')
    # E8: guardrails at baseline/boundary/96 and matched energy at baseline and boundary
    g = iter(lane_ids[::-1] * 6)
    for acq in range(3):
        for n in sorted({304, boundary, 96}):
            p.vllm(next(g), st, 'e8', 'qwen7b_b8', n, acq=acq, requires='release_e5e8')
            p.vllm(next(g), st, 'e8', 'qwen32b', n, acq=acq, tokens=2, tag='__prefill', requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'qwen32b_c16k', 304, acq=acq, requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'qwen32b_c16k', boundary, acq=acq, requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'mistral7b', 304, acq=acq, requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'mistral7b', boundary, acq=acq, requires='release_e5e8')
        for n in energy_levels:
            p.vllm(next(g), st, 'e8', 'qwen32b', n, acq=acq, mode='energy',
                   config='configs_energy/qwen32b_graph_b1_c4096_t10.json', tag='__energy', requires='release_e5e8')


def finalize(p, src, args):
    lanes = {f'L{i}': {'gpu_index': i} for i in range(8)}
    budget, stages = {}, {}
    for s in p.steps:
        b = budget.setdefault(s['lane'], {'steps': 0, 'est_s': 0})
        b['steps'] += 1; b['est_s'] += s['est_s']
        stages.setdefault(s['stage'], {}).setdefault(s['lane'], 0)
        stages[s['stage']][s['lane']] += s['est_s']
    longest = sum(max(v.values()) for v in stages.values())  # stages run one after another
    rates = {'mi300x': 8.0, 'mi325x': 12.8}  # allocation units per node-hour observed on this account
    plan = {
        'version': 'cubw_v1', 'kind': p.kind, 'cohort': args.cohort, 'seed': SEED,
        'assignment': 'docs/amd/compute_bandwidth_followup_20261026.md',
        'full_cus': FULL, 'competitor_mask': COMPETITOR_MASK, 'levels': LEVELS, 'e3_levels': E3_LEVELS,
        'e3_treatments': E3_TREATMENTS, 'e3_minimal_subset': E3_MINIMAL, 'boundary_level': args.boundary,
        'lanes': lanes, 'derived_configs': derived_configs(src), 'config_source': str(src),
        'statistics': {'warmups': 3, 'trials_per_acquisition': 10, 'acquisitions': 'one per block/lane; >=3 per cell',
                       'bootstrap': '10,000 draws seed 20261018, blocks resampled when cells are blocked, trials nested',
                       'noninferiority': 'upper 95% CI of R(n)=T(n)/T(304) <= 1.05 (latency)'},
        'gates': {
            'placement_ok': 'every decoder/competitor mask: distinct physical CUs == requested, per-XCD counts differ by <= 1',
            'lifecycle_ok': 'every competitor: started event, stopped event, PID gone, no owned residual GPU process',
            'outputs_ok': 'Qwen32B pilot runs complete with one output digest across all CU levels and treatments',
            'leak_ok': 'probe TFLOP/s during decoder trials >= 0.85 x probe alone on the same mask',
            'clock_ok': 'pooled in-window gfx and memory clocks over Qwen32B E2 pilot trials: max/min <= 1.05, coverage >= 0.5',
            'release_e2': 'placement_ok and outputs_ok (observational label if clock_ok fails)',
            'release_e3_full': 'placement_ok and lifecycle_ok and outputs_ok and leak_ok and clock_ok',
            'release_e3_minimal': 'placement_ok and lifecycle_ok and outputs_ok and leak_ok',
            'release_e5e8': 'outputs_ok'},
        'retry_policy': 'two attempts maximum, only for execution/tool failures; correctness failures never retried',
        'timeouts': 'per step: max(900 s, 3 x estimate) for decoder steps; native 120 s; competitor alone seconds+120',
        'reserve_s': 1800,
        'stages': list(stages),
        'budget': {'per_lane': budget, 'per_stage_longest_lane_s': {k: max(v.values()) for k, v in stages.items()},
                   'longest_lane_s': longest, 'node_hours_est': round(longest / 3600 + 0.25, 2),
                   'allocation_units_est': round((longest / 3600 + 0.25) * rates[args.cohort], 1),
                   'rate_units_per_node_hour': rates[args.cohort]},
        'steps': p.steps,
    }
    return plan


def main(args):
    src = Path(args.config_source)
    if args.kind == 'pilot':
        p = build_pilot(args.cohort)
    elif args.kind == 'production':
        p = Plan('production', args.cohort)
        production(p, 'production', [('qwen32b', 5), ('qwen14b', 3)], args.boundary, sorted({304, args.boundary}))
    else:  # auto: reduced pilot, gate, then a budget-limited production stage
        p = build_pilot(args.cohort, reduced=True)
        p.kind = 'auto'
        production(p, 'production', [('qwen32b', 3), ('qwen14b', 3)], args.boundary, [304])
    plan = finalize(p, src, args)
    text = json.dumps(plan, indent=1) + '\n'
    plan['plan_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    args.out.write_text(json.dumps(plan, indent=1) + '\n')
    b = plan['budget']
    print(f"{args.kind}/{args.cohort}: {len(plan['steps'])} steps; longest lane {b['longest_lane_s'] / 3600:.2f} h; "
          f"~{b['node_hours_est']} node-h ~{b['allocation_units_est']} units")
    for lane, v in sorted(b['per_lane'].items()):
        print(f"  {lane}: {v['steps']} steps, {v['est_s'] / 60:.0f} min")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--kind', choices=['pilot', 'production', 'auto'], required=True)
    q.add_argument('--cohort', choices=['mi300x', 'mi325x'], required=True)
    q.add_argument('--boundary', type=int, default=152, help='E2 boundary level used by E5/E8 (set after the pilot)')
    q.add_argument('--config-source', default='/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/amd_20261018_mi300x')
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
