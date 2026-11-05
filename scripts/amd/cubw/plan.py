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
  plan.py --kind auto --pilot minimal --cohort mi300x --boundary 200 --out plan.json

v1.1 (after the MI300X pilot, job 481167): graph-mode rocprofv3 counter collection is not planned (both
pilot attempts failed: illegal memory access, then a hang to the 1200 s timeout, leaving PPID=1 rocprofv3
processes); kernel-trace timelines stay. Leak probes gain placement-proven native co-runner references
(leak_suite) and E3 release uses leak_ok_v11 (scripts/amd/cubw/analyze.py). A "minimal" pilot stage
re-proves only placement, lifecycle (fixed residual-process check and orphan sweep), leak references and
Qwen32B outputs/clocks before the production stage of the same allocation.

v1.4 MI350X campaign (--kind auto --campaign --cohort mi350x): one allocation at MI350X levels (256 CUs; see HW and
build_campaign): pilot (placement at every level, capability audit, leak v1.1 and v2, E3 lifecycle, E2 pilot,
instrumentation distortion), gate, then production (E2 Qwen32B x8 blocks at nominal+fine levels, Qwen14B x3, E3-minimal
gated by leak_ok_v2, sized-kernel decode, E4 x3, kernel-trace timelines, E6 shapes and counters, E8 guardrails/energy).

v1.5 MI350X follow-up (--kind auto --followup --cohort mi350x; --kind pilot --followup --smoke on mi3501x first),
pre-registered in docs/amd/mi350x_followup_plan_20261103.md:
  pilot         placement; leak test v3 (svc_bench_v5 duration probe beside native read/fma co-runners on 0..n-1,
                beside positive controls fma, read and the v4 full-LDS probe on 0..n+7, and beside the decoder at n)
                on two GPUs; the dispatch probe (wvSplitK-shaped launches) at every fine level; E3 lifecycle; E2 pilot
  gate          analyze.py gate: leak_ok_v3, release_e3_minimal_v3
  production    E2 at the fine levels and 'e6_balanced' decode (wvSplitK sized for the largest shader-engine-balanced
                count <= n) on 8 blocks; e6_decode at the balanced levels; E3-minimal on 5 blocks if
                release_e3_minimal_v3; kernel traces with the balanced size; E6 shapes with --cu-counts x3

v1.6 follow-ups, pre-registered in docs/amd/followups_v16_plan_20261104.md:
  --leak4 (MI350X)   leak test v4 (svc_bench_v6 start-delay probe on n..255 alone; beside native read/fma on 0..n-1;
                     beside positive controls on 0..n+7: the v4 full-LDS probe and the decoder itself (required), fma and
                     read (recorded); beside the decoder at n) on every GPU, gate leak_ok_v4, then E3-minimal on 8 blocks
                     (release_e3_minimal_v4) with E2 at 256 in each block
  --se-study (MI300X) shader-engine balance of wvSplitK on gfx942: placement and the dispatch probe at SE_LEVELS, E2 pilot,
                     then on 8 blocks E2 at every level, 'e6_balanced' (size balanced(n)) at the unbalanced levels including
                     the full device (304 -> 288) and 'e6_decode' at the balanced masks; kernel traces; shapes x3
  --smoke with either: the one-GPU version (mi3501x / mi3001x), pilot-labelled, releases nothing

v1.3 extension (--kind auto --extension, after the production jobs 487341/489594), one allocation per cohort:
  pilot stage   placement at the fine levels; leak test v2 (svc_bench_v4 per-CU probe on n..303 alone, beside
                native read/fma co-runners on 0..n-1, beside a positive-control probe overlapping 8 probe CUs,
                and beside the decoder at n), on two GPUs (MI300X) or one (MI325X cross-check)
  gate          analyze.py gate: leak_ok_v2, release_e3_minimal_v2
  production    E2-fine (Qwen32B, 8 blocks, levels 304/272/248/232/216/200/184/152); E6 decode with wvSplitK sized
                for the active CUs (CUBW_CU_COUNT_OVERRIDE, 5 blocks); kernel-trace timelines at every fine level
                (+ the override at 232/200); E6 kernel-shape controls x3 per level; pre-MALL counters on the shape
                controls and on eager decode; MI300X only: E3-minimal if release_e3_minimal_v2
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
FINE = [304, 272, 248, 232, 216, 200, 184, 152]      # extension: per-XCD 38/34/31/29/27/25/23/19
OVERRIDE_LEVELS = [272, 248, 232, 216, 200, 184, 152]
LEAK_LEVELS = (232, 152)                     # leak suites: E3 partition top and the half-device level
COUNTER_LEVELS = (304, 232, 200, 152)
BOUNDARY = 200
# v1.4 (MI350X port): per-cohort hardware. MI350X levels keep MI300X's per-XCD fractions on 8 XCDs x 32 CUs
# (24/32 ~ 29/38, 21/32 ~ 25/38, 16/32 = 19/38, 10/32 ~ 12/38); the competitor holds the top 8 CUs per XCD.
HW = {
    'mi300x': {'full': 304, 'arch': 'gfx942', 'gpu_name': 'MI300X', 'partition': 'mi3008x', 'rate': 8.0, 'se_per_xcd': 4},
    'mi325x': {'full': 304, 'arch': 'gfx942', 'gpu_name': 'MI325X', 'partition': 'mi3258x', 'rate': 12.8, 'se_per_xcd': 4},
    'mi350x': {'full': 256, 'arch': 'gfx950', 'gpu_name': 'MI350X', 'partition': 'mi3508x', 'rate': 11.2,
               'levels': [256, 192, 168, 128, 80], 'fine': [256, 232, 208, 192, 184, 168, 152, 128],
               'counter_levels': (256, 192, 168, 128), 'boundary': 168, 'se_per_xcd': 4},
}
XCDS = 8
# v1.6: MI300X shader-engine study levels. Contiguous masks give 4 SEs per XCD (job 492956 placement keys): 304 is
# 9,10,10,9 per SE; 288/256/224/192/128 are balanced; 272/232/200/152 are not.
SE_LEVELS = [304, 288, 272, 256, 232, 224, 200, 192, 152, 128]
SINGLE_GPU = {'mi300x': 'mi3001x', 'mi350x': 'mi3501x'}


def configure(cohort):
    """Set the module's level constants for one cohort (MI300X/MI325X values are the v1.3 literals)."""
    global FULL, LEVELS, E3_LEVELS, E3_MINIMAL, COMPETITOR_MASK, FINE, OVERRIDE_LEVELS, LEAK_LEVELS, COUNTER_LEVELS, BOUNDARY
    h = HW[cohort]
    if 'levels' not in h:
        return
    FULL, LEVELS, FINE = h['full'], h['levels'], h['fine']
    E3_LEVELS = LEVELS[1:]
    E3_MINIMAL = dict(E3_MINIMAL, levels=[LEVELS[1], FULL // 2])
    COMPETITOR_MASK = f'{LEVELS[1]}-{FULL - 1}'
    OVERRIDE_LEVELS = FINE[1:]
    LEAK_LEVELS = (LEVELS[1], FULL // 2)
    COUNTER_LEVELS = h['counter_levels']
    BOUNDARY = h['boundary']


def balanced(n, cohort='mi350x'):
    """v1.5: the largest CU count <= n that gives every shader engine of every XCD the same number of CUs under a
    contiguous mask (masks interleave over XCDs, then over the shader engines of each XCD; jobs 496074/492956)."""
    step = XCDS * HW[cohort].get('se_per_xcd', 4)
    return n - n % step


EXT_COUNTERS = ['FETCH_SIZE', 'TCC_EA0_RDREQ', 'TCC_EA0_RDREQ_LEVEL', 'GRBM_GUI_ACTIVE']


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
    eager = json.loads((src / 'configs_profile/qwen32b_eager_b1_c4096.json').read_text())
    eager = dict(eager, stage='cubw_v13_derived', derived_from='configs_profile/qwen32b_eager_b1_c4096.json',
                 purpose='extension: pre-MALL counters on eager decode (graph-mode --pmc unavailable)')
    out = {}
    for name, cfg in {'configs/qwen32b_eager_b1_c4096_profile.json': eager,'configs/qwen32b_graph_b1_c16384.json': c16, 'configs_energy/qwen32b_graph_b1_c4096_t10.json': en10}.items():
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
             profile=None, comp_mask=None, requires=None, config_dir='configs', config=None, tag='', extra_env=None):
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
        if extra_env:
            s['extra_env'] = extra_env
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

    def shell(self, lane, stage, exp, name, cmd, est=60, timeout=300, requires=None, **extra):
        s = self.add(lane, stage, exp, kind='shell', id=f'{exp}:{name}', cell=name, cmd=cmd,
                     out=f'{exp}/{name}', est_s=est, timeout_s=timeout, no_retry=True, **extra)
        if requires:
            s['requires'] = requires
        return s

    def competitor_only(self, lane, stage, exp, name, treatment, seconds, comp_mask=None, acq=0, corunner=None):
        s = self.add(lane, stage, exp, kind='competitor_only', id=f'{exp}:{name}:a{acq}', cell=name, acquisition=acq,
                     treatment=treatment, seconds=seconds, out=f'{exp}/{name}/a{acq}', est_s=seconds + 20,
                     timeout_s=seconds + 120, no_retry=True)
        if comp_mask:
            s['competitor_mask'] = comp_mask
        if corunner:
            s['corunner'] = corunner
        return s


def mask_for(n):
    return None if n >= FULL else f'0-{n - 1}'


def e4_suite(n):
    """E4 controls at one CU allocation: (name, args). Footprints >= 4 GiB are far above the 256 MiB MALL."""
    g4 = str(4 << 30)
    rows = [('seqread4g', ['--mode', 'read', '--bytes', g4, '--reps', '5'])]
    rows += [(f'ilp{k}', ['--mode', 'read_ilp', '--ilp', str(k), '--bytes', g4, '--reps', '5']) for k in (1, 2, 4, 8, 16)]
    rows += [(f'grid{b}', ['--mode', 'read', '--bytes', g4, '--reps', '5', '--blocks', str(b)]) for b in (FULL, 4 * FULL, 32 * FULL)]
    rows += [(f'stride{k}', ['--mode', 'strided', '--stride', str(k), '--bytes', g4, '--reps', '5']) for k in (4, 16)]
    rows += [('gather1g', ['--mode', 'gather', '--bytes', str(1 << 30), '--reps', '5'])]
    rows += [(f'chase_c{c}', ['--mode', 'chase', '--chains', str(c), '--steps', '20000', '--bytes', str(1 << 30),
                              '--trials', '5']) for c in (1, FULL, 16 * FULL, 64 * FULL)]
    rows += [(f'reuse{m}m', ['--mode', 'read', '--bytes', str(m << 20), '--passes', str(p), '--reps', '5'])
             for m, p in ((8, 256), (32, 64), (128, 16), (512, 4))]
    rows += [('mfma', ['--mode', 'mfma', '--iters', '4096']), ('fma', ['--mode', 'fma', '--iters', '20000', '--reps', '5'])]
    return [(f'n{n}_{name}', args) for name, args in rows]


def placement_core(plan, stage, lanes=None, levels=None):
    """Per-GPU placement proof for every decoder level and the competitor partition (first steps of each lane)."""
    for lane in lanes or [f'L{i}' for i in range(8)]:
        for n in levels or LEVELS:
            plan.native(lane, stage, 'e1_placement', f'{lane}_n{n}', ['--mode', 'placement', '--blocks', str(8 * FULL),
                        '--trials', '5', '--expect-cus', str(n)], mask_for(n), level=n)
        lo = int(COMPETITOR_MASK.split('-')[0])
        plan.native(lane, stage, 'e1_placement', f'{lane}_competitor_{COMPETITOR_MASK}', ['--mode', 'placement', '--blocks',
                    str(8 * FULL), '--trials', '5', '--expect-cus', str(FULL - lo)], COMPETITOR_MASK, level='competitor')


def placement_negative(plan, lane, stage):
    """Historical observations to revalidate (recorded outcomes either way): comma lists were rejected; a mask
    naming another GPU's index did not restrict this GPU."""
    plan.native(lane, stage, 'e1_placement', 'comma_list_0-7,16-23', ['--mode', 'placement', '--blocks', str(8 * FULL),
                '--trials', '3', '--expect-cus', '16'], '0-7,16-23', expect_failure=True)
    plan.native(lane, stage, 'e1_placement', 'wrong_index_mask', ['--mode', 'placement', '--blocks', str(8 * FULL), '--trials',
                '3', '--expect-cus', str(FULL)], None, expect_failure=True)['mask_other_index'] = '0-95'


def caps(plan, lane, stage):
    plan.shell(lane, stage, 'e0_capability', 'host_rocprofv3_list', 'rocprofv3 --version; rocprofv3 -L')
    plan.shell(lane, stage, 'e0_capability', 'container_rocprofv3_list',
               'AMD_ENTRY=rocprofv3 scripts/amd/vllm_in_container.sh --version; AMD_ENTRY=rocprofv3 scripts/amd/vllm_in_container.sh -L')
    plan.shell(lane, stage, 'e0_capability', 'amd_smi_static', 'amd-smi static -g $HIP_VISIBLE_DEVICES; amd-smi metric -g $HIP_VISIBLE_DEVICES',
               est=30)
    plan.shell(lane, stage, 'e5_counter_probe', 'native_counter_validation',
               '$PY scripts/amd/cubw/counter_probe.py --out-dir "$CUBW_STEP_DIR"', est=400, timeout=1500)


def leak_suite(plan, lane, stage, rng):
    """FMA probe on n..303: alone, beside placement-proven native co-runners (fma, read) on 0..n-1, and beside the
    decoder on 0..n-1 (n=232 is the E3 competitor partition). Names carry the lane: refs pair within a GPU."""
    for n, treatment in ((LEAK_LEVELS[0], 'arith100'), (LEAK_LEVELS[1], 'leakprobe')):
        probe = f'{n}-{FULL - 1}'
        steps = [lambda: plan.competitor_only(lane, stage, 'e1_leak', f'alone_fma_{probe}_{lane}', 'leakprobe', 30,
                                              comp_mask=probe)]
        for co in ('arith100', 'mem100'):
            steps.append(lambda co=co: plan.competitor_only(lane, stage, 'e1_leak', f'corun_{co}_0-{n - 1}_fma_{probe}_{lane}',
                                                            'leakprobe', 30, comp_mask=probe,
                                                            corunner={'treatment': co, 'mask': f'0-{n - 1}'}))
        steps.append(lambda: plan.vllm(lane, stage, 'e1_leak', 'qwen32b', n, treatment, comp_mask=probe, tag=f'_{lane}'))
        rng.shuffle(steps)
        for f in steps:
            f()


def build_minimal_pilot(cohort):
    """v1.1 re-pilot: placement on every GPU, lifecycle, leak references on two GPUs, Qwen32B outputs/clocks."""
    p = Plan('pilot', cohort)
    st = 'pilot'
    placement_core(p, st)
    rng = random.Random(f'{SEED}:minimal_pilot:{cohort}')
    leak_suite(p, 'L0', st, rng)
    leak_suite(p, 'L1', st, rng)
    p.competitor_only('L2', st, 'e1_e3_pilot', f'alone_fma_{COMPETITOR_MASK}', 'arith100', 30)
    p.competitor_only('L2', st, 'e1_e3_pilot', f'alone_read100_{COMPETITOR_MASK}', 'mem100', 30)
    p.competitor_only('L2', st, 'e1_e3_pilot', f'alone_idle_{COMPETITOR_MASK}', 'inactive', 10)
    tr = ['absent', 'inactive', 'mem100', 'arith100']; rng.shuffle(tr)
    for t in tr:
        p.vllm('L2', st, 'e1_e3_pilot', 'qwen32b', E3_LEVELS[0], t)
    for lane in ('L3', 'L4'):
        order = LEVELS[:]; rng.shuffle(order)
        for n in order:
            p.vllm(lane, st, 'e1_e2_pilot', 'qwen32b', n, tag=f'__{lane}')
    return p


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
    p.native('L0', st, 'e1_placement', 'empty_xcd_0-6', ['--mode', 'placement', '--blocks', str(8 * FULL), '--trials', '3',
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
    p.competitor_only('L2', st, 'e1_e3_pilot', f'alone_fma_{COMPETITOR_MASK}', 'arith100', 30)
    p.competitor_only('L2', st, 'e1_e3_pilot', f'alone_read100_{COMPETITOR_MASK}', 'mem100', 30)
    p.competitor_only('L2', st, 'e1_e3_pilot', f'alone_idle_{COMPETITOR_MASK}', 'inactive', 10)
    tr = ['absent', 'inactive', 'mem50', 'mem100', 'arith100'] if not reduced else ['absent', 'inactive', 'mem100', 'arith100']
    rng.shuffle(tr)
    for t in tr:
        p.vllm('L2', st, 'e1_e3_pilot', 'qwen32b', E3_LEVELS[0], t)
    tr = ['absent', 'mem50', 'mem100', 'arith100'] if not reduced else ['absent', 'mem100', 'arith100']
    rng.shuffle(tr)
    for t in tr:
        p.vllm('L3', st, 'e1_e3_pilot', 'qwen32b', FULL // 2, t)
    # graph-mask leak probes with native co-runner references (v1.1)
    leak_suite(p, 'L3', st, rng)
    order = LEVELS[:]; rng.shuffle(order)
    for n in order:
        p.vllm('L4', st, 'e1_e2_pilot', 'qwen14b', n)
    # instrumentation distortion: ordinary vs kernel-trace timeline at 304 and 152; graph-mode counters
    for n in (FULL, FULL // 2):
        p.vllm('L5', st, 'e1_timeline', 'qwen32b', n)
        p.vllm('L5', st, 'e1_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace')
    # E8 guardrail pilots
    for n in (FULL, FULL // 2):
        p.vllm('L6', st, 'e8_pilot', 'qwen7b_b8', n)
        p.vllm('L6', st, 'e8_pilot', 'qwen32b', n, tokens=2, tag='__prefill')
    p.vllm('L6', st, 'e8_pilot', 'qwen32b_c16k', FULL)
    if not reduced:
        p.vllm('L6', st, 'e8_pilot', 'qwen32b', FULL, mode='energy', config='configs_energy/qwen32b_graph_b1_c4096_t10.json',
               tag='__energy')
    # second pilot block on another GPU (GPU-to-GPU spread and clocks)
    order = LEVELS[:]; rng.shuffle(order)
    for n in order:
        p.vllm('L7', st, 'e1_e2_pilot', 'qwen32b', n, tag='__gpu7')
    return p


def production(p, st, models_blocks, boundary, energy_levels, e3_full_requires='release_e3_full',
               e3_min_requires='release_e3_minimal', e4=True):
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
            cells.append(('e3', E3_LEVELS[0], 'inactive', e3_min_requires))  # named validation control, not a grid cell
        rng.shuffle(cells)
        for exp, n, t, req in cells:
            p.vllm(lane, st, exp, model, n, t, acq=b, block=b, requires=req if exp == 'e3' else 'release_e2')
    # after the blocks: E4 (3 acquisitions, one per lane triple), E5 and E8 spread over all lanes
    tail = []
    for acq in range(3 if e4 else 0):  # e4=False: the three E4 acquisitions already exist from an earlier job
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
    # E5: kernel-trace timelines for baseline and boundary cells (graph-mode counters unavailable, v1.1)
    e5_lanes = iter(lane_ids * 4)
    for n in sorted({FULL, boundary}):
        p.vllm(next(e5_lanes), st, 'e5', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace',
               requires='release_e5e8')
    # E8: guardrails at baseline/boundary/96 and matched energy at baseline and boundary
    g = iter(lane_ids[::-1] * 6)
    for acq in range(3):
        for n in sorted({FULL, boundary, LEVELS[-1]}):
            p.vllm(next(g), st, 'e8', 'qwen7b_b8', n, acq=acq, requires='release_e5e8')
            p.vllm(next(g), st, 'e8', 'qwen32b', n, acq=acq, tokens=2, tag='__prefill', requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'qwen32b_c16k', FULL, acq=acq, requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'qwen32b_c16k', boundary, acq=acq, requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'mistral7b', FULL, acq=acq, requires='release_e5e8')
        p.vllm(next(g), st, 'e8', 'mistral7b', boundary, acq=acq, requires='release_e5e8')
        for n in energy_levels:
            p.vllm(next(g), st, 'e8', 'qwen32b', n, acq=acq, mode='energy',
                   config='configs_energy/qwen32b_graph_b1_c4096_t10.json', tag='__energy', requires='release_e5e8')


def leak_suite_v2(plan, lane, stage, rng):
    """Per-CU probe (one full-LDS workgroup per CU) on n..303: alone, beside native read/fma co-runners on 0..n-1,
    beside a positive-control probe on 0..n+7 (8 shared CUs: the test must see them), beside the decoder at n."""
    for n in LEAK_LEVELS:
        probe = f'{n}-{FULL - 1}'
        steps = [lambda: plan.competitor_only(lane, stage, 'e1_leak_v2', f'alone_cuprobe_{probe}_{lane}', 'cuprobe', 30,
                                              comp_mask=probe)]
        for co, mask in (('mem100', f'0-{n - 1}'), ('arith100', f'0-{n - 1}'), ('cuprobe', f'0-{n + 7}')):
            steps.append(lambda co=co, mask=mask: plan.competitor_only(
                lane, stage, 'e1_leak_v2', f'corun_{co}_{mask}_cuprobe_{probe}_{lane}', 'cuprobe', 30, comp_mask=probe,
                corunner={'treatment': co, 'mask': mask}))
        steps.append(lambda: plan.vllm(lane, stage, 'e1_leak_v2', 'qwen32b', n, 'cuprobe', comp_mask=probe, tag=f'_{lane}'))
        rng.shuffle(steps)
        for f in steps:
            f()


def e6_cmd(n, out, light=False, counter=None, cu_counts=None):
    env = 'HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES,CUBW_STEP_TAG=$CUBW_STEP_TAG'
    if n < FULL:
        env += f',HSA_CU_MASK=$HIP_VISIBLE_DEVICES:0-{n - 1}'
    args = f'scripts/amd/cubw/e6_shapes.py --n-active {n} --out "$CUBW_STEP_DIR/{out}"' + (' --light' if light else '')
    if cu_counts:
        args += ' --cu-counts ' + ','.join(map(str, cu_counts))
    if counter:
        return (f'AMD_ENTRY=rocprofv3 AMD_EXTRA_ENV="{env}" scripts/amd/vllm_in_container.sh --pmc {counter} '
                f'--kernel-trace --output-format csv -d "$CUBW_STEP_DIR/rocprof" -o p -- python3 {args}')
    return f'AMD_EXTRA_ENV="{env}" scripts/amd/vllm_in_container.sh {args}'


def leak_suite_v3(plan, lane, stage, rng):
    """v1.5 duration probe (cuprobe_t) on n..FULL-1: alone; beside native read/fma co-runners on 0..n-1; beside
    positive controls sharing 8 probe CUs (0..n+7): fma, read and the v4 full-LDS probe (wvSplitK-like residency);
    beside the decoder at n. Names carry the lane: references pair within a GPU."""
    for n in LEAK_LEVELS:
        probe = f'{n}-{FULL - 1}'
        steps = [lambda: plan.competitor_only(lane, stage, 'e1_leak_v3', f'alone_cuprobe_t_{probe}_{lane}', 'cuprobe_t', 30,
                                              comp_mask=probe)]
        for co, mask in (('mem100', f'0-{n - 1}'), ('arith100', f'0-{n - 1}'), ('arith100', f'0-{n + 7}'),
                         ('mem100', f'0-{n + 7}'), ('cuprobe', f'0-{n + 7}')):
            steps.append(lambda co=co, mask=mask: plan.competitor_only(
                lane, stage, 'e1_leak_v3', f'corun_{co}_{mask}_cuprobe_t_{probe}_{lane}', 'cuprobe_t', 30, comp_mask=probe,
                corunner={'treatment': co, 'mask': mask}))
        steps.append(lambda: plan.vllm(lane, stage, 'e1_leak_v3', 'qwen32b', n, 'cuprobe_t', comp_mask=probe, tag=f'_{lane}'))
        rng.shuffle(steps)
        for f in steps:
            f()


def dispatch_suite(plan, lane, stage, levels, tag=''):
    """v1.5 dispatch probe (svc_bench_v5) under each mask: wvSplitK-shaped launches (16 waves, full LDS) with the
    grid sized for the device, for the mask and for the balanced count, plus a no-LDS control at the mask size.
    tag suffixes the cell names; it is required when the suite runs on more than one lane (attempt dirs are per cell)."""
    for n in levels:
        grids = sorted({FULL, n, balanced(n)}, reverse=True)
        for g in grids:
            plan.native(lane, stage, 'e6_dispatch', f'n{n}_g{g}{tag}', ['--mode', 'dispatch', '--blocks', str(g), '--threads', '1024',
                        '--bytes', str(1 << 30), '--trials', '5', '--warmups', '2'], mask_for(n), level=n)['bench'] = 'v5'
        plan.native(lane, stage, 'e6_dispatch', f'n{n}_g{n}_lds0{tag}', ['--mode', 'dispatch', '--blocks', str(n), '--threads',
                    '1024', '--lds', '1', '--bytes', str(1 << 30), '--trials', '5', '--warmups', '2'], mask_for(n),
                    level=n)['bench'] = 'v5'


def leak_suite_v4(plan, lane, stage, rng):
    """v1.6 start-delay probe (gapprobe) on n..FULL-1: alone; beside native read/fma on 0..n-1; beside positive controls
    on 0..n+7 (the v4 full-LDS probe, fma, read) and beside the decoder on 0..n+7; beside the decoder at n."""
    for n in LEAK_LEVELS:
        probe = f'{n}-{FULL - 1}'
        steps = [lambda: plan.competitor_only(lane, stage, 'e1_leak_v4', f'alone_gapprobe_{probe}_{lane}', 'gapprobe', 30,
                                              comp_mask=probe)]
        for co, mask in (('mem100', f'0-{n - 1}'), ('arith100', f'0-{n - 1}'), ('cuprobe', f'0-{n + 7}'),
                         ('arith100', f'0-{n + 7}'), ('mem100', f'0-{n + 7}')):
            steps.append(lambda co=co, mask=mask: plan.competitor_only(
                lane, stage, 'e1_leak_v4', f'corun_{co}_{mask}_gapprobe_{probe}_{lane}', 'gapprobe', 30, comp_mask=probe,
                corunner={'treatment': co, 'mask': mask}))
        steps.append(lambda: plan.vllm(lane, stage, 'e1_leak_v4', 'qwen32b', n, 'gapprobe', comp_mask=probe, tag=f'_{lane}'))
        steps.append(lambda: plan.vllm(lane, stage, 'e1_leak_v4', 'qwen32b', n + 8, 'gapprobe', comp_mask=probe,
                                       tag=f'_positive_{lane}'))
        rng.shuffle(steps)
        for f in steps:
            f()


def build_leak4(cohort, smoke=False):
    """v1.6 MI350X: leak test v4 on every GPU, gate, E3-minimal on 8 blocks. smoke: one GPU, releases nothing."""
    levels = sorted({FULL} | {n for n in LEAK_LEVELS} | {n + 8 for n in LEAK_LEVELS}, reverse=True)
    lanes = ['L0'] if smoke else [f'L{i}' for i in range(8)]
    p = Plan('pilot' if smoke else 'auto', cohort)
    placement_core(p, 'pilot', lanes=lanes, levels=levels)
    for lane in lanes:
        leak_suite_v4(p, lane, 'pilot', random.Random(f'{SEED}:leak4:{cohort}:{lane}'))
    p.levels, p.v16 = levels, True
    if smoke:
        p.n_lanes, p.partition = 1, SINGLE_GPU[cohort]
        return p
    for b, lane in enumerate(lanes):
        rng = random.Random(f'{SEED}:{cohort}:leak4:block{b}')
        cells = [('e2', FULL, 'absent', 'release_e2')]
        cells += [('e3', n, t, 'release_e3_minimal_v4') for n in E3_MINIMAL['levels'] for t in E3_MINIMAL['treatments']]
        if b == 0:
            cells.append(('e3', E3_MINIMAL['levels'][0], 'inactive', 'release_e3_minimal_v4'))
        rng.shuffle(cells)
        for exp, n, t, req in cells:
            p.vllm(lane, 'production', exp, 'qwen32b', n, t, acq=b, block=b, requires=req)
    return p


def build_se_study(cohort, smoke=False):
    """v1.6 MI300X shader-engine balance study (H_SE on gfx942). smoke: one GPU, releases nothing."""
    levels = SE_LEVELS
    unbalanced = [n for n in levels if balanced(n, cohort) != n]
    bal_masks = [n for n in levels if balanced(n, cohort) == n]
    sized = lambda n: [f'CUBW_CU_COUNT_OVERRIDE={balanced(n, cohort)}']
    if smoke:
        p = Plan('pilot', cohort)
        placement_core(p, 'pilot', lanes=['L0'], levels=levels)
        dispatch_suite(p, 'L0', 'pilot', levels)
        for n in (FULL, 272):
            p.vllm('L0', 'pilot', 'e1_e2_pilot', 'qwen32b', n)
            p.vllm('L0', 'pilot', 'e6_balanced', 'qwen32b', n, tag=f'__cu{balanced(n, cohort)}', extra_env=sized(n))
        p.vllm('L0', 'pilot', 'e6_decode', 'qwen32b', 256, tag='__cu256', extra_env=['CUBW_CU_COUNT_OVERRIDE=256'])
        for n in (FULL, 272):
            p.shell('L0', 'pilot', 'e6_shapes', f'n{n}_a0', e6_cmd(n, 'e6.json', cu_counts=[balanced(n, cohort)]),
                    est=180, timeout=900, level=n, acquisition=0)
        p.n_lanes, p.partition, p.levels, p.v16 = 1, SINGLE_GPU[cohort], levels, True
        return p
    p = Plan('auto', cohort)
    lanes = [f'L{i}' for i in range(8)]
    placement_core(p, 'pilot', levels=levels)
    dispatch_suite(p, 'L0', 'pilot', levels, tag='_L0')  # v1.6.1: 499334 collided on untagged cells (L1 crashed)
    dispatch_suite(p, 'L1', 'pilot', levels, tag='_L1')
    rng = random.Random(f'{SEED}:se_study:{cohort}')
    for lane in lanes[2:]:
        order = [FULL, 288, 272]; rng.shuffle(order)
        for n in order:
            p.vllm(lane, 'pilot', 'e1_e2_pilot', 'qwen32b', n, tag=f'__{lane}')
    for b, lane in enumerate(lanes):
        rng = random.Random(f'{SEED}:{cohort}:se_study:block{b}')
        cells = [('e2', n, None) for n in levels]
        cells += [('e6_balanced', n, balanced(n, cohort)) for n in unbalanced]
        cells += [('e6_decode', n, n) for n in bal_masks]
        rng.shuffle(cells)
        for exp, n, cu in cells:
            p.vllm(lane, 'production', exp, 'qwen32b', n, acq=b, block=b, requires='release_e2', tag=f'__cu{cu}' if cu else '',
                   extra_env=[f'CUBW_CU_COUNT_OVERRIDE={cu}'] if cu else None)
    order = iter((lanes[4:] + lanes[:4]) * 20)
    for n in (FULL, 272):
        p.vllm(next(order), 'production', 'e6_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace',
               requires='release_e2')
    for n in unbalanced:
        p.vllm(next(order), 'production', 'e6_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'},
               tag=f'__cu{balanced(n, cohort)}__ktrace', requires='release_e2', extra_env=sized(n))
    for acq in range(3):
        for n in levels:
            p.shell(next(order), 'production', 'e6_shapes', f'n{n}_a{acq}',
                    e6_cmd(n, 'e6.json', cu_counts=[balanced(n, cohort)] if balanced(n, cohort) != n else None),
                    est=180, timeout=900, level=n, acquisition=acq)
    p.levels, p.v16 = levels, True
    return p


def build_followup(cohort, smoke=False):
    """v1.5 MI350X follow-up (leak test v3 -> E3-minimal; shader-engine balance of wvSplitK). smoke: one GPU,
    pilot-labelled, releases nothing."""
    unbalanced = [n for n in FINE if balanced(n) != n]
    if smoke:
        p = Plan('pilot', cohort)
        st = 'pilot'
        placement_core(p, st, lanes=['L0'], levels=FINE)
        rng = random.Random(f'{SEED}:followup-smoke:{cohort}')
        leak_suite_v3(p, 'L0', st, rng)
        dispatch_suite(p, 'L0', st, FINE)
        for n in (FULL, 208):
            p.vllm('L0', st, 'e1_e2_pilot', 'qwen32b', n)
        p.vllm('L0', st, 'e6_balanced', 'qwen32b', 208, tag=f'__cu{balanced(208)}',
               extra_env=[f'CUBW_CU_COUNT_OVERRIDE={balanced(208)}'])
        p.vllm('L0', st, 'e6_decode', 'qwen32b', 192, tag='__cu192', extra_env=['CUBW_CU_COUNT_OVERRIDE=192'])
        for n in (FULL, 208, 192):
            p.shell('L0', st, 'e6_shapes', f'n{n}_a0', e6_cmd(n, 'e6.json', cu_counts=[balanced(n)] if balanced(n) != n else None),
                    est=180, timeout=900, level=n, acquisition=0)
        p.n_lanes, p.partition, p.levels = 1, 'mi3501x', FINE
        p.followup = True
        return p
    p = Plan('auto', cohort)
    lanes = [f'L{i}' for i in range(8)]
    st = 'pilot'
    placement_core(p, st, levels=FINE)
    rng = random.Random(f'{SEED}:followup:{cohort}')
    leak_suite_v3(p, 'L0', st, rng)
    leak_suite_v3(p, 'L1', st, rng)
    dispatch_suite(p, 'L2', st, FINE)
    p.competitor_only('L3', st, 'e1_e3_pilot', f'alone_fma_{COMPETITOR_MASK}', 'arith100', 30)
    p.competitor_only('L3', st, 'e1_e3_pilot', f'alone_read100_{COMPETITOR_MASK}', 'mem100', 30)
    p.competitor_only('L3', st, 'e1_e3_pilot', f'alone_idle_{COMPETITOR_MASK}', 'inactive', 10)
    tr = ['absent', 'inactive', 'mem100', 'arith100']; rng.shuffle(tr)
    for t in tr:
        p.vllm('L3', st, 'e1_e3_pilot', 'qwen32b', E3_LEVELS[0], t)
    for lane in ('L4', 'L5', 'L6', 'L7'):
        order = [FULL] + unbalanced[:2]; rng.shuffle(order)
        for n in order:
            p.vllm(lane, st, 'e1_e2_pilot', 'qwen32b', n, tag=f'__{lane}')
    st = 'production'
    # blocks: lane b holds Qwen32B block b: E2 at every fine level, the balanced-size decode at every unbalanced level,
    # the sized decode at the balanced levels below the device, and (blocks 0-4) E3-minimal; all paired within one GPU
    for b, lane in enumerate(lanes):
        rng = random.Random(f'{SEED}:{cohort}:followup:block{b}')
        cells = [('e2', n, 'absent', None, 'release_e2') for n in FINE]
        cells += [('e6_balanced', n, 'absent', balanced(n), 'release_e2') for n in unbalanced]
        cells += [('e6_decode', n, 'absent', n, 'release_e2') for n in FINE if n != FULL and n not in unbalanced]
        if b < 5:
            cells += [('e3', n, t, None, 'release_e3_minimal_v3') for n in E3_MINIMAL['levels'] for t in E3_MINIMAL['treatments']]
        if b == 0:
            cells.append(('e3', E3_LEVELS[0], 'inactive', None, 'release_e3_minimal_v3'))
        rng.shuffle(cells)
        for exp, n, t, cu, req in cells:
            p.vllm(lane, st, exp, 'qwen32b', n, t, acq=b, block=b, requires=req, tag=f'__cu{cu}' if cu else '',
                   extra_env=[f'CUBW_CU_COUNT_OVERRIDE={cu}'] if cu else None)
    order = iter((lanes[5:] + lanes[:5]) * 20)
    for n in unbalanced:
        p.vllm(next(order), st, 'e6_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag=f'__cu{balanced(n)}__ktrace',
               requires='release_e2', extra_env=[f'CUBW_CU_COUNT_OVERRIDE={balanced(n)}'])
    for acq in range(3):
        for n in FINE:
            p.shell(next(order), st, 'e6_shapes', f'n{n}_a{acq}',
                    e6_cmd(n, 'e6.json', cu_counts=[balanced(n)] if balanced(n) != n else None), est=180, timeout=900,
                    level=n, acquisition=acq)
    p.followup = True
    p.levels = FINE
    return p


def build_extension(cohort):
    p = Plan('auto', cohort)
    lanes = [f'L{i}' for i in range(8)]
    st = 'pilot'
    placement_core(p, st, levels=FINE)
    rng = random.Random(f'{SEED}:extension:{cohort}')
    for lane in (('L0', 'L1') if cohort == 'mi300x' else ('L0',)):
        leak_suite_v2(p, lane, st, rng)
    st = 'production'
    # E2-fine blocks (8) and E6 decode with the kernel sized for the active CUs (5 blocks), paired within a GPU
    for b, lane in enumerate(lanes):
        rng = random.Random(f'{SEED}:{cohort}:extension:block{b}')
        cells = [('e2', n, None) for n in FINE]
        if b in (3, 4, 5, 6, 7):
            cells += [('e6_decode', n, n) for n in OVERRIDE_LEVELS]
        rng.shuffle(cells)
        for exp, n, cu in cells:
            p.vllm(lane, st, exp, 'qwen32b', n, acq=b, block=b, requires='release_e2', tag=f'__cu{cu}' if cu else '',
                   extra_env=[f'CUBW_CU_COUNT_OVERRIDE={cu}'] if cu else None)
    # MI300X: E3-minimal (Qwen32B 232/152 x absent/mem50/mem100/arith100, 5 blocks) if the v2 leak test passes
    if cohort == 'mi300x':
        for b, lane in enumerate(lanes[:5]):
            rng = random.Random(f'{SEED}:{cohort}:extension:e3:block{b}')
            cells = [(n, t) for n in E3_MINIMAL['levels'] for t in E3_MINIMAL['treatments']]
            rng.shuffle(cells)
            for n, t in cells:
                p.vllm(lane, st, 'e3', 'qwen32b', n, t, acq=b, block=b, requires='release_e3_minimal_v2')
    # tail work spread over the lanes with the fewest decoder cells first
    order = iter((lanes[5:] + lanes[:5]) * 12)
    for n in FINE:
        p.vllm(next(order), st, 'e6_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace',
               requires='release_e2')
    for n in COUNTER_LEVELS[1:3]:
        p.vllm(next(order), st, 'e6_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag=f'__cu{n}__ktrace',
               requires='release_e2', extra_env=[f'CUBW_CU_COUNT_OVERRIDE={n}'])
    for acq in range(3):
        for n in FINE:
            p.shell(next(order), st, 'e6_shapes', f'n{n}_a{acq}', e6_cmd(n, 'e6.json'), est=180, timeout=900,
                    level=n, acquisition=acq)
    for n in COUNTER_LEVELS:
        for c in EXT_COUNTERS:
            p.shell(next(order), st, 'e6_counters', f'n{n}_{c}', e6_cmd(n, 'e6.json', light=True, counter=c), est=150,
                    timeout=900, level=n, counter=c)
    for n in (FULL, BOUNDARY):
        for c in ('FETCH_SIZE', 'TCC_EA0_RDREQ_LEVEL'):
            p.vllm(next(order), st, 'e6_eager_counters', 'qwen32b', n, profile={'kind': 'counters', 'counters': [c]},
                   config='configs/qwen32b_eager_b1_c4096_profile.json', tag=f'__eager__{c}', requires='release_e2')
    p.extension = True
    return p


def build_campaign(cohort):
    """v1.4 new-cohort campaign (MI350X), one allocation: the MI300X/MI325X pilot, production and extension work
    at this cohort's levels. E3 uses the per-CU leak test (release_e3_minimal_v2); the full E3 grid needs the clock
    gate, which no earlier cohort passed, so it is not planned."""
    p = Plan('auto', cohort)
    lanes = [f'L{i}' for i in range(8)]
    union = sorted(set(LEVELS) | set(FINE), reverse=True)
    st = 'pilot'
    placement_core(p, st, levels=union)
    caps(p, 'L0', st)
    placement_negative(p, 'L0', st)
    rng = random.Random(f'{SEED}:campaign:{cohort}')
    leak_suite(p, 'L1', st, rng)
    leak_suite_v2(p, 'L2', st, rng)
    leak_suite_v2(p, 'L3', st, rng)
    p.competitor_only('L4', st, 'e1_e3_pilot', f'alone_fma_{COMPETITOR_MASK}', 'arith100', 30)
    p.competitor_only('L4', st, 'e1_e3_pilot', f'alone_read100_{COMPETITOR_MASK}', 'mem100', 30)
    p.competitor_only('L4', st, 'e1_e3_pilot', f'alone_idle_{COMPETITOR_MASK}', 'inactive', 10)
    tr = ['absent', 'inactive', 'mem100', 'arith100']; rng.shuffle(tr)
    for t in tr:
        p.vllm('L4', st, 'e1_e3_pilot', 'qwen32b', E3_LEVELS[0], t)
    for lane in ('L5', 'L6'):
        order = LEVELS[:]; rng.shuffle(order)
        for n in order:
            p.vllm(lane, st, 'e1_e2_pilot', 'qwen32b', n, tag=f'__{lane}')
    for n in (FULL, FULL // 2):
        p.vllm('L7', st, 'e1_timeline', 'qwen32b', n)
        p.vllm('L7', st, 'e1_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace')
    st = 'production'
    # blocks: lane b holds Qwen32B block b (E2 at every level, E3-minimal on blocks 0-4, the sized-kernel decode on
    # blocks 3-7) and, on lanes 0-2, Qwen14B block b at the nominal levels; all paired within one GPU
    for b, lane in enumerate(lanes):
        rng = random.Random(f'{SEED}:{cohort}:campaign:block{b}')
        cells = [('e2', 'qwen32b', n, 'absent', None, 'release_e2') for n in union]
        if b < 5:
            cells += [('e3', 'qwen32b', n, t, None, 'release_e3_minimal_v2')
                      for n in E3_MINIMAL['levels'] for t in E3_MINIMAL['treatments']]
        if b == 0:
            cells.append(('e3', 'qwen32b', E3_LEVELS[0], 'inactive', None, 'release_e3_minimal_v2'))
        if b >= 3:
            cells += [('e6_decode', 'qwen32b', n, 'absent', n, 'release_e2') for n in OVERRIDE_LEVELS]
        if b < 3:
            cells += [('e2', 'qwen14b', n, 'absent', None, 'release_e2') for n in LEVELS]
        rng.shuffle(cells)
        for exp, model, n, t, cu, req in cells:
            p.vllm(lane, st, exp, model, n, t, acq=b, block=b, requires=req, tag=f'__cu{cu}' if cu else '',
                   extra_env=[f'CUBW_CU_COUNT_OVERRIDE={cu}'] if cu else None)
    # tail, spread over the lanes with the fewest block cells first
    order = iter((lanes[5:] + lanes[3:5] + lanes[:3]) * 40)
    for acq in range(3):
        for n in LEVELS + ['competitor']:
            suite = e4_suite(n) if n != 'competitor' else [
                ('comp_seqread4g', ['--mode', 'read', '--bytes', str(4 << 30), '--reps', '5']),
                ('comp_read1g', ['--mode', 'read', '--bytes', str(1 << 30), '--reps', '10']),
                ('comp_fma', ['--mode', 'fma', '--iters', '2000', '--reps', '5'])]
            rng = random.Random(f'{SEED}:{cohort}:e4:{acq}:{n}')
            suite = suite[:]; rng.shuffle(suite)
            lane = next(order)
            for name, args in suite:
                p.native(lane, st, 'e4', name, args, COMPETITOR_MASK if n == 'competitor' else mask_for(n), acq=acq,
                         level=n)
    for n in FINE:  # kernel-trace timelines at every fine level (E5 at the full device and the boundary included)
        p.vllm(next(order), st, 'e6_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag='__ktrace',
               requires='release_e2')
    for n in COUNTER_LEVELS[1:3]:
        p.vllm(next(order), st, 'e6_timeline', 'qwen32b', n, profile={'kind': 'kernel_trace'}, tag=f'__cu{n}__ktrace',
               requires='release_e2', extra_env=[f'CUBW_CU_COUNT_OVERRIDE={n}'])
    for acq in range(3):
        for n in FINE:
            p.shell(next(order), st, 'e6_shapes', f'n{n}_a{acq}', e6_cmd(n, 'e6.json'), est=180, timeout=900,
                    level=n, acquisition=acq)
    for n in COUNTER_LEVELS:
        for c in EXT_COUNTERS:
            p.shell(next(order), st, 'e6_counters', f'n{n}_{c}', e6_cmd(n, 'e6.json', light=True, counter=c), est=150,
                    timeout=900, level=n, counter=c)
    for n in (FULL, BOUNDARY):
        for c in ('FETCH_SIZE', 'TCC_EA0_RDREQ_LEVEL'):
            p.vllm(next(order), st, 'e6_eager_counters', 'qwen32b', n, profile={'kind': 'counters', 'counters': [c]},
                   config='configs/qwen32b_eager_b1_c4096_profile.json', tag=f'__eager__{c}', requires='release_e2')
    for acq in range(3):  # E8 guardrails at the full device, the boundary and the lowest level; matched energy
        for n in sorted({FULL, BOUNDARY, LEVELS[-1]}):
            p.vllm(next(order), st, 'e8', 'qwen7b_b8', n, acq=acq, requires='release_e5e8')
            p.vllm(next(order), st, 'e8', 'qwen32b', n, acq=acq, tokens=2, tag='__prefill', requires='release_e5e8')
        for n in (FULL, BOUNDARY):
            p.vllm(next(order), st, 'e8', 'qwen32b_c16k', n, acq=acq, requires='release_e5e8')
            p.vllm(next(order), st, 'e8', 'mistral7b', n, acq=acq, requires='release_e5e8')
            p.vllm(next(order), st, 'e8', 'qwen32b', n, acq=acq, mode='energy',
                   config='configs_energy/qwen32b_graph_b1_c4096_t10.json', tag='__energy', requires='release_e5e8')
    p.extension = True
    p.campaign = True
    p.levels = union
    return p


def build_smoke(cohort):
    """v1.4 one-GPU smoke on the cohort's single-GPU partition (MI350X: mi3501x) before its 8-GPU campaign: placement,
    capability/counter audit, leak test v2, Qwen32B decode at three levels and with the sized kernel, a kernel trace,
    E6 shapes and two counters, and a few E4 rows. Pilot-labelled; it releases nothing."""
    p = Plan('pilot', cohort)
    st = 'pilot'
    placement_core(p, st, lanes=['L0'], levels=[FULL, LEVELS[1], FULL // 2])
    caps(p, 'L0', st)
    rng = random.Random(f'{SEED}:smoke:{cohort}')
    leak_suite_v2(p, 'L0', st, rng)
    for n in (FULL, LEVELS[1], FULL // 2):
        p.vllm('L0', st, 'e1_e2_pilot', 'qwen32b', n)
    p.vllm('L0', st, 'e6_decode', 'qwen32b', FULL // 2, tag=f'__cu{FULL // 2}', extra_env=[f'CUBW_CU_COUNT_OVERRIDE={FULL // 2}'])
    p.vllm('L0', st, 'e6_timeline', 'qwen32b', FULL, profile={'kind': 'kernel_trace'}, tag='__ktrace')
    for n in (FULL, FULL // 2):
        p.shell('L0', st, 'e6_shapes', f'n{n}_a0', e6_cmd(n, 'e6.json'), est=180, timeout=900, level=n, acquisition=0)
        for name, args in e4_suite(n):
            if name.split('_', 1)[1] in ('seqread4g', 'ilp1', 'ilp8', f'chase_c{FULL}', 'fma'):
                p.native('L0', st, 'e4_pilot', name, args, mask_for(n), level=n)
    for c in ('FETCH_SIZE', 'TCC_EA0_RDREQ_LEVEL'):
        p.shell('L0', st, 'e6_counters', f'n{FULL}_{c}', e6_cmd(FULL, 'e6.json', light=True, counter=c), est=150,
                timeout=900, level=FULL, counter=c)
    p.n_lanes, p.partition, p.levels = 1, 'mi3501x', [FULL, LEVELS[1], FULL // 2]
    return p


def finalize(p, src, args):
    lanes = {f'L{i}': {'gpu_index': i} for i in range(getattr(p, 'n_lanes', 8))}
    budget, stages = {}, {}
    for s in p.steps:
        b = budget.setdefault(s['lane'], {'steps': 0, 'est_s': 0})
        b['steps'] += 1; b['est_s'] += s['est_s']
        stages.setdefault(s['stage'], {}).setdefault(s['lane'], 0)
        stages[s['stage']][s['lane']] += s['est_s']
    longest = sum(max(v.values()) for v in stages.values())  # stages run one after another
    rates = {k: v['rate'] for k, v in HW.items()}  # allocation units per node-hour (MI350X: 8 x 1.4, not yet observed)
    rate = {'mi3501x': 1.4, 'mi3001x': 1.0}.get(getattr(p, 'partition', None), rates[args.cohort])  # 1-GPU slices: 8 x 0.175 / 0.125
    plan = {
        'version': 'cubw_v1.6' if getattr(p, 'v16', False) else 'cubw_v1.5' if getattr(p, 'followup', False) else 'cubw_v1.4' if getattr(p, 'campaign', False) else 'cubw_v1.3' if getattr(p, 'extension', False) else 'cubw_v1.2',
        'e4_done_in': args.e4_done_in, 'campaign': getattr(p, 'campaign', False), 'followup': getattr(p, 'followup', False),
        'extension': getattr(p, 'extension', False), 'fine_levels': FINE if getattr(p, 'extension', False) else None, 'kind': p.kind, 'cohort': args.cohort, 'seed': SEED,
        'assignment': 'docs/amd/compute_bandwidth_followup_20261026.md',
        'full_cus': FULL, 'competitor_mask': COMPETITOR_MASK, 'arch': HW[args.cohort]['arch'],
        'gpu_name': HW[args.cohort]['gpu_name'], 'partition': getattr(p, 'partition', HW[args.cohort]['partition']),
        'levels': getattr(p, 'levels', None) or (FINE if getattr(p, 'extension', False) else LEVELS), 'e3_levels': E3_LEVELS,
        'nominal_levels': LEVELS,
        'e3_treatments': E3_TREATMENTS, 'e3_minimal_subset': E3_MINIMAL, 'boundary_level': args.boundary,
        'lanes': lanes, 'derived_configs': derived_configs(src), 'config_source': str(src),
        'statistics': {'warmups': 3, 'trials_per_acquisition': 10, 'acquisitions': 'one per block/lane; >=3 per cell',
                       'bootstrap': '10,000 draws seed 20261018, blocks resampled when cells are blocked, trials nested',
                       'noninferiority': f'upper 95% CI of R(n)=T(n)/T({FULL}) <= 1.05 (latency)'},
        'gates': {
            'placement_ok': 'every decoder/competitor mask: distinct physical CUs == requested, per-XCD counts differ by <= 1',
            'lifecycle_ok': 'every competitor: started event, stopped event, PID gone, no owned residual GPU process',
            'outputs_ok': 'Qwen32B pilot runs complete with one output digest across all CU levels and treatments',
            'leak_ok': 'v1, recorded only: probe TFLOP/s during decoder trials >= 0.85 x probe alone on the same mask',
            'leak_ok_v11': 'every decoder probe ratio (probe during decoder / probe alone) >= 0.85 x the highest ratio of the '
                           'same probe beside a placement-proven native co-runner (fma or read) on the decoder mask, same GPU',
            'clock_ok': 'pooled in-window gfx and memory clocks over Qwen32B E2 pilot trials: max/min <= 1.05, coverage >= 0.5',
            'release_e2': 'placement_ok and outputs_ok (observational label if clock_ok fails)',
            'release_e3_full': 'placement_ok and lifecycle_ok and outputs_ok and leak_ok_v11 and clock_ok',
            'release_e3_minimal': 'placement_ok and lifecycle_ok and outputs_ok and leak_ok_v11',
            'release_e5e8': 'outputs_ok',
            'leak_ok_v2': 'v1.3, predeclared before any per-CU data: per-CU probe rate ratio r_c = rate_c(beside X) / '
                          'rate_c(probe alone), same GPU and probe mask; q_c = r_c / median_c(r_c). Every run covers all '
                          f'{FULL}-n probe CUs. Sensitivity: the positive-control run (probe co-runner sharing 8 CUs) has >= 4 '
                          'CUs with q_c < 0.85, else the test is insensitive and leak_ok_v2 is false. Pass iff, for every '
                          'decoder run, min_c q_c >= (lowest min_c q_c of the native read/fma co-runner runs on the same '
                          'GPU and mask) - 0.05',
            'release_e3_minimal_v2': 'placement_ok and lifecycle_ok and outputs_ok and leak_ok_v2',
            'leak_ok_v3': 'v1.5, predeclared before any duration-probe data (docs/amd/mi350x_followup_plan_20261103.md): '
                          'the leak_ok_v2 rule applied to per-CU speed s_c = completed probe workgroups / summed workgroup '
                          'residency (svc_bench_v5 cuprobe_t) instead of completion rate. Natives: co-runners on 0..n-1; '
                          'positive controls: fma, read and the v4 full-LDS probe on 0..n+7 (8 shared CUs), each of which '
                          'must flag >= 4 CUs with q_c < 0.85, else the test is insensitive and leak_ok_v3 is false',
            'release_e3_minimal_v3': 'placement_ok and lifecycle_ok and outputs_ok and leak_ok_v3',
            'leak_ok_v4': 'v1.6, predeclared before any start-delay-probe data (docs/amd/followups_v16_plan_20261104.md): '
                          'svc_bench_v6 gapprobe on n..FULL-1, one full-LDS 16-wave workgroup per probe CU per launch, 200 us '
                          'host gap. Per CU c: f_c = (launches whose first workgroup on c started > 5 us after the launch\'s '
                          'first start + launches with no workgroup on c) / launches; e_c = f_c(beside X) - f_c(alone), same '
                          'GPU and mask; c is flagged if e_c >= 0.10. The alone run must cover all FULL-n probe CUs. '
                          'Sensitivity: the v4 full-LDS probe on 0..n+7 and the decoder on 0..n+7 must each flag >= 4 CUs '
                          '(fma and read on 0..n+7 are recorded, not required), else the test is insensitive and leak_ok_v4 '
                          'is false. Pass iff every decoder run at n flags no more CUs than the most flagged by a native '
                          'read/fma co-runner on 0..n-1, on every GPU and level',
            'release_e3_minimal_v4': 'placement_ok and lifecycle_ok and outputs_ok and leak_ok_v4'},
        'unavailable': {'e5_graph_counters': 'rocprofv3 --pmc on graph-mode vLLM decode failed in both MI300X pilot '
                                              'attempts (illegal memory access; hang to timeout), job 481167'},
        'retry_policy': 'two attempts maximum, only for execution/tool failures; correctness failures never retried',
        'timeouts': 'per step: max(900 s, 3 x estimate) for decoder steps; native 120 s; competitor alone seconds+120',
        'reserve_s': 1800,
        'stages': list(stages),
        'budget': {'per_lane': budget, 'per_stage_longest_lane_s': {k: max(v.values()) for k, v in stages.items()},
                   'longest_lane_s': longest, 'node_hours_est': round(longest / 3600 + 0.25, 2),
                   'allocation_units_est': round((longest / 3600 + 0.25) * rate, 1),
                   'rate_units_per_node_hour': rate},
        'steps': p.steps,
    }
    return plan


def main(args):
    src = Path(args.config_source)
    configure(args.cohort)
    if args.leak4:
        p = build_leak4(args.cohort, smoke=args.smoke)
    elif args.se_study:
        p = build_se_study(args.cohort, smoke=args.smoke)
    elif args.followup:
        p = build_followup(args.cohort, smoke=args.smoke)
    elif args.smoke:
        p = build_smoke(args.cohort)
    elif args.campaign:
        p = build_campaign(args.cohort)
    elif args.kind == 'pilot':
        p = build_pilot(args.cohort)
    elif args.kind == 'production':
        p = Plan('production', args.cohort)
        production(p, 'production', [('qwen32b', 5), ('qwen14b', 3)], args.boundary, sorted({FULL, args.boundary}),
                   e4=not args.e4_done_in)
    elif args.extension:
        p = build_extension(args.cohort)
    elif args.pilot == 'minimal':  # auto after a full pilot of this cohort: v1.1 re-pilot, gate, production
        p = build_minimal_pilot(args.cohort)
        p.kind = 'auto'
        production(p, 'production', [('qwen32b', 5), ('qwen14b', 3)], args.boundary, sorted({FULL, args.boundary}))
    else:  # auto: reduced pilot, gate, then a budget-limited production stage
        p = build_pilot(args.cohort, reduced=True)
        p.kind = 'auto'
        production(p, 'production', [('qwen32b', 3), ('qwen14b', 3)], args.boundary, [FULL])
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
    q.add_argument('--cohort', choices=sorted(HW), required=True)
    q.add_argument('--pilot', choices=['reduced', 'minimal'], default='reduced', help='pilot stage of an auto plan')
    q.add_argument('--boundary', type=int, default=152, help='E2 boundary level used by E5/E8 (set after the pilot)')
    q.add_argument('--config-source', default='/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/amd_20261018_mi300x')
    q.add_argument('--e4-done-in', help='production only: omit E4; its 3 acquisitions are in this earlier job root')
    q.add_argument('--extension', action='store_true', help='auto: v1.3 extension (leak v2, E2-fine, E6, counters)')
    q.add_argument('--smoke', action='store_true', help='pilot: v1.4 one-GPU smoke (MI350X on mi3501x)')
    q.add_argument('--campaign', action='store_true', help='auto: v1.4 new-cohort campaign (MI350X)')
    q.add_argument('--followup', action='store_true', help='v1.5 MI350X follow-up (with --smoke: the one-GPU smoke)')
    q.add_argument('--leak4', action='store_true', help='v1.6 MI350X leak test v4 + E3-minimal (with --smoke: one GPU)')
    q.add_argument('--se-study', action='store_true', help='v1.6 MI300X shader-engine study (with --smoke: one GPU)')
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
