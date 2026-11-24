#!/usr/bin/env python3
"""Long-context / long-output chunked-prefill campaign plans (docs/amd/lc_campaign_design_20261124.md; handoff
docs/amd/long_context_chunked_prefill_20261124.md). Same plan format, runner (scripts/amd/cp/cp_run.py) and job
(scripts/amd/cp/cp.sbatch) as the November 18 campaign; one plan = one persistent run root.

Plans:
  smoke  one MI300X GPU (mi3001x): the 8K regression anchor, one 32K prompt whole and chunked, a genuinely mixed step at
         a 32K resident context, kernel-trace and counter paths, the two-request tail, a long-context serving trace and
         the host counter audit
  lc1    one mi3008x node: L0 anchors, L1 (prefill cost), L2 (decode vs prefill attention), L3 part A (tails), L4 (counters,
         streaming reference, disjoint-CU competition) and the SLO/capacity pilot for L3 part B and L5
  lc2 / lc3   written after the lc1 pilot (design sections 7-8)

  lc_plan.py smoke|lc1 --cohort mi300x --out plan.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cp'))
from cp_plan import CFG, COHORTS, Plan, TREATMENTS, finalize  # noqa: E402,F401

R = 'scripts/amd/cp/cp_replay.py'
LR = 'scripts/amd/lc/lc_replay.py'
LP = 'scripts/amd/lc/lc_prefill.py'
LT = 'scripts/amd/lc/lc_tail.py'
SV = 'scripts/amd/cp/cp_serve.py'
LENGTHS = '1024,8192,16384,24576,32000'
CONTEXTS = (1024, 8192, 16384, 24576, 32000)
ARMS = {'whole': ('whole', 32768), 'mix256': ('mixed', 256), 'mix1024': ('mixed', 1024), 'mix4096': ('mixed', 4096)}
# tail policies: (cp_sched policy, budget, whole prompts only). Common max_model_len 16384.
TAIL = {'whole': ('separate', 16384, True), 'sep2048': ('separate', 2048, False), 'mix256': ('mixed', 256, False),
        'mix2048': ('mixed', 2048, False), 'mix4096': ('mixed', 4096, False)}
COUNTERS = ('FETCH_SIZE', 'WRITE_SIZE', 'SQ_INSTS_VALU_MFMA_MOPS_BF16', 'TCC_EA0_RDREQ_DRAM_sum')
L4_TREAT = ('absent', 'arith100', 'mem50', 'mem100')
# SLO pilot (stock: mixed B = 2048, full allocation, low rate, development seed 101) and capacity ladder (seed 102)
PILOT = {'lo512': (0.2, 40, [0.5, 1, 1.5, 2, 3, 4]), 'lo2048': (0.05, 40, [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]),
         'lo4096': (0.03, 30, [0.08, 0.12, 0.18, 0.25, 0.35, 0.5]), 'lctx': (0.03, 30, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]),
         'lprompt': (0.02, 30, [0.03, 0.05, 0.08, 0.12, 0.16, 0.24])}
DRAIN = {'lo512': 240, 'lo2048': 300, 'lo4096': 420, 'lctx': 300, 'lprompt': 240}


def base(m='q32'):
    return ['--config', CFG[m], '--gpu-uuid', '@UUID']


def smoke(cohort):
    p = Plan('lc_smoke', cohort)
    p.add('a', 'lc0', 'sm_anchor_e1_S8192_N8', R, base() + ['--contexts', 8192, '--widths', 8, '--chunks', '0,256,1024',
          '--reps', 2, '--budget', 4096, '--seed-offset', 1], est=600, model='q32', pin='L0')
    p.add('a', 'lc1', 'sm_prefill_whole', LP, base() + ['--arm', 'whole', '--budget', 32768, '--lengths', '1024,32000',
          '--reps', 1], est=600, model='q32', arm='whole', pin='L0')
    p.add('a', 'lc1', 'sm_prefill_mix1024', LP, base() + ['--arm', 'mixed', '--budget', 1024, '--lengths', '1024,32000',
          '--reps', 1], est=600, model='q32', arm='mix1024', pin='L0')
    p.add('a', 'lc2', 'sm_replay_S32000_N8_P4096', LR, base() + ['--contexts', 32000, '--widths', 8, '--chunks', '0,1024',
          '--prefix', 4096, '--anchors', '--reps', 2], est=900, model='q32', pin='L0')
    p.add('a', 'lc2p', 'sm_ktrace_replay_S16384_N1', LR, base() + ['--contexts', 16384, '--widths', 1, '--chunks', '0,1024',
          '--prefix', 4096, '--anchors', '--reps', 1, '--markers'], est=900, model='q32', profile=True, pin='L0')
    p.add('a', 'lc1p', 'sm_ktrace_prefill_whole_8192', LP, base() + ['--arm', 'whole', '--budget', 32768, '--lengths', 8192,
          '--reps', 1, '--markers'], est=600, model='q32', profile=True, pin='L0')
    p.add('a', 'lc4c', 'sm_pmc_replay_FETCH_SIZE', LR, base() + ['--contexts', 8192, '--widths', 1, '--chunks', '0,256',
          '--prefix', 4096, '--reps', 1, '--markers', '--eager'], est=1200, model='q32', profile=True, pmc=['FETCH_SIZE'],
          pin='L0')
    p.add('a', 'lc3a', 'sm_tail_mix2048', LT, base() + ['--policy', 'mixed', '--budget', 2048, '--outputs', 512,
          '--injects', '0,8192', '--reps', 1], est=600, model='q32', pin='L0')
    p.add('a', 'lc5', 'sm_serve_lctx', SV, base() + ['--policy', 'mixed', '--budget', 2048, '--workload', 'lctx', '--rates', 0.1,
          '--seeds', 101, '--requests', 8, '--drain-s', 180, '--max-model-len', 32768, '--energy'], est=900, model='q32',
          pin='L0')
    p.add('a', 'lc4a', 'sm_counter_audit', 'scripts/amd/goal/counter_audit.py', ['--max-counters', 2, '--reps', 1], est=600,
          kind='host', pin='L0')
    p.steps = [dict(s, order=i, lane='L0') for i, s in enumerate(p.steps)]
    lanes = {'a': {'L0': [0]}}
    return p, lanes, {'a': {'L0': sum(s['est_s'] for s in p.steps)}}


def lc1(cohort):
    p = Plan('lc1', cohort)
    c = p.c
    full = c['full']
    # L0 anchors: experiment-1 cell S 8192 N 8 with the s1 settings (budget 4096, max_model_len 16384, seed offset 1)
    p.add('a', 'lc0', 'lc0_e1_S8192_N8', R, base() + ['--contexts', 8192, '--widths', 8, '--chunks', '0,256,1024', '--reps', 10,
          '--budget', 4096, '--seed-offset', 1], est=900, model='q32')
    # serving anchor: experiment-2 held-out dheavy, tuned mixed B = 1024, 2 rps, seed 201 (s3 e2h)
    p.add('a', 'lc0', 'lc0_e2h_dheavy_mix1024_r2_s201', SV, base() + ['--policy', 'mixed', '--budget', 1024, '--workload',
          'dheavy', '--rates', 2.0, '--seeds', 201, '--arrivals', 'poisson', '--requests', 150, '--drain-s', 60, '--energy'],
          est=600, model='q32', workload='dheavy')
    # L1: prefill cost, clean timing; kernel trace; eager calibration; counters (eager)
    for an, (arm, b) in ARMS.items():
        args = base() + ['--arm', arm, '--budget', b, '--lengths', LENGTHS]
        p.add('a', 'lc1', f'lc1_time_{an}', LP, args + ['--reps', 8], est=1200, model='q32', arm=an)
        p.add('a', 'lc1p', f'lc1_ktrace_{an}', LP, args + ['--reps', 2, '--markers'], est=1200, model='q32', arm=an, profile=True)
    for an in ('whole', 'mix256', 'mix1024'):
        arm, b = ARMS[an]
        args = base() + ['--arm', arm, '--budget', b, '--lengths', '8192,32000', '--eager']
        p.add('a', 'lc1e', f'lc1_eager_{an}', LP, args + ['--reps', 3], est=900, model='q32', arm=an)
        for ctr in COUNTERS:
            p.add('a', 'lc1c', f'lc1_pmc_{an}_{ctr}', LP, args + ['--reps', 1, '--markers'], est=1500, model='q32', arm=an,
                  profile=True, pmc=[ctr], counter=ctr)
    # L2: decode vs prefill attention (timing per context; kernel trace per context)
    for S in CONTEXTS:
        extra = ['--prefix-sweep', '0,4096,16384,28672', '--sweep-cell', '8192:8', '--anchors'] if S == 8192 else []
        p.add('a', 'lc2', f'lc2_time_S{S}', LR, base() + ['--contexts', S, '--widths', '1,8,32', '--max-tokens', 300000,
              '--chunks', '0,256,1024', '--prefix', 4096, '--reps', 10] + extra, est=1500, model='q32', S=S)
        p.add('a', 'lc2p', f'lc2_ktrace_S{S}', LR, base() + ['--contexts', S, '--widths', '1,8', '--chunks', '0,256,1024',
              '--prefix', 4096, '--reps', 2, '--markers'] + (['--anchors'] if S == 8192 else []), est=1200, model='q32', S=S,
              profile=True)
    # L3 part A: two-request tails (timing; kernel trace on O = 512)
    for pn, (pol, b, whole) in TAIL.items():
        args = base() + ['--policy', pol, '--budget', b] + (['--no-chunk'] if whole else [])
        p.add('a', 'lc3a', f'lc3a_tail_{pn}', LT, args + ['--reps', 3], est=2400, model='q32', pname=pn)
        p.add('a', 'lc3p', f'lc3a_ktrace_{pn}', LT, args + ['--outputs', 512, '--reps', 1, '--markers'], est=900, model='q32',
              pname=pn, profile=True)
    # L4: counters on pure-decode and mixed long-context cells (eager), eager calibration, known-byte audit, streaming
    l4 = ['--contexts', '1024,16384,32000', '--widths', '1,8', '--chunks', '0,1024', '--prefix', 4096]
    p.add('a', 'lc4e', 'lc4_eager_time', LR, base() + l4 + ['--reps', 4, '--eager'], est=1200, model='q32')
    for ctr in COUNTERS:
        p.add('a', 'lc4c', f'lc4_pmc_{ctr}', LR, base() + l4 + ['--reps', 2, '--eager', '--markers'], est=2400, model='q32',
              profile=True, pmc=[ctr], counter=ctr)
    p.add('a', 'lc4a', 'lc4_counter_audit', 'scripts/amd/goal/counter_audit.py', ['--max-counters', 16, '--reps', 2], est=1800,
          kind='host')
    # L4 competition: decoder at 256t, competitor on 256-303; two blocks pinned to lanes L6 and L7, placement first
    for b, lane in ((0, 'L6'), (1, 'L7')):
        p.add('a', 'lc4s', f'lc4_stream_full_b{b}', None, ['--mode', 'read', '--seconds', 20, '--duty', 1.0, '--bytes', 1 << 30],
              est=60, kind='native', pin=lane)
        p.add('a', 'lc4s', f'lc4_stream_256_b{b}', None, ['--mode', 'read', '--seconds', 20, '--duty', 1.0, '--bytes', 1 << 30],
              est=60, kind='native', mask=f'0-{c["comp_levels"][0] - 1}', pin=lane)
        p.add('a', 'lc4m', f'lc4_place_256_b{b}', None, ['--mode', 'placement', '--blocks', 8 * full, '--trials', 5,
              '--expect-cus', c['comp_levels'][0]], est=10, kind='native', mask=f'0-{c["comp_levels"][0] - 1}', pin=lane)
        p.add('a', 'lc4m', f'lc4_place_pool_b{b}', None, ['--mode', 'placement', '--blocks', 8 * full, '--trials', 5,
              '--expect-cus', full - c['comp_levels'][0]], est=10, kind='native', mask=c['pool'], pin=lane)
        n = c['comp_levels'][0]
        for t in L4_TREAT:
            p.add('a', 'lc4m', f'lc4_comp_n{n}_{t}_b{b}', LR, base() + l4 + ['--reps', 6], est=1500, cus=n, grid=n, treatment=t,
                  pin=lane, model='q32', alloc=f'{n}t', block=b)
    # SLO pilot and capacity ladder for L3 part B / L5 (stock mixed B = 2048, full allocation, max_model_len 32768)
    for wl, (lo, n_lo, ladder) in PILOT.items():
        common = base() + ['--policy', 'mixed', '--budget', 2048, '--workload', wl, '--max-model-len', 32768]
        p.add('a', 'pilot', f'pilot_slo_{wl}', SV, common + ['--rates', lo, '--seeds', 101, '--requests', n_lo, '--drain-s',
              DRAIN[wl]], est=int(n_lo / lo + DRAIN[wl] + 400), model='q32', workload=wl)
        p.add('a', 'pilot', f'pilot_cap_{wl}', SV, common + ['--rates', ','.join(map(str, ladder)), '--seeds', 102, '--requests', 60,
              '--drain-s', DRAIN[wl], '--energy'], est=int(sum(60 / r + DRAIN[wl] * 0.6 for r in ladder) + 400), model='q32',
              workload=wl)
    lanes = {'a': {f'L{i}': [i] for i in range(8)}}
    return p, lanes, {'a': p.assign('a', list(lanes['a']))}


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('which', choices=('smoke', 'lc1'))
    q.add_argument('--cohort', choices=sorted(COHORTS), required=True)
    q.add_argument('--out', type=Path, required=True)
    a = q.parse_args()
    p, lanes, load = smoke(a.cohort) if a.which == 'smoke' else lc1(a.cohort)
    plan = finalize(p, lanes, load)
    if a.which == 'smoke':  # one MI300X GPU on the single-GPU partition
        plan.update(partition='mi3001x', gpus=1, charge_rate=0.125)
        plan['budget']['allocation_units_est'] = round(plan['budget']['node_hours_est'] * 0.125, 2)
    plan['campaign'] = 'lc_20261124'
    plan['design'] = 'docs/amd/lc_campaign_design_20261124.md'
    plan['handoff'] = 'docs/amd/long_context_chunked_prefill_20261124.md'
    plan.pop('plan_sha256', None)
    a.out.write_text(json.dumps(plan, indent=1) + '\n')
    print(f"{plan['plan']}/{a.cohort}: {len(plan['steps'])} steps; longest lane {plan['budget']['longest_lane_s']}; "
          f"~{plan['budget']['node_hours_est']} node-h ~{plan['budget']['allocation_units_est']} units")
