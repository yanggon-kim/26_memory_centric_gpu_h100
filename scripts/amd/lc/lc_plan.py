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
DRAIN = {'lo512': 240, 'lo2048': 300, 'lo4096': 420, 'lctx': 300, 'lprompt': 240, 'balanced': 60}


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


GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SERVE_POL = {'whole': ('separate', 32768, True), 'sep2048': ('separate', 2048, False), 'mix256': ('mixed', 256, False),
             'mix512': ('mixed', 512, False), 'mix1024': ('mixed', 1024, False), 'mix2048': ('mixed', 2048, False),
             'mix4096': ('mixed', 4096, False), 'mix8192': ('mixed', 8192, False)}
L3B_POL = ('whole', 'sep2048', 'mix256', 'mix2048', 'mix4096')
L5_WL = ('lo2048', 'lctx', 'lprompt')
L3B_WL = ('lo512', 'lo4096', 'balanced')


def serve(p, stage, exp, sid, wl, pn, rates, seed, n, drain, **k):
    pol, b, whole = SERVE_POL[pn]
    args = base() + ['--policy', pol, '--budget', b, '--workload', wl, '--rates', ','.join(f'{r:.4g}' for r in rates),
                     '--seeds', seed, '--arrivals', k.pop('arrivals', 'poisson'), '--requests', n, '--drain-s', drain,
                     '--max-model-len', 32768, '--energy'] + (['--no-chunk'] if whole else [])
    p.add(stage, exp, sid, SV, args, est=int(sum(n / r + 0.6 * drain + 30 for r in rates) + 400), model='q32', workload=wl,
          pname=pn, policy=pol, budget=b, whole=whole, seed=seed, rates=list(rates), **k)


def lc2(cohort, params):
    """Development stage (design section 7, amendment 2): L5 tuning on seeds 101 and 102 for lo2048, lctx, lprompt
    (all eight policies); L3 part B matched-policy traces on seed 101 for lo512, lo4096 and the balanced control (five
    policies); a fixed-rate slice at lambda*(lo4096) for lo512 and lo2048 (five policies). Rates GRID x lambda*."""
    p = Plan('lc2', cohort)
    lam = params['lambda_star']
    for wl in L5_WL:
        for pn in SERVE_POL:
            for seed in (101, 102):
                serve(p, 'a', 'lc5d', f'lc5d_{wl}_{pn}_s{seed}', wl, pn, [x * lam[wl] for x in GRID], seed,
                      params['requests'][wl], DRAIN[wl])
    for wl in L3B_WL:
        for pn in L3B_POL:
            serve(p, 'a', 'lc3b', f'lc3b_{wl}_{pn}_s101', wl, pn, [x * lam[wl] for x in GRID], 101, params['requests'][wl],
                  DRAIN.get(wl, 120))
    # L4 FETCH_SIZE redo (amendment 2): lc1 lc4_pmc_FETCH_SIZE hit an illegal memory access in the S=32000 N=8 cell on
    # both attempts (kept as failed); the other cells as one step, the S=32000 N=8 cell alone as another
    l4 = ['--widths', '1,8', '--chunks', '0,1024', '--prefix', 4096, '--reps', 2, '--eager', '--markers']
    p.add('a', 'lc4c', 'lc4_pmc_FETCH_SIZE_r2_main', LR, base() + ['--contexts', '1024,16384,32000', '--max-tokens', 131072]
          + l4, est=2000, model='q32', profile=True, pmc=['FETCH_SIZE'], counter='FETCH_SIZE')
    p.add('a', 'lc4c', 'lc4_pmc_FETCH_SIZE_r2_S32000_N8', LR, base() + ['--contexts', 32000, '--widths', 8, '--chunks', '0,1024',
          '--prefix', 4096, '--reps', 2, '--eager', '--markers'], est=900, model='q32', profile=True, pmc=['FETCH_SIZE'],
          counter='FETCH_SIZE')
    # known-byte counter audit redo (amendment 2): lc1 lc4_counter_audit saw no device (host launcher set HIP and ROCR
    # visibility to the same physical index; fixed in cp_run.run_host); kept as a failed acquisition
    p.add('a', 'lc4a', 'lc4_counter_audit_r2', 'scripts/amd/goal/counter_audit.py', ['--max-counters', 16, '--reps', 2],
          est=1800, kind='host')
    rf = lam['lo4096']
    for wl in ('lo512', 'lo2048'):
        for pn in L3B_POL:
            serve(p, 'a', 'lc3s', f'lc3s_{wl}_{pn}_r{rf:.3g}_s101', wl, pn, [rf], 101, params['requests']['lo4096'], DRAIN[wl])
    lanes = {'a': {f'L{i}': [i] for i in range(8)}}
    return p, lanes, {'a': p.assign('a', list(lanes['a']))}


def lc3(cohort, params):
    """Held-out stage (design section 7, amendments 2-3). Policies per L5 workload from the lc2 selection:
    the selected mixed budget, the selected separate policy, stock mixed B 2048 and whole (amendment 3).
      lc5h  held-out seeds 201-203, poisson, GRID x lambda*
      lc5b  bursty repeat (gamma gaps, CV 4), seed 201, GRID x lambda*
      7B replication (Qwen2.5-7B, same harness): L1 timing (whole, B 256 / 1024 / 4096), L2 timing (S 1K-32K, N 1/8/32),
      L3 part A tails (whole, mix2048), SLO pilot + stock capacity ladder for lctx and lprompt, and matched ladders of the
      32B-selected policies on seed 201 (experiment lc5r)"""
    p = Plan('lc3', cohort)
    lam, sel = params['lambda_star'], params['selection']
    ext = params.get('grid_ext', {})  # amendment 3: extra multiples of lambda* where lc2 was censored at 2 x lambda*
    for wl in L5_WL:
        pols = list(dict.fromkeys([sel[wl]['mixed'], sel[wl]['separate'], 'mix2048', 'whole']))
        for pn in pols:
            rates = [x * lam[wl] for x in GRID + tuple(ext.get(wl, ()))]
            for seed in (201, 202, 203):
                serve(p, 'a', 'lc5h', f'lc5h_{wl}_{pn}_s{seed}', wl, pn, rates, seed, params['requests'][wl], DRAIN[wl])
            serve(p, 'a', 'lc5b', f'lc5b_{wl}_{pn}_s201', wl, pn, rates, 201, params['requests'][wl], DRAIN[wl],
                  arrivals='bursty')
    b7 = lambda: base('q7')  # noqa: E731
    for an, (arm, b) in ARMS.items():
        p.add('a', 'lc1', f'q7_lc1_time_{an}', LP, b7() + ['--arm', arm, '--budget', b, '--lengths', LENGTHS, '--reps', 8],
              est=600, model='q7', arm=an)
    for S in CONTEXTS:
        p.add('a', 'lc2', f'q7_lc2_time_S{S}', LR, b7() + ['--contexts', S, '--widths', '1,8,32', '--max-tokens', 1200000,
              '--chunks', '0,256,1024', '--prefix', 4096, '--reps', 10], est=900, model='q7', S=S)
    for pn in ('whole', 'mix2048'):
        pol, b, whole = TAIL[pn]
        p.add('a', 'lc3a', f'q7_lc3a_tail_{pn}', LT, b7() + ['--policy', pol, '--budget', b] + (['--no-chunk'] if whole else [])
              + ['--reps', 2], est=1200, model='q7', pname=pn)
    q7 = params['q7']
    for wl in ('lctx', 'lprompt'):
        lo, n_lo, ladder = q7['pilot'][wl]
        common = b7() + ['--policy', 'mixed', '--budget', 2048, '--workload', wl, '--max-model-len', 32768]
        p.add('a', 'pilot', f'q7_pilot_slo_{wl}', SV, common + ['--rates', lo, '--seeds', 101, '--requests', n_lo, '--drain-s',
              DRAIN[wl]], est=int(n_lo / lo + DRAIN[wl] + 400), model='q7', workload=wl)
        p.add('a', 'pilot', f'q7_pilot_cap_{wl}', SV, common + ['--rates', ','.join(map(str, ladder)), '--seeds', 102,
              '--requests', 60, '--drain-s', DRAIN[wl], '--energy'], est=int(sum(60 / r + DRAIN[wl] * 0.6 for r in ladder) + 400),
              model='q7', workload=wl)
        for pn in dict.fromkeys([sel[wl]['mixed'], sel[wl]['separate'], 'mix2048', 'whole']):
            pol, b, whole = SERVE_POL[pn]
            args = b7() + ['--policy', pol, '--budget', b, '--workload', wl, '--rates', ','.join(map(str, ladder)), '--seeds', 201,
                           '--arrivals', 'poisson', '--requests', 60, '--drain-s', DRAIN[wl], '--max-model-len', 32768,
                           '--energy'] + (['--no-chunk'] if whole else [])
            p.add('a', 'lc5r', f'q7_lc5r_{wl}_{pn}_s201', SV, args, est=int(sum(60 / r + DRAIN[wl] * 0.6 for r in ladder) + 400),
                  model='q7', workload=wl, pname=pn, policy=pol, budget=b, whole=whole, seed=201, rates=list(ladder))
    lanes = {'a': {f'L{i}': [i] for i in range(8)}}
    return p, lanes, {'a': p.assign('a', list(lanes['a']))}


def lc3fx(cohort, params):
    """Amendment 4 re-run: the lc3 steps listed in params['rerun'], same arguments and seed, id <step>_fx, fixed release.
    With params['rerun_split'] each ladder is split into one step per rate (<step>_fx_r<i>) so all eight GPUs are used."""
    src, _, _ = lc3(cohort, params)
    p = Plan('lc3fx', cohort)
    want = set(params['rerun'])
    for st in src.steps:
        if st['id'] not in want:
            continue
        want.discard(st['id'])
        rates = st.get('rates')
        if params.get('rerun_split') and rates and len(rates) > 1:
            a = st['args']
            i = a.index('--rates')
            n, drain = int(a[a.index('--requests') + 1]), int(a[a.index('--drain-s') + 1])
            for j, r in enumerate(rates):
                q = dict(st, id=f"{st['id']}_fx_r{j}", rates=[r], rerun_of=st['id'],
                         args=a[:i + 1] + [f'{r:.4g}'] + a[i + 2:], est_s=int(n / r + 0.6 * drain + 430))
                q['timeout_s'] = int(q['est_s'] * 3 + 900)
                p.steps.append(q)
        else:
            p.steps.append(dict(st, id=f"{st['id']}_fx", rerun_of=st['id']))
    assert not want, f'not in the lc3 plan: {sorted(want)}'
    for st in p.steps:
        st.pop('lane', None), st.pop('order', None)
    lanes = {'a': {f'L{i}': [i] for i in range(8)}}
    return p, lanes, {'a': p.assign('a', list(lanes['a']))}


DG = 'scripts/amd/cp/cp_disagg.py'
Q4 = {'Q0123': [0, 1, 2, 3], 'Q4567': [4, 5, 6, 7]}


def ld(cohort, params, smoke=False):
    """Amendment 5: prefill/decode disaggregation at long prompts (lctx, lprompt), Qwen2.5-32B, four MI300X per lane.
    Deployments on four GPUs: colo4_whole and colo4_mix2048 (four colocated engines, least-outstanding-work routing)
    and two disaggregated ratios per workload (lctx 1P3D / 2P2D, lprompt 2P2D / 3P1D): producers run whole prompts
    (separate, no chunking, budget 32768), consumers stock mixed scheduling, P2pNccl PUT_ASYNC KV transfer.
    Rates = per-GPU lc ladder x 4; held-out seeds 201 and 202; every engine 64 seats and max_model_len 32768."""
    p = Plan('ld' + ('_smoke' if smoke else ''), cohort)
    lam, n_req = params['lambda_star'], params['ld_requests']
    base4 = ['--config', CFG['q32'], '--gpu-uuids', '@UUIDS', '--gpu-indices', '@IDX', '--max-model-len', 32768,
             '--max-seqs', 64]
    whole = ['--policy', 'separate', '--budget', 32768, '--no-chunk']
    dis = lambda npf: ['--deployment', 'disagg', '--prefill-workers', npf, '--decode-policy', 'mixed',  # noqa: E731
                       '--decode-gpu-mem', params['ld_decode_gpu_mem'], '--decode-kv-buffer', params['ld_decode_kv_buffer'],
                       '--p2p-none-blocks-fix'] + whole
    deps = {'colo4_whole': ['--deployment', 'colocated'] + whole,
            'colo4_mix2048': ['--deployment', 'colocated', '--policy', 'mixed', '--budget', 2048]}
    ratios = params['ld_ratios']
    est = lambda n, rates, drain: int(sum(n / r + 0.6 * drain + 30 for r in rates) + 600)  # noqa: E731
    if smoke:
        for wl, npf, r in (('lprompt', 3, 0.6), ('lctx', 1, 0.8)):
            p.add('a', 'ld', f'lds_{wl}_d{npf}p{4 - npf}', DG, base4 + dis(npf) + ['--workload', wl, '--rates', r, '--seeds', 201,
                  '--requests', 24, '--drain-s', DRAIN[wl]], est=est(24, [r], DRAIN[wl]), model='q32', workload=wl,
                  deployment=f'd{npf}p{4 - npf}', prefill_workers=npf, seed=201, rates=[r])
        return p, {'a': Q4}, {'a': p.assign('a', list(Q4))}
    for wl in ('lctx', 'lprompt'):
        rates = [4 * x * lam[wl] for x in GRID + tuple(params['grid_ext'][wl])]
        dd = dict(deps, **{f'd{k}p{4 - k}': dis(k) for k in ratios[wl]})
        for dn, extra in dd.items():
            for seed in (201, 202):
                p.add('a', 'ld', f'ld_{wl}_{dn}_s{seed}', DG, base4 + extra + ['--workload', wl, '--rates',
                      ','.join(f'{r:.4g}' for r in rates), '--seeds', seed, '--requests', n_req[wl], '--drain-s', DRAIN[wl]],
                      est=est(n_req[wl], rates, DRAIN[wl]), model='q32', workload=wl, deployment=dn,
                      prefill_workers=int(dn[1]) if dn.startswith('d') else None, seed=seed, rates=rates)
    return p, {'a': Q4}, {'a': p.assign('a', list(Q4))}


def ldfx(cohort, params):
    """Amendment 6: re-run the ld steps whose decode engine died (hipErrorLaunchOutOfResources) as <step>_fx, identical
    except consumers at gpu_memory_utilization params['ldfx_decode_gpu_mem'] (more runtime headroom)."""
    full, _, _ = ld(cohort, dict(params, ld_decode_gpu_mem=params['ldfx_decode_gpu_mem']))
    p = Plan('ldfx', cohort)
    for st in full.steps:
        if st['id'] in params['ldfx_steps']:
            p.steps.append(dict(st, id=st['id'] + '_fx', rerun_of=st['id'], decode_gpu_mem=params['ldfx_decode_gpu_mem']))
    assert len(p.steps) == len(params['ldfx_steps'])
    for st in p.steps:
        st.pop('lane', None), st.pop('order', None)
    return p, {'a': Q4}, {'a': p.assign('a', list(Q4))}


O8 = {'O8': [0, 1, 2, 3, 4, 5, 6, 7]}


def _ld2_dis(params, npf, kv):
    ch, st = kv
    return ['--deployment', 'disagg', '--prefill-workers', npf, '--decode-policy', 'mixed', '--decode-gpu-mem',
            params['ld2_decode_gpu_mem'], '--decode-kv-buffer', params['ld2_decode_kv_buffer'], '--p2p-none-blocks-fix',
            '--kv-channels', ch, '--kv-send-type', st, '--policy', 'separate', '--budget', 32768, '--no-chunk']


_LD2_COLO = {'whole': ['--deployment', 'colocated', '--policy', 'separate', '--budget', 32768, '--no-chunk'],
             'mix2048': ['--deployment', 'colocated', '--policy', 'mixed', '--budget', 2048],
             'mix8192': ['--deployment', 'colocated', '--policy', 'mixed', '--budget', 8192]}
_LD2_BASE = ['--config', CFG['q32'], '--gpu-uuids', '@UUIDS', '--gpu-indices', '@IDX', '--max-model-len', 32768, '--max-seqs', 64]


def _ld2_est(n, rates, drain):
    return int(sum(n / r + 0.6 * drain + 30 for r in rates) + 600)


def ld2a(cohort, params):
    """Amendment 7 phase A: connector micro-test. lprompt 3P1D at 0.1 rps, 30 requests, development seed 101, one step per
    P2pNccl setting (nccl_num_channels, send_type) plus colo4_whole at the same load; two 4-GPU lanes."""
    p = Plan('ld2a', cohort)
    tr = ['--workload', 'lprompt', '--rates', 0.1, '--seeds', 101, '--requests', 30, '--drain-s', params['ld2_drain']['lprompt']]
    est = _ld2_est(30, [0.1], params['ld2_drain']['lprompt'])
    for name, kv in params['ld2a_settings'].items():
        p.add('a', 'ld2a', f'ld2a_{name}', DG, _LD2_BASE + _ld2_dis(params, 3, kv) + tr, est=est, model='q32',
              workload='lprompt', deployment='d3p1', prefill_workers=3, seed=101, rates=[0.1], setting=name)
    p.add('a', 'ld2a', 'ld2a_colo4_whole', DG, _LD2_BASE + _LD2_COLO['whole'] + tr, est=est, model='q32', workload='lprompt',
          deployment='colo4_whole', prefill_workers=None, seed=101, rates=[0.1], setting='colo')
    return p, {'a': Q4}, {'a': p.assign('a', list(Q4))}


def ld2b(cohort, params):
    """Amendment 7 phase B: one workload (lclo), one 8-GPU deployment at a time. colo8_{whole,mix2048,mix8192} and
    d{k}p{8-k} for k = 2..6 with the phase A connector setting; held-out seeds 201 and 202."""
    assert params.get('ld2_kv'), 'ld2_kv must be set from the phase A rule first'
    p = Plan('ld2b', cohort)
    wl, n, dr, rates = 'lclo', params['ld2_requests']['lclo'], params['ld2_drain']['lclo'], params['ld2_rates']['lclo']
    deps = {f'colo8_{k}': v for k, v in _LD2_COLO.items()}
    deps.update({f'd{k}p{8 - k}': _ld2_dis(params, k, params['ld2_kv']) for k in range(2, 7)})
    for dn, extra in deps.items():
        for seed in (201, 202):
            p.add('a', 'ld', f'ld2b_{wl}_{dn}_s{seed}', DG, _LD2_BASE + extra + ['--workload', wl, '--rates',
                  ','.join(f'{r:.4g}' for r in rates), '--seeds', seed, '--requests', n, '--drain-s', dr],
                  est=_ld2_est(n, rates, dr), model='q32', workload=wl, deployment=dn,
                  prefill_workers=int(dn[1]) if dn.startswith('d') else None, seed=seed, rates=rates)
    return p, {'a': O8}, {'a': p.assign('a', list(O8))}


def ld2c(cohort, params):
    """Amendment 7 phase C (deployments from the phase B rule, params['ld2c']): lclo with bursty arrivals; the selected
    disaggregated ratio with consumers restricted to params['ld2c']['decode_cus'] CUs; the mixed-tenant workload mixdoc."""
    c = params.get('ld2c')
    assert c, 'ld2c must be set from the phase B rule first'
    p = Plan('ld2c', cohort)
    kv = params['ld2_kv']

    def dep(dn):
        return _LD2_COLO[dn[6:]] if dn.startswith('colo8_') else _ld2_dis(params, int(dn[1]), kv)

    def add(tag, wl, dn, extra, arrival, seed):
        n, dr, rates = params['ld2_requests'][wl], params['ld2_drain'][wl], params['ld2_rates'][wl]
        base = dn.split('_dcu')[0]
        p.add('a', 'ld', f'ld2c_{tag}_{wl}_{dn}_s{seed}', DG, _LD2_BASE + extra + ['--workload', wl, '--arrivals', arrival,
              '--rates', ','.join(f'{r:.4g}' for r in rates), '--seeds', seed, '--requests', n, '--drain-s', dr],
              est=_ld2_est(n, rates, dr), model='q32', workload=wl, deployment=dn,
              prefill_workers=int(base[1]) if base.startswith('d') else None, seed=seed, rates=rates, arrival=arrival)
    for seed in (201, 202):
        for dn in c['bursty_deps']:
            add('bursty', 'lclo', dn, dep(dn), 'bursty', seed)
        dn = c['cu_dep']
        add('cu', 'lclo', f"{dn}_dcu{c['decode_cus']}", dep(dn) + ['--decode-cus', c['decode_cus'], '--decode-grid',
            c['decode_cus']], 'poisson', seed)
        for dn in c['mixdoc_deps']:
            add('mix', 'mixdoc', dn, dep(dn), 'poisson', seed)
    return p, {'a': O8}, {'a': p.assign('a', list(O8))}


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('which', choices=('smoke', 'lc1', 'lc2', 'lc3', 'lc3fx', 'ld', 'ld_smoke', 'ldfx', 'ld2a', 'ld2b', 'ld2c'))
    q.add_argument('--params', type=Path)
    q.add_argument('--cohort', choices=sorted(COHORTS), required=True)
    q.add_argument('--out', type=Path, required=True)
    a = q.parse_args()
    p, lanes, load = (smoke(a.cohort) if a.which == 'smoke' else lc1(a.cohort) if a.which == 'lc1'
                      else {'lc2': lc2, 'lc3': lc3, 'lc3fx': lc3fx, 'ld': ld, 'ldfx': ldfx, 'ld2a': ld2a, 'ld2b': ld2b, 'ld2c': ld2c,
                            'ld_smoke': lambda c, prm: ld(c, prm, smoke=True)}[a.which](a.cohort, json.loads(a.params.read_text())))
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
