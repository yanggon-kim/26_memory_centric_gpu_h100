#!/usr/bin/env python3
"""Score the pre-registered MI300X shader-engine predictions (followups_v16_plan_20261104.md §B) for job 499334.

Usage: judge_se_mi300x.py <raw run root in $WORK/results_amd> <out hse_judgment.json>. CPU only; reads analysis/cubw_tables.json
and e6_dispatch/*/a0/attempt1/out.json. P1m is judged on lane L0 only (L1 crashed, see mi300x_se_review_20261105.md §1)."""
import sys, json, pathlib, collections, statistics as st
sys.path.insert(0, '/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/scripts/amd/cubw')
import analyze as A
R = pathlib.Path(sys.argv[1]); out = sys.argv[2]
t = json.load(open(R / 'analysis/cubw_tables.json'))
BAL = [288, 256, 224, 192, 128]; UNB = [272, 232, 200, 152]
bal = lambda n: n - n % 32
# P1m from the raw out.json, per trial (only lane L0 ran: L1 crashed on a directory collision)
disp = {}
for d in sorted((R / 'e6_dispatch').iterdir()):
    o = d / 'a0/attempt1/out.json'
    if not o.exists(): continue
    res = json.loads(o.read_text())
    if isinstance(res, list): res = res[-1]
    s = A.dispatch_summary(res)
    trials = []
    for tr in res['trials']:
        x = A.dispatch_summary({'trials': [tr], 'blocks': res.get('blocks')})
        trials.append({'max_wg': x['max_wg_per_cu'], 'serialized': x['serialized_wgs_mean'], 'ms': round(x['kernel_ms_median'], 3)})
    disp[d.name] = {'lds': s['lds_bytes'], 'ms': round(s['kernel_ms_median'], 3), 'cus_used': s['cus_used'],
                    'max_wg': s['max_wg_per_cu'], 'serialized_mean': s['serialized_wgs_mean'],
                    'wg_per_se_xcd0': s['wg_per_se_xcd0'], 'trials': trials}
one = lambda k: all(tr['max_wg'] == 1 and tr['serialized'] == 0 for tr in disp[k]['trials'])
ser = lambda k: disp[k]['max_wg'] >= 2 and disp[k]['serialized_mean'] > 0
a = all(one(f'n{n}_g{n}') for n in BAL)
b304 = ser('n304_g304'); bred = sum(ser(f'n{n}_g{n}') for n in UNB)
c = all(one(f'n{n}_g{bal(n) if n != 304 else 288}') for n in [304] + UNB) and all(one(f'n{n}_g{n}') for n in BAL)
P1 = {'gpus_judged': ['L0'], 'gpus_missing': {'L1': 'lane crashed: FileExistsError, both lanes mapped to e6_dispatch/<cell>/a0'},
      'a_balanced_one_per_cu_every_trial': a, 'b_full_device_304_serialized': b304, 'b_reduced_unbalanced_serialized': f'{bred}/4',
      'b_pass': b304 and bred >= 3, 'c_balanced_grid_one_per_cu': c}
P1['pass'] = a and P1['b_pass'] and c
P1['dispatch'] = disp
# P2m
sh = t['e6']['shapes']; shapes = ['down', 'gate_up', 'lm_head', 'o_proj', 'qkv']
P2 = {}
for n in [304] + UNB:
    bn = 288 if n == 304 else bal(n); rows = {}
    for s in shapes:
        b_ = sh[f'{s}:wvsplitk_c{bn}'][str(n)]['mean_us']; n_ = sh[f'{s}:wvsplitk_n'][str(n)]['mean_us']
        rows[s] = {'balanced_us': round(b_, 1), 'size_n_us': round(n_, 1), 'faster': b_ < n_}
    P2[str(n)] = {'balanced_size': bn, 'shapes': rows, 'faster': sum(r['faster'] for r in rows.values()),
                  'pass': sum(r['faster'] for r in rows.values()) >= 4}
P2 = {'levels': P2, 'pass': all(v['pass'] for v in P2.values())}
# P3m
def blk(exp, n, extra=None):
    out = {}
    for x in t['acquisitions']:
        if x['experiment'] == exp and x['n'] == n and A.good(x) and x['model'] == 'qwen32b':
            out[x['block'] if x['block'] is not None else x['lane']] = x['ms_per_step']
    return out
base = blk('e2', 304)
P3a = {}
for n in UNB:
    rb = A.ratio_ci(blk('e6_balanced', n), base); re = A.ratio_ci(blk('e2', n), base)
    P3a[str(n)] = {'R_balanced': rb, 'R_e2': re, 'pass': rb['ratio'] < re['ratio']}
r304 = A.ratio_ci(blk('e6_balanced', 304), base)
P3 = {'a': P3a, 'a_pass': all(v['pass'] for v in P3a.values()),
      'b_T304_cu288_over_T304_stock': r304, 'b_pass': r304['ratio'] < 1 and r304['ci95'][1] < 1}
P3['recorded_e6_decode'] = {str(n): A.ratio_ci(blk('e6_decode', n), base) for n in BAL}
P3['pass'] = P3['a_pass'] and P3['b_pass']
verdict = ('holds on MI300X' if P1['pass'] and P3['pass'] else 'refuted on MI300X' if not P1['pass']
           else 'dispatch artefact exists on gfx942 but does not drive MI300X decode')
J = {'job': 499334, 'pre_registration': 'docs/amd/followups_v16_plan_20261104.md', 'P1m': P1, 'P2m': P2, 'P3m': P3,
     'verdict_pre_registered': verdict}
json.dump(J, open(out, 'w'), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('dispatch', 'levels', 'a', 'recorded_e6_decode')} if isinstance(v, dict) else v for k, v in J.items()}, indent=1))
for n, v in P2['levels'].items(): print('P2', n, v['faster'])
for n, v in P3a.items(): print('P3a', n, round(v['R_balanced']['ratio'],4), [round(x,4) for x in v['R_balanced']['ci95']], round(v['R_e2']['ratio'],4), [round(x,4) for x in v['R_e2']['ci95']])
for n, v in P3['recorded_e6_decode'].items(): print('e6_decode', n, round(v['ratio'],4), [round(x,4) for x in v['ci95']])
