#!/usr/bin/env python3
"""CPU checks for goal_v1 (no GPU): plan invariants for every cohort and smoke, serving statistics helpers, G8 freeze
reproducibility. Run: python3.12 scripts/amd/goal/test_goal.py"""
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import plan_goal as G  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {detail}' if detail and not ok else ''))
    if not ok:
        FAIL.append(name)


def plans():
    src = Path('/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/amd_20261018_mi300x')
    for cohort in G.COHORTS:
        for smoke in ((False, True) if G.COHORTS[cohort]['single'] else (False,)):
            p = G.build(cohort, smoke)
            yield cohort, smoke, G.finalize(p, cohort, src), src


def main():
    for cohort, smoke, plan, src in plans():
        tag = f'{cohort}{" smoke" if smoke else ""}'
        st = plan['steps']
        check(f'{tag}: unique attempt dirs', len({s['out'] for s in st}) == len(st))
        check(f'{tag}: unique step ids per lane', len({(s['id'], s['lane']) for s in st}) == len(st))
        cfgs = {s['config'] for s in st if s.get('config')}
        missing = [c for c in cfgs if c not in plan['derived_configs'] and not (src / c).exists()]
        check(f'{tag}: every config exists or is derived', not missing, str(missing))
        bad = [s['id'] for s in st if s.get('extra_env') and '__stock' in s['cell']]
        check(f'{tag}: stock cells carry no override', not bad, str(bad[:3]))
        bad = [s['id'] for s in st if s['kind'] in ('vllm', 'serve') and '__g' in s['cell'] and not s.get('extra_env')
               and s['experiment'] not in ('e1_leak_v4', 'e1_leak_v2')]
        check(f'{tag}: tuned cells carry the override', not bad, str(bad[:3]))
        leak_dec = [s for s in st if s['experiment'] == 'e1_leak_v4' and s['kind'] == 'vllm' and 'positive' not in s['cell']]
        check(f'{tag}: leak v4 decoder is tuned (grid = n)',
              all(s['extra_env'] == [f"CUBW_CU_COUNT_OVERRIDE={s['decoder_cus']}"] for s in leak_dec) and leak_dec)
        pos = [s for s in st if s['experiment'] == 'e1_leak_v4' and 'positive' in s['cell']]
        check(f'{tag}: leak v4 decoder positive is stock', pos and all(not s.get('extra_env') for s in pos))
        if smoke:
            check(f'{tag}: one lane, single-GPU partition', list(plan['lanes']) == ['L0'] and plan['partition'] == G.COHORTS[cohort]['single'])
            continue
        prod = [s for s in st if s['stage'] == 'production']
        check(f'{tag}: every production step is gated', all(s.get('requires') in ('release_g1', 'release_g5') for s in prod))
        check(f'{tag}: G5 cells require release_g5', all(s['requires'] == 'release_g5' for s in prod if s['experiment'] == 'g5'))
        for lane in plan['lanes']:
            g1 = [s for s in prod if s['lane'] == lane]
            check(f'{tag} {lane}: block starts and ends with stock full', g1[0]['cell'].endswith('__first') and g1[-1]['cell'].endswith('__last'))
        g5 = [s for s in prod if s['experiment'] == 'g5']
        lo = int(plan['competitor_mask'].split('-')[0])
        check(f'{tag}: G5 decoder masks stay below the competitor pool', all(s['decoder_cus'] <= lo for s in g5))
        check(f'{tag}: G5 competitor is the fixed pool', all(s.get('competitor_mask') == plan['competitor_mask'] for s in g5))
        serve = [s for s in st if s['kind'] == 'serve']
        check(f'{tag}: serving has 4 blocks x 5 conditions x 2 concurrencies', len([s for s in serve if s['experiment'] == 'g2_serve']) == 40)
        n_g1 = len([s for s in prod if s['experiment'] == 'g1'])
        check(f'{tag}: G1 has 8 blocks of every cell', n_g1 == 8 * (len(plan['g1_cells']) + 2))
        routes = [s for s in st if s['experiment'] == 'g2_route']
        check(f'{tag}: kernel routes for 7 cases x A,C', len(routes) == 14 and all(s.get('profile') for s in routes))
        check(f'{tag}: no profiled step outside route/counter experiments',
              all(s['experiment'] in ('g2_route', 'g7_counters') for s in st if s.get('profile')))
        check(f'{tag}: budget under 12 h', plan['budget']['longest_lane_s'] + plan['reserve_s'] < 11.5 * 3600,
              str(plan['budget']['longest_lane_s']))
    import serve_mix as S
    check('pct interpolates', abs(S.pct([1, 2, 3, 4], 0.5) - 2.5) < 1e-9 and S.pct([5], 0.99) == 5)
    check('summary keys', set(S.summary([1.0, 2.0])) == {'n', 'mean', 'p50', 'p90', 'p99', 'max'})
    tr = S.trace(10, [3, 5], [2, 4], 100, 7)
    check('trace deterministic', tr == S.trace(10, [3, 5], [2, 4], 100, 7) and [len(r['prompt_ids']) for r in tr[:2]] == [3, 5])
    import g8_model as M
    with tempfile.TemporaryDirectory() as d:
        class A:
            out = Path(d) / 'f.json'
        M.freeze(A)
        x = json.loads(A.out.read_text())
        frozen = HERE.parents[2] / 'docs/amd/goal/g8_frozen_model.json'
        if frozen.exists():
            y = json.loads(frozen.read_text())
            check('G8 freeze reproduces the committed predictions',
                  x['parameters'] == y['parameters'] and x['held_out_predictions'] == y['held_out_predictions'])
        check('G8 held-out points are unmeasured cells', len(x['held_out_predictions']) == 10)
    print(f'{len(FAIL)} failures')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
