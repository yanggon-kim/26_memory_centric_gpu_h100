#!/usr/bin/env python3
"""Generate an immutable OC-v0 launch plan (docs/amd/overlap_calibration_followup_20261105.md; execution plan
docs/amd/overlap_calibration_execution_plan_20261105.md). Steps run with scripts/amd/cubw/cubw.py (kinds vllm, shell,
native) under scripts/amd/oc/oc.sbatch, on one single-GPU node (mi3001x / mi3501x) per job, separate from G1-G8.

Kinds:
  pilot   OC0 capability probe (PC sampling, thread trace), OC0/OC1/OC3/OC4 counter validation on known-work native
          kernels and on the decode projection route, OC2/OC3 route-matched references and known-schedule controls,
          OC1/OC6 development-model kernel controls (Qwen32B, Qwen1.5B), OC1 eager counter passes and a graph kernel trace.
          One acquisition per cell; pilot-labelled; judges nothing.
  dev     (written after the pilot review) OC2/OC3 references and OC6 development cells, 3 acquisitions x 10 trials.
  heldout (written after the frozen OC6 commit) fresh held-out cells only.

  plan_oc.py --cohort mi300x --kind pilot --out plan.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cubw'))
import plan as P  # noqa: E402

SEED = 20261106
VERSION = 'oc_v0'
COHORTS = {
    'mi300x': {'full': 304, 'arch': 'gfx942', 'gpu_name': 'MI300X', 'partition': 'mi3001x', 'rate': 0.125, 'n_p': 128,
               'pool': '256-303'},
    'mi350x': {'full': 256, 'arch': 'gfx950', 'gpu_name': 'MI350X', 'partition': 'mi3501x', 'rate': 0.175, 'n_p': 192,
               'pool': '224-255'},
}
DEV_MODELS = 'qwen32b,qwen1p5b'  # OC6 development; qwen7b, mistral7b and the qwen14b MLP shapes stay unexposed
OC1_GROUPS = {'inst': ['SQ_WAVES', 'SQ_INSTS_VALU', 'SQ_INSTS_VMEM_RD', 'SQ_INSTS_SALU'],
              'mfma': ['SQ_INSTS_MFMA', 'SQ_INSTS_VALU_MFMA_MOPS_BF16', 'SQ_VALU_MFMA_BUSY_CYCLES', 'GRBM_GUI_ACTIVE'],
              'fetch': ['FETCH_SIZE']}


def env(n, full):
    e = 'HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES,CUBW_STEP_TAG=$CUBW_STEP_TAG'
    return e + (f',HSA_CU_MASK=$HIP_VISIBLE_DEVICES:0-{n - 1}' if n < full else '')


def bench(n, full, args, out='out.json'):
    m = f'HSA_CU_MASK=$HIP_VISIBLE_DEVICES:0-{n - 1} ' if n < full else ''
    return f'{m}"$CUBW_BENCH7" {args} > "$CUBW_STEP_DIR/{out}"'


def kernels_cmd(n, full, models, grids, extra=''):
    return (f'AMD_EXTRA_ENV="{env(n, full)}" scripts/amd/vllm_in_container.sh scripts/amd/oc/oc_kernels.py --models {models} '
            f'--grids {grids} --out "$CUBW_STEP_DIR/k.json" {extra}')


def refs(c, n, g):
    """Route-matched arithmetic / memory references and known-schedule controls at mask n with g workgroups."""
    f = c['full']
    G = 1 << 30
    return {
        f'mfma4_n{n}_g{g}': f'--mode mfma4 --iters 8192 --blocks {g} --threads 1024 --reps 20 --trials 10 --warmups 3',
        f'mfma4lds_n{n}_g{g}': f'--mode mfma4_lds --iters 8192 --blocks {g} --threads 1024 --lds 16384 --reps 20 --trials 10 --warmups 3',
        f'read_n{n}_g{g}': f'--mode read --bytes {G} --blocks {g} --threads 1024 --reps 20 --trials 10 --warmups 3',
        f'read_n{n}_g8x': f'--mode read --bytes {G} --blocks {8 * n} --threads 256 --reps 20 --trials 10 --warmups 3',
        f'read_mall_n{n}_g{g}': f'--mode read --bytes {64 << 20} --passes 16 --blocks {g} --threads 1024 --reps 20 --trials 10 --warmups 3',
        **{f'ilp{k}_n{n}_g{g}': f'--mode read_ilp --ilp {k} --bytes {G} --blocks {g} --threads 1024 --reps 20 --trials 10 --warmups 3'
           for k in (1, 2, 4, 8)},
        f'split_n{n}_g{g}': f'--mode split --bytes {G} --iters 16384 --blocks {2 * g} --threads 1024 --reps 10 --trials 10 --warmups 3',
        f'fused_n{n}_g{g}': f'--mode fused --bytes {G} --intensity 8 --blocks {g} --threads 1024 --reps 10 --trials 10 --warmups 3',
    }


def build(cohort, kind):
    c = COHORTS[cohort]
    P.FULL, P.COMPETITOR_MASK = c['full'], c['pool']
    f, n_p = c['full'], c['n_p']
    p = P.Plan(kind, cohort)
    st = 'pilot'
    if kind != 'pilot':
        raise SystemExit('dev/heldout plans are generated after the pilot review (execution plan section 6)')
    P.placement_core(p, st, lanes=['L0'], levels=[f, n_p])
    p.shell('L0', st, 'oc0_probe', 'probe', f'$PY scripts/amd/oc/oc_probe.py --out-dir "$CUBW_STEP_DIR" --bench "$CUBW_BENCH7" --cus {f}',
            est=900, timeout=3600)
    p.shell('L0', st, 'oc0_counters', 'validation', f'$PY scripts/amd/oc/oc_counters.py --out-dir "$CUBW_STEP_DIR" '
            f'--bench "$CUBW_BENCH7" --cus {f}', est=2400, timeout=6000)
    rng = random.Random(f'{SEED}:{cohort}:{kind}')
    cells = []
    for n, g in ((f, f), (n_p, n_p), (f, n_p)):
        for name, args in refs(c, n, g).items():
            if name not in {x[2] for x in cells}:  # the 8-blocks-per-CU read does not depend on g
                cells.append(('ref', n, name, args))
        cells.append(('kern', n, f'kernels_n{n}_g{g}', (g,)))
    rng.shuffle(cells)
    for kind_, n, name, args in cells:
        if kind_ == 'ref':
            p.shell('L0', st, 'oc23_refs', name, bench(n, f, args), est=30, timeout=600, level=n)
        else:
            g = args[0]
            grids = 'dev' if g == f else f'dev,{g}'
            extra = '--overhead' + (' --prefill 4096' if n == f and g == f else '')
            p.shell('L0', st, 'oc16_kernels', name, kernels_cmd(n, f, DEV_MODELS, grids, extra), est=420, timeout=2400, level=n)
    for g_name, ctrs in OC1_GROUPS.items():
        s = p.vllm('L0', st, 'oc1_eager', 'qwen32b', f, config='configs/qwen32b_eager_b1_c4096_profile.json',
                   profile={'kind': 'counters', 'counters': ctrs, 'exclude_regex': '^(_|triton)'}, tag=f'__{g_name}')
        s['timeout_s'] = 1800
    p.vllm('L0', st, 'oc1_graph', 'qwen32b', f, profile={'kind': 'kernel_trace'}, tag='__ktrace')
    p.vllm('L0', st, 'oc1_graph', 'qwen32b', f, tag='__ordinary')
    p.levels = [f, n_p]
    return p


def finalize(p, cohort, src):
    c = COHORTS[cohort]
    longest = sum(s['est_s'] for s in p.steps)
    outs = [s['out'] for s in p.steps]
    assert len(set(outs)) == len(outs), 'two steps share an attempt directory'
    return {
        'version': VERSION, 'kind': p.kind, 'cohort': cohort, 'seed': SEED, 'smoke': p.kind == 'pilot',
        'assignment': 'docs/amd/overlap_calibration_followup_20261105.md',
        'framework': 'docs/compute_memory_overlap_calibration_framework_v0_20261105.md',
        'execution_plan': 'docs/amd/overlap_calibration_execution_plan_20261105.md',
        'full_cus': c['full'], 'competitor_mask': c['pool'], 'arch': c['arch'], 'gpu_name': c['gpu_name'],
        'partition': c['partition'], 'levels': p.levels, 'lanes': {'L0': {'gpu_index': 0}},
        'derived_configs': P.derived_configs(src), 'config_source': str(src), 'dev_models': DEV_MODELS,
        'unavailable': {'controlled_clocks': 'not authorized (owner): not attempted'},
        'retry_policy': 'two attempts maximum, only for execution/tool failures', 'reserve_s': 900, 'stages': ['pilot'],
        'budget': {'longest_lane_s': longest, 'node_hours_est': round(longest / 3600 + 0.3, 2),
                   'allocation_units_est': round((longest / 3600 + 0.3) * c['rate'], 3), 'rate_units_per_node_hour': c['rate']},
        'steps': p.steps,
    }


def main(a):
    p = build(a.cohort, a.kind)
    plan = finalize(p, a.cohort, Path(a.config_source))
    text = json.dumps(plan, indent=1) + '\n'
    plan['plan_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    a.out.write_text(json.dumps(plan, indent=1) + '\n')
    b = plan['budget']
    print(f"{plan['kind']}/{a.cohort}: {len(plan['steps'])} steps; ~{b['node_hours_est']} node-h ~{b['allocation_units_est']} units")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--cohort', choices=sorted(COHORTS), required=True)
    q.add_argument('--kind', choices=['pilot', 'dev', 'heldout'], default='pilot')
    q.add_argument('--config-source', default='/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/amd_20261018_mi300x')
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
