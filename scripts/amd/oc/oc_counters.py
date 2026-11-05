#!/usr/bin/env python3
"""OC-v0 counter validation and kernel-demand capture (OC0/OC1/OC3/OC4; host side, one GPU, HIP_VISIBLE_DEVICES set).

Each (target, group) runs once under rocprofv3 --pmc <group> (one pass; a group that cannot be collected in one pass
fails and is recorded, never split silently). Targets:

  native   svc_bench v7 kernels with known work, under host rocprofv3 (--bench):
             read     1 GiB stream, blocks = CUs, 1024 threads           VMEM_RD wave-instructions = n_float4 / 64
             mfma4_w1 / mfma4_w4   v_mfma 4x4x4 bf16, blocks = CUs, 256 / 1024 threads (1 / 4 waves per SIMD)
                                     MFMA wave-instructions = 8 * iters * waves; tests per-wave vs per-SIMD activity
             fma_w1 / fma_w4       VALU FMA, same occupancies
             ilp1 / ilp8           read_ilp 1 GiB, 1 and 8 loads in flight per thread (in-flight level counters)
             split    known schedule: even workgroups stream, odd run mfma4; one workgroup per CU (full LDS)
             fused    known schedule: loads in flight while independent MFMA issue in the same wave
  gemv     scripts/amd/oc/oc_kernels.py --light (Qwen32B shapes, grid = device CUs) under the container's rocprofv3
           (--container): issued MFMA / VMEM / LDS work and traffic of the actual decode projection route

Per run: return code, stderr tail, and per kernel name: dispatches, summed counter values per counter, summed
dispatch duration (profiled, replay-affected). Expected values and the validation verdicts are computed by
scripts/amd/oc/analyze_oc.py from this raw file; nothing here declares a counter valid.

For the gemv target each run also keeps `dispatch_rows`: the wvSplitK dispatches in dispatch order with grid and
counters, so a --count invocation of oc_kernels.py (one call per shape and grid, `call_order`) maps row i to call i.

  oc_counters.py --out-dir <dir> --bench <svc_bench_v7> [--targets native,gemv] [--light]
  oc_counters.py --out-dir <dir> --bench <b> --targets gemv --groups mfma,inst,fetch \
      --gemv-args '--models qwen32b,qwen1p5b --grids 304,128,192 --count'
"""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[3]
GROUPS = {
    'inst': ['SQ_WAVES', 'SQ_INSTS_VALU', 'SQ_INSTS_VMEM_RD', 'SQ_INSTS_SALU'],
    'mfma': ['SQ_INSTS_MFMA', 'SQ_INSTS_VALU_MFMA_MOPS_BF16', 'SQ_VALU_MFMA_BUSY_CYCLES', 'GRBM_GUI_ACTIVE'],
    'active': ['SQ_BUSY_CYCLES', 'SQ_BUSY_CU_CYCLES', 'SQ_ACTIVE_INST_VALU', 'SQ_ACTIVE_INST_VMEM'],
    'active2': ['SQ_ACTIVE_INST_ANY', 'SQ_WAVE_CYCLES', 'SQ_INST_CYCLES_VMEM_RD', 'GRBM_GUI_ACTIVE'],
    'vmem_level': ['SQ_INST_LEVEL_VMEM', 'SQ_INSTS_VMEM_RD', 'SQ_INSTS_VMEM_WR', 'GRBM_GUI_ACTIVE'],
    'ea_level': ['TCC_EA0_RDREQ_LEVEL', 'TCC_EA0_RDREQ', 'TCC_BUSY', 'GRBM_GUI_ACTIVE'],
    'lds': ['SQ_INSTS_LDS', 'SQ_LDS_BANK_CONFLICT', 'SQ_INSTS_SMEM', 'SQ_INSTS_VMEM_WR'],
    'fetch': ['FETCH_SIZE'], 'write': ['WRITE_SIZE'],
    'req_size': ['TCC_EA0_RDREQ_32B', 'TCC_EA0_RDREQ_64B', 'TCC_EA0_RDREQ_128B', 'TCC_EA0_RDREQ_DRAM'],
    'l2': ['TCC_HIT', 'TCC_MISS', 'TCC_REQ', 'TCC_EA0_WRREQ'],
}
G = 1 << 30


def native_targets(cus, light):
    it = 1024 if light else 4096
    return {
        'read': ['--mode', 'read', '--bytes', str(G), '--blocks', str(cus), '--threads', '1024'],
        'mfma4_w1': ['--mode', 'mfma4', '--iters', str(it), '--blocks', str(cus), '--threads', '256'],
        'mfma4_w4': ['--mode', 'mfma4', '--iters', str(it), '--blocks', str(cus), '--threads', '1024'],
        'fma_w1': ['--mode', 'fma', '--iters', str(it), '--blocks', str(cus), '--threads', '256'],
        'fma_w4': ['--mode', 'fma', '--iters', str(it), '--blocks', str(cus), '--threads', '1024'],
        'ilp1': ['--mode', 'read_ilp', '--ilp', '1', '--bytes', str(G), '--blocks', str(cus), '--threads', '1024'],
        'ilp8': ['--mode', 'read_ilp', '--ilp', '8', '--bytes', str(G), '--blocks', str(cus), '--threads', '1024'],
        'split': ['--mode', 'split', '--bytes', str(G), '--iters', str(it * 4), '--blocks', str(2 * cus), '--threads', '1024'],
        'fused': ['--mode', 'fused', '--bytes', str(G), '--intensity', '8', '--blocks', str(cus), '--threads', '1024'],
    }


def listing(container):
    if container:
        env = dict(os.environ, AMD_ENTRY='rocprofv3', AMD_EXTRA_ENV=f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '0')}"
                + (f",HSA_CU_MASK={os.environ['HSA_CU_MASK']}" if os.environ.get('HSA_CU_MASK') else ''))
        p = subprocess.run([str(REPO / 'scripts/amd/vllm_in_container.sh'), '-L'], capture_output=True, text=True, env=env, timeout=600)
    else:
        p = subprocess.run(['rocprofv3', '-L'], capture_output=True, text=True, timeout=600)
    return p.stdout + p.stderr


def dispatch_rows(d):
    rows = collections.OrderedDict()
    for f in sorted(d.rglob('*counter_collection.csv')):
        for r in csv.DictReader(open(f)):
            if 'wvSplitK' not in r['Kernel_Name']:
                continue
            did = int(r.get('Dispatch_Id') or r.get('Correlation_Id'))
            e = rows.setdefault(did, {'dispatch_id': did, 'kernel': r['Kernel_Name'][:120], 'grid': r.get('Grid_Size'),
                                      'workgroup': r.get('Workgroup_Size'), 'counters': collections.Counter()})
            e['counters'][r['Counter_Name']] += float(r['Counter_Value'])
    return [dict(v, counters=dict(v['counters'])) for _, v in sorted(rows.items())]


def parse(d, args_kernel_prefix=None):
    per = collections.OrderedDict()
    for f in sorted(d.rglob('*counter_collection.csv')):
        for r in csv.DictReader(open(f)):
            name = r['Kernel_Name']
            if args_kernel_prefix and not name.replace('void ', '', 1).startswith(args_kernel_prefix) and 'wvSplitK' not in name:
                continue
            k = per.setdefault(name[:160], {'dispatch_ids': set(), 'counters': collections.Counter(), 'duration_ns': 0,
                                            'grid': r.get('Grid_Size'), 'workgroup': r.get('Workgroup_Size'),
                                            'lds': r.get('LDS_Block_Size') or r.get('Lds_Size'),
                                            'vgpr': r.get('Arch_VGPR_Count') or r.get('VGPR_Count')})
            did = r.get('Dispatch_Id') or r.get('Correlation_Id')
            if did not in k['dispatch_ids']:
                k['dispatch_ids'].add(did)
                try:
                    k['duration_ns'] += int(r['End_Timestamp']) - int(r['Start_Timestamp'])
                except (KeyError, ValueError):
                    pass
            k['counters'][r['Counter_Name']] += float(r['Counter_Value'])
    return {n: {'dispatches': len(v['dispatch_ids']), 'counters': dict(v['counters']), 'duration_ns': v['duration_ns'],
                'grid': v['grid'], 'workgroup': v['workgroup'], 'lds': v['lds'], 'vgpr': v['vgpr']} for n, v in per.items()}


def main(a):
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cus = a.cus
    res = {'classification': 'measured-real-GPU (rocprofv3 --pmc, one pass per group)', 'scope': __doc__.strip(),
           'groups': GROUPS, 'cus': cus, 'runs': [], 'listing_names': {}}
    for tgt in a.targets:
        container = tgt == 'gemv'
        lst = listing(container)
        (a.out_dir / f'rocprofv3_L_{"container" if container else "host"}.txt').write_text(lst)
        names = set(re.findall(r'\b([A-Z][A-Z0-9_]{2,})\b', lst))
        res['listing_names'][tgt] = len(names)
        loads = native_targets(cus, a.light) if tgt == 'native' else {
            'gemv_qwen32b': ['python3', 'scripts/amd/oc/oc_kernels.py'] + (a.gemv_args.split() if a.gemv_args else
                                                                          ['--models', 'qwen32b', '--grids', 'dev', '--light'])}
        for load, argv in loads.items():
            for g, ctrs in GROUPS.items():
                if a.groups and g not in a.groups:
                    continue
                if tgt == 'gemv':
                    argv = argv[:2] + [x for x in argv[2:] if x != '--out' and not x.startswith('/')]
                    argv = argv + ['--out', str(a.out_dir / f'calls_{g}.json')]
                have = [c for c in ctrs if c in names]
                miss = [c for c in ctrs if c not in names]
                run = {'target': tgt, 'load': load, 'group': g, 'counters': have, 'not_listed': miss}
                if not have:
                    run['returncode'] = None
                    res['runs'].append(run)
                    continue
                d = a.out_dir / 'rocprof' / f'{load}_{g}'
                pre = ['--pmc'] + have + ['--kernel-trace', '-d', str(d), '-o', 'p', '--output-format', 'csv']
                if container:
                    cmd = [str(REPO / 'scripts/amd/vllm_in_container.sh')] + pre + ['--'] + argv
                    env = dict(os.environ, AMD_ENTRY='rocprofv3',  # --cleanenv drops HIP_VISIBLE_DEVICES
                               AMD_EXTRA_ENV=f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '0')}"
                + (f",HSA_CU_MASK={os.environ['HSA_CU_MASK']}" if os.environ.get('HSA_CU_MASK') else ''))
                else:
                    cmd = ['rocprofv3'] + pre + ['--', str(a.bench)] + argv + ['--reps', '1', '--trials', '1', '--warmups', '0']
                    env = dict(os.environ)
                try:
                    p = subprocess.run(['timeout', '-s', 'KILL', str(a.timeout)] + cmd, capture_output=True, text=True, env=env, cwd=REPO)
                    run.update(returncode=p.returncode, stderr_tail=p.stderr[-600:] if p.returncode else '',
                               stdout_tail=p.stdout[-400:] if tgt == 'native' else '')
                except Exception as error:
                    run.update(returncode=-1, stderr_tail=str(error)[:400])
                run['args'] = argv
                run['kernels'] = parse(d, 'k_' if tgt == 'native' else None)
                if tgt == 'gemv':
                    run['dispatch_rows'] = dispatch_rows(d)
                    cf = a.out_dir / f'calls_{g}.json'
                    run['call_order'] = json.loads(cf.read_text()).get('call_order') if cf.exists() else None
                res['runs'].append(run)
                print(json.dumps({k: run.get(k) for k in ('target', 'load', 'group', 'returncode')}), flush=True)
                (a.out_dir / 'oc_counters.json').write_text(json.dumps(res, indent=1, default=list) + '\n')
    (a.out_dir / 'oc_counters.json').write_text(json.dumps(res, indent=1, default=list) + '\n')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--out-dir', type=Path, required=True)
    q.add_argument('--bench', type=Path, required=True)
    q.add_argument('--cus', type=int, required=True, help='device CU count (blocks of the native loads)')
    q.add_argument('--targets', type=lambda v: v.split(','), default=['native', 'gemv'])
    q.add_argument('--timeout', type=int, default=300)
    q.add_argument('--light', action='store_true')
    q.add_argument('--groups', type=lambda v: v.split(','), default=None)
    q.add_argument('--gemv-args', default='', help='oc_kernels.py arguments for the gemv target (default: qwen32b --light)')
    main(q.parse_args())
