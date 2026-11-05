#!/usr/bin/env python3
"""OC0 capability probe for time-domain activity observables (one GPU, HIP_VISIBLE_DEVICES set; host side).

Tries, and records rather than assumes, each route the framework asks about:

  pc_host_trap / pc_stochastic   rocprofv3 PC sampling (beta) on the decode projection route
                                 (oc_kernels.py --light, container rocprofv3); configs listed by -L are saved first
  att_gemv                       advanced thread trace (--att, one CU, SIMD mask 0xF, --att-activity 8) on the same
                                 route, container rocprofv3 with its bundled trace decoder
  att_native_<load>              thread trace on the known-schedule controls (svc_bench v7 split / fused / mfma4 /
                                 read), host rocprofv3 with --att-library-path set to a copy of the container's decoder
                                 ($WORK/lib/rocprof-trace-decoder, sha256 recorded)

Per route: command, return code, wall time, stderr tail, every output file with its size, and a short head of each
CSV/JSON output. No estimator is built here; the OC4 decision rule (execution plan) reads this file.

Development mode (--dev): no PC sampling; ATT on each development projection shape separately (oc_kernels.py --count,
one call per grid, so the decoded dispatch is that shape) and on split variants that put both workgroup types on the
traced CU (4 and 8 workgroups per CU, target CUs 0 and 1), plus fused and read as controls.

  oc_probe.py --out-dir <dir> --bench <svc_bench_v7> --cus <n>
  oc_probe.py --out-dir <dir> --bench <b> --cus <n> --dev --gemv-shapes qwen32b:qkv,qwen1p5b:down --grids 304
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get('WORK', '/work1/proj_blaisetine/yanggon'))
DECODER = WORK / 'lib/rocprof-trace-decoder'
GEMV = ['python3', 'scripts/amd/oc/oc_kernels.py', '--models', 'qwen32b', '--grids', 'dev', '--light', '--out', '/dev/null']


def run(name, cmd, out, env=None, timeout=900):
    t0 = time.time()
    try:
        p = subprocess.run(['timeout', '-s', 'KILL', str(timeout)] + cmd, capture_output=True, text=True, env=env, cwd=REPO)
        rc, err, so = p.returncode, p.stderr, p.stdout
    except Exception as error:
        rc, err, so = -1, str(error), ''
    files = []
    if out.exists():
        for f in sorted(out.rglob('*')):
            if f.is_file():
                e = {'path': str(f.relative_to(out)), 'bytes': f.stat().st_size}
                if f.suffix in ('.csv', '.json', '.txt') and f.stat().st_size:
                    with open(f, errors='replace') as h:
                        e['head'] = h.read(600)
                files.append(e)
    return {'name': name, 'cmd': cmd, 'returncode': rc, 'wall_s': round(time.time() - t0, 1), 'stderr_tail': err[-1500:],
            'stdout_tail': so[-600:], 'files': files[:200], 'n_files': len(files)}


def att_args(cu):
    return ['--att', '--att-target-cu', str(cu), '--att-simd-select', '0xF', '--att-activity', '8']


def dev(a, res, wrap, cenv):
    for spec in a.gemv_shapes:
        model, shape = spec.split(':')
        d = a.out_dir / f'att_gemv_{model}_{shape}'
        res['routes'].append(run(d.name, wrap + att_args(1) + ['--kernel-include-regex', 'wvSplitK', '-d', str(d), '-o', 'p',
                                                              '--output-format', 'csv', '--', 'python3', 'scripts/amd/oc/oc_kernels.py',
                                                              '--models', model, '--shapes', shape, '--grids', a.grids, '--count',
                                                              '--out', str(d / 'calls.json')], d, env=cenv))
    G = str(1 << 28)
    natives = {f'split_b{k}_cu{cu}': (cu, ['--mode', 'split', '--bytes', G, '--iters', '512', '--blocks', str(k * a.cus), '--threads', '1024'])
               for k in (4, 8) for cu in (0, 1)}
    natives['fused'] = (1, ['--mode', 'fused', '--bytes', G, '--intensity', '8', '--blocks', str(a.cus), '--threads', '1024'])
    natives['read'] = (1, ['--mode', 'read', '--bytes', G, '--blocks', str(a.cus), '--threads', '1024'])
    for load, (cu, args) in natives.items():
        d = a.out_dir / f'att_native_{load}'
        res['routes'].append(run(d.name, ['rocprofv3'] + att_args(cu) + ['--att-library-path', str(DECODER), '-d', str(d), '-o', 'p',
                                                                         '--output-format', 'csv', '--', str(a.bench)] + args
                                 + ['--reps', '1', '--trials', '1', '--warmups', '0'], d, timeout=600))


def main(a):
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cenv = dict(os.environ, AMD_ENTRY='rocprofv3',  # --cleanenv drops HIP_VISIBLE_DEVICES; pass it explicitly
                AMD_EXTRA_ENV=f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '0')}"
                + (f",HSA_CU_MASK={os.environ['HSA_CU_MASK']}" if os.environ.get('HSA_CU_MASK') else ''))
    wrap = [str(REPO / 'scripts/amd/vllm_in_container.sh')]
    res = {'scope': __doc__.strip(), 'routes': [], 'decoder_copy': {}}
    for f in sorted(DECODER.glob('*')):
        res['decoder_copy'][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    if a.dev:
        dev(a, res, wrap, cenv)
        for r in res['routes']:
            print(json.dumps({k: r[k] for k in ('name', 'returncode', 'wall_s', 'n_files')}), flush=True)
        (a.out_dir / 'oc_probe.json').write_text(json.dumps(res, indent=1) + '\n')
        return
    res['container_L'] = run('container_L', wrap + ['-L'], a.out_dir / 'none', env=cenv)
    (a.out_dir / 'container_L.txt').write_text(res['container_L'].pop('stdout_tail') + res['container_L']['stderr_tail'])
    for meth, unit, interval in (('host_trap', 'time', '100'), ('stochastic', 'cycles', '1048576')):
        d = a.out_dir / f'pc_{meth}'
        res['routes'].append(run(f'pc_{meth}', wrap + ['--pc-sampling-beta-enabled', '--pc-sampling-method', meth,
                                                       '--pc-sampling-unit', unit, '--pc-sampling-interval', interval,
                                                       '-d', str(d), '-o', 'p', '--output-format', 'csv', '--'] + GEMV, d, env=cenv))
    d = a.out_dir / 'att_gemv'
    res['routes'].append(run('att_gemv', wrap + ['--att', '--att-target-cu', '1', '--att-simd-select', '0xF', '--att-activity', '8',
                                                 '--kernel-include-regex', 'wvSplitK', '-d', str(d), '-o', 'p',
                                                 '--output-format', 'csv', '--'] + GEMV, d, env=cenv))
    G = str(1 << 28)
    natives = {'split': ['--mode', 'split', '--bytes', G, '--iters', '4096', '--blocks', str(2 * a.cus), '--threads', '1024'],
               'fused': ['--mode', 'fused', '--bytes', G, '--intensity', '8', '--blocks', str(a.cus), '--threads', '1024'],
               'mfma4': ['--mode', 'mfma4', '--iters', '1024', '--blocks', str(a.cus), '--threads', '1024'],
               'read': ['--mode', 'read', '--bytes', G, '--blocks', str(a.cus), '--threads', '1024']}
    for load, args in natives.items():
        d = a.out_dir / f'att_native_{load}'
        res['routes'].append(run(f'att_native_{load}', ['rocprofv3', '--att', '--att-library-path', str(DECODER), '--att-target-cu', '1',
                                                        '--att-simd-select', '0xF', '--att-activity', '8', '-d', str(d), '-o', 'p',
                                                        '--output-format', 'csv', '--', str(a.bench)] + args
                                 + ['--reps', '1', '--trials', '1', '--warmups', '0'], d, timeout=600))
    for r in res['routes']:
        print(json.dumps({k: r[k] for k in ('name', 'returncode', 'wall_s', 'n_files')}), flush=True)
    (a.out_dir / 'oc_probe.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--out-dir', type=Path, required=True)
    q.add_argument('--bench', type=Path, required=True)
    q.add_argument('--cus', type=int, required=True)
    q.add_argument('--dev', action='store_true')
    q.add_argument('--gemv-shapes', type=lambda v: v.split(','), default=[])
    q.add_argument('--grids', default='dev')
    main(q.parse_args())
