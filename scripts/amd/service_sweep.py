#!/usr/bin/env python3
"""M4 / A-D independent service controls, repeated under CU masks for M7 (one GPU).

Each condition runs as three independent process acquisitions (10 trials, 3 warmups each).
One extra acquisition per condition runs under rocprofv3 to record physical L2->memory
read/write bytes (FETCH_SIZE and WRITE_SIZE in separate passes: requesting both at once
exceeds the hardware counter budget on gfx942 and hangs rocprofv3).
CU restriction uses HSA_CU_MASK=<physical index>:0-(k-1). Contiguous CU indices are
interleaved across the 8 XCDs on MI300X, so k a multiple of 8 restricts every XCD equally.
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

GiB, MiB = 1 << 30, 1 << 20


def conditions():
    c = []
    for size, passes in [(2 * MiB, 400), (8 * MiB, 100), (32 * MiB, 25), (128 * MiB, 8), (256 * MiB, 4),
                         (512 * MiB, 2), (GiB, 1), (4 * GiB, 1), (16 * GiB, 1)]:
        c.append(('read', dict(bytes=size, passes=passes, reps=5 if size >= GiB else 10)))
    c += [('write', dict(bytes=4 * GiB, reps=5)), ('copy', dict(bytes=4 * GiB, reps=5))]
    c += [('strided', dict(bytes=4 * GiB, stride=s, reps=5)) for s in (2, 4, 8, 16)]
    c += [('gather', dict(bytes=b, reps=5)) for b in (64 * MiB, GiB)]
    c += [('chase', dict(bytes=b, chains=1, steps=100000)) for b in (MiB, 3 * MiB, 16 * MiB, 128 * MiB, 512 * MiB, 4 * GiB)]
    c += [('chase', dict(bytes=4 * GiB, chains=k, steps=20000)) for k in (8, 64, 512, 4096, 32768)]
    c += [('fma', dict(iters=20000, reps=1)), ('mfma', dict(iters=20000, reps=1))]
    c += [('mixed', dict(bytes=4 * GiB, intensity=k, reps=3)) for k in (0, 4, 16, 64, 256)]
    return c


def condition_id(mode, opts, cus):
    return f"{mode}_" + '_'.join(f'{k}{v}' for k, v in sorted(opts.items())) + f'_cu{cus}'


def run(binary, mode, opts, env, timeout, prefix=()):
    cmd = list(prefix) + [binary, '--mode', mode, '--trials', '10', '--warmups', '3'] + \
          [x for k, v in opts.items() for x in (f'--{k}', str(v))]
    t = time.time_ns()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return cmd, t, p


def counter_sum(directory, kernel_prefix):
    total = {}
    for f in glob.glob(f'{directory}/**/*counter_collection.csv', recursive=True):
        for r in csv.DictReader(open(f)):
            if r['Kernel_Name'].startswith(kernel_prefix):
                total[r['Counter_Name']] = total.get(r['Counter_Name'], 0.0) + float(r['Counter_Value'])
    return total


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / 'service.jsonl'
    done = set()
    if out.exists():
        done = {(r['condition_id'], r['acquisition']) for r in map(json.loads, out.read_text().splitlines())}
    base = dict(os.environ, HIP_VISIBLE_DEVICES=str(args.physical_index))
    base.pop('HSA_CU_MASK', None)
    for cus in args.cus:
        env = dict(base)
        if cus < 304:
            env['HSA_CU_MASK'] = f'{args.physical_index}:0-{cus - 1}'
        for mode, opts in conditions():
            cid = condition_id(mode, opts, cus)
            for acq in list(range(3)) + ['counters']:
                if (cid, acq) in done:
                    continue
                row = {'condition_id': cid, 'acquisition': acq, 'cus_requested': cus,
                       'hsa_cu_mask': env.get('HSA_CU_MASK', ''), 'gpu_uuid': args.gpu_uuid,
                       'physical_index': args.physical_index}
                try:
                    if acq != 'counters':
                        cmd, t0, p = run(args.binary, mode, opts, env, 600)
                        row.update(command=cmd, start_unix_ns=t0, end_unix_ns=time.time_ns(), returncode=p.returncode,
                                   stderr=p.stderr[-500:])
                        if p.returncode == 0:
                            row['result'] = json.loads(p.stdout.strip().splitlines()[-1])
                    else:
                        row['counters'] = {}
                        for counter in ('FETCH_SIZE', 'WRITE_SIZE'):
                            d = args.output / 'pmc' / f'{cid}_{counter}'
                            shutil.rmtree(d, ignore_errors=True)
                            prefix = ['timeout', '-s', 'KILL', '300', 'rocprofv3', '--pmc', counter, '-d', str(d),
                                      '-o', 'pmc', '--output-format', 'csv', '--']
                            cmd, t0, p = run(args.binary, mode, opts, env, 330, prefix)
                            sums = counter_sum(d, 'k_')
                            row['counters'][counter] = {'returncode': p.returncode, 'kb_sum_over_k_kernels': sums.get(counter),
                                                        'scope': 'all trials+warmups of one acquisition; L2->memory bytes (KB), not post-MALL HBM bytes'}
                            if p.returncode == 0:
                                row['counters'][counter]['result'] = json.loads(p.stdout.strip().splitlines()[-1])
                except Exception as error:
                    row['error'] = f'{type(error).__name__}: {error}'
                with out.open('a') as f:
                    f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--physical-index', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cus', type=int, nargs='+', default=[304, 232, 152, 96])
    main(p.parse_args())
