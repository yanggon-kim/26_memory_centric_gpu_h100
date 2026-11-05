#!/usr/bin/env python3
"""G4 traffic-observation audit (goal follow-up, docs/amd/goal_based_followup_20261105.md §6). Host rocprofv3 on the
compute node, HIP_VISIBLE_DEVICES set, one GPU.

1. Inventory: rocprofv3 version and counter list (-L), the installed counter definitions file(s) found under the ROCm
   tree, and amd-smi firmware (rocprofv3 refused MI325X counters with CP firmware 177 < 186, job 492957).
2. Candidates: every listed counter whose name matches memory-side patterns (FETCH/WRITE_SIZE, TCC_EA*_RD/WRREQ*,
   *DRAM*, *MALL*, *HBM*, DF_*, TCC_HIT/MISS), preferring the aggregated "_sum" form when both exist.
3. Known traffic (svc_bench_v3, one kernel per run, cold start; buffer initialisation is a separate dispatch and is
   excluded by kernel name): reads at 8 MiB x 256 passes (L2-resident), 128 MiB x 16 (MALL-resident, above L2),
   1 GiB x 4 and 4 GiB x 1 (streaming, far above the 256 MiB MALL); writes of 128 MiB and 4 GiB; copies of 128 MiB and
   2 GiB (read + write). Each (counter, load) runs --reps times, one counter per rocprofv3 pass.
4. Per run: counter total over the k_<mode> dispatch rows, useful bytes read and written, and the raw ratio under
   each candidate unit (bytes, KiB, 32/64/128-byte requests). A counter is a physical-HBM candidate only if, for
   streaming loads, ratio ~ 1 in some unit for reads (or writes) AND it falls toward the compulsory footprint for the
   cache-resident loads (expected direction: counts << useful bytes). The judgment is written by
   scripts/amd/goal/analyze_goal.py from this raw file; nothing here declares a unit.

  counter_audit.py --out-dir <dir> [--max-counters 16 --reps 2 --bench <svc_bench_v3>]
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path
import re
import subprocess

WORK = Path(os.environ.get('WORK', '/work1/proj_blaisetine/yanggon'))
PATTERN = re.compile(r'(MALL|DRAM|HBM|^DF_|TCC_EA0?_RD|TCC_EA0?_WR|RDREQ|WRREQ|^FETCH_SIZE$|^WRITE_SIZE$|'
                     r'^TCC_HIT(_sum)?$|^TCC_MISS(_sum)?$|^TCC_REQ(_sum)?$)')
NOT_VOLUME = re.compile(r'(STALL|LEVEL|BUSY|FIFO|CREDIT|PROBE_COMMAND|_PERF_SEL_)')
G = 1 << 30
M = 1 << 20
LOADS = [  # name, mode, bytes per buffer, passes; useful read/write bytes derived below
    ('read8m_x256', 'read', 8 * M, 256), ('read128m_x16', 'read', 128 * M, 16), ('read1g_x4', 'read', G, 4),
    ('read4g_x1', 'read', 4 * G, 1), ('write128m', 'write', 128 * M, 1), ('write4g', 'write', 4 * G, 1),
    ('copy128m', 'copy', 128 * M, 1), ('copy2g', 'copy', 2 * G, 1)]


def useful(mode, nbytes, passes):
    return {'read': (nbytes * passes, 0), 'write': (0, nbytes), 'copy': (nbytes, nbytes)}[mode]


def sh(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return f'TIMEOUT {cmd}'


def main(a):
    a.out_dir.mkdir(parents=True, exist_ok=True)
    inv = {'rocprofv3_version': sh('rocprofv3 --version'), 'amd_smi_firmware': sh('amd-smi firmware -g $HIP_VISIBLE_DEVICES'),
           'amd_smi_static': sh('amd-smi static -g $HIP_VISIBLE_DEVICES -a -b -v'),
           'rocm_path': os.environ.get('ROCM_PATH', ''), 'definition_files': []}
    listing = sh('rocprofv3 -L', timeout=600)
    (a.out_dir / 'rocprofv3_L.txt').write_text(listing)
    for f in sorted(set(glob.glob('/opt/rocm*/share/rocprofiler-sdk/*.yaml') + glob.glob('/opt/rocm*/lib/rocprofiler-sdk/*.yaml')
                        + glob.glob('/opt/rocm*/share/rocprofiler-sdk/counter_defs*'))):
        inv['definition_files'].append(f)
        try:
            (a.out_dir / ('defs_' + Path(f).name)).write_text(Path(f).read_text())
        except OSError:
            pass
    names = sorted({w for w in re.findall(r'\b([A-Z][A-Z0-9_]{3,})\b', listing) if '_' in w and PATTERN.search(w)})
    # traffic-volume counters only: stall/occupancy/credit/FIFO counters cannot carry a byte count. Ordering puts the
    # derived sizes, then EA request counts by size class (gfx950 lists TCC_EA0_RDREQ_128B; rocprofv3 there exposes the
    # per-block names without a _sum alias), then DRAM-qualified requests, then L2 hit/miss; MALL/DF/HBM names follow.
    names = [n for n in names if not NOT_VOLUME.search(n)]
    base = ['FETCH_SIZE', 'WRITE_SIZE', 'TCC_EA0_RDREQ', 'TCC_EA0_RDREQ_32B', 'TCC_EA0_RDREQ_64B', 'TCC_EA0_RDREQ_128B',
            'TCC_EA0_RDREQ_DRAM', 'TCC_EA0_RDREQ_DRAM_32B', 'TCC_EA0_WRREQ', 'TCC_EA0_WRREQ_64B', 'TCC_EA0_WRREQ_DRAM',
            'TCC_EA0_WRREQ_WRITE_DRAM_32B', 'TCC_HIT', 'TCC_MISS', 'TCC_REQ', 'TCC_EA0_RD_UNCACHED_32B']
    prio = [b + '_sum' if b + '_sum' in names else b for b in base]
    rest = [n for n in names if n not in prio and not any(n == b or n == b + '_sum' for b in base)
            and (n.endswith('_sum') or n + '_sum' not in names)]
    chosen = [c for c in prio if c in names] + [n for n in rest if re.search(r'MALL|HBM|DRAM|^DF_', n)] + \
             [n for n in rest if not re.search(r'MALL|HBM|DRAM|^DF_', n)]
    chosen = chosen[:a.max_counters]
    res = {'classification': 'measured-real-GPU (native known traffic under rocprofv3 --pmc)', 'scope': __doc__.strip(),
           'inventory': inv, 'candidates_found': names, 'counters_tested': chosen, 'runs': []}
    (a.out_dir / 'counter_audit.json').write_text(json.dumps(res, indent=1) + '\n')
    for counter in chosen:
        for load, mode, nbytes, passes in LOADS:
            for rep in range(a.reps):
                d = a.out_dir / 'rocprof' / f'{counter}_{load}_r{rep}'
                args = [str(a.bench), '--mode', mode, '--bytes', str(nbytes), '--reps', '1', '--trials', '1', '--warmups', '0']
                if mode == 'read':
                    args += ['--passes', str(passes)]
                cmd = ['timeout', '-s', 'KILL', '180', 'rocprofv3', '--pmc', counter, '-d', str(d), '-o', 'p',
                       '--output-format', 'csv', '--'] + args
                p = subprocess.run(cmd, capture_output=True, text=True)
                total, rows, other = None, 0, {}
                for f in d.rglob('*counter_collection.csv'):
                    for r in csv.DictReader(open(f)):
                        if r['Kernel_Name'].startswith(f'k_{mode}'):
                            total = (total or 0) + float(r['Counter_Value']); rows += 1
                        else:
                            other[r['Kernel_Name'][:40]] = other.get(r['Kernel_Name'][:40], 0) + float(r['Counter_Value'])
                rd, wr = useful(mode, nbytes, passes)
                run = {'counter': counter, 'load': load, 'mode': mode, 'buffer_bytes': nbytes, 'passes': passes, 'rep': rep,
                       'useful_read_bytes': rd, 'useful_write_bytes': wr, 'returncode': p.returncode,
                       'dispatch_rows': rows, 'counter_total': total, 'other_dispatches': other,
                       'stderr_tail': p.stderr[-400:] if p.returncode else ''}
                ub = rd + wr
                if total and ub:
                    run['ratio'] = {u: total * s / ub for u, s in (('bytes', 1), ('KiB', 1024), ('req32', 32), ('req64', 64),
                                                                   ('req128', 128))}
                res['runs'].append(run)
                print(json.dumps({k: run[k] for k in ('counter', 'load', 'rep', 'returncode', 'counter_total')}), flush=True)
            (a.out_dir / 'counter_audit.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--out-dir', type=Path, required=True)
    q.add_argument('--max-counters', type=int, default=16)
    q.add_argument('--reps', type=int, default=2)
    q.add_argument('--bench', type=Path, default=Path(os.environ.get('CUBW_BENCH', WORK / 'bin/svc_bench_gfx942_v3')))
    main(q.parse_args())
