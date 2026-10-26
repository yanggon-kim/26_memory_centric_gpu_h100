#!/usr/bin/env python3
"""E5 counter capability and unit validation on known native traffic (cubw_v1).

Lists the counters rocprofv3 exposes on this node, selects memory-hierarchy candidates (pre-MALL
FETCH/WRITE, TCC hit/miss, TCC->EA/fabric requests, anything named MALL/DRAM/HBM/DF), and collects each
one, one counter per run, on three svc_bench_v3 reads with known useful bytes:
  4 GiB streaming read (above the 256 MiB MALL), 128 MiB x 16 passes (MALL-resident, above L2),
  8 MiB x 256 passes (L2-resident per XCD).
The ratio counter/useful bytes is reported raw; units and scope are declared only where the ratio
matches a unit on every footprint. Nothing is inferred from an assumed hit rate.
Run on the compute node with HIP_VISIBLE_DEVICES set (host rocprofv3).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess

WORK = Path(os.environ.get('WORK', '/work1/proj_blaisetine/yanggon'))
BIN = Path(os.environ.get('CUBW_BENCH', WORK / 'bin/svc_bench_gfx942_v3'))
PATTERN = re.compile(r'(MALL|DRAM|HBM|^DF_|EA0?_RD|EA0?_WR|TCC_EA|RDREQ|WRREQ|^FETCH_SIZE$|^WRITE_SIZE$|^TCC_HIT$|^TCC_MISS$)')
LOADS = [('read4g', 4 << 30, 1), ('reuse128m', 128 << 20, 16), ('reuse8m', 8 << 20, 256)]


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(['rocprofv3', '-L'], capture_output=True, text=True, timeout=300)
    (args.out_dir / 'rocprofv3_L.txt').write_text(listing.stdout + listing.stderr)
    names = sorted({w for w in re.findall(r'\b([A-Z][A-Z0-9_]{3,})\b', listing.stdout) if PATTERN.search(w)})
    # derived "_sum" forms aggregate instances; prefer them, keep raw names when no sum exists
    chosen = [n for n in names if n.endswith('_sum') or not any(m == n + '_sum' for m in names)][:args.max_counters]
    result = {'classification': 'measured-real-GPU (native known traffic)', 'scope': __doc__.strip(),
              'candidates_found': names, 'counters_tested': chosen, 'runs': []}
    for counter in chosen:
        for load, nbytes, passes in LOADS:
            d = args.out_dir / f'{counter}_{load}'
            cmd = ['timeout', '-s', 'KILL', '120', 'rocprofv3', '--pmc', counter, '-d', str(d), '-o', 'p',
                   '--output-format', 'csv', '--', str(BIN), '--mode', 'read', '--bytes', str(nbytes), '--passes',
                   str(passes), '--reps', '1', '--trials', '1', '--warmups', '0']
            p = subprocess.run(cmd, capture_output=True, text=True)
            total, rows = None, 0
            for f in d.rglob('*counter_collection.csv'):
                for r in csv.DictReader(open(f)):
                    if r['Kernel_Name'].startswith('k_read'):
                        total = (total or 0) + float(r['Counter_Value']); rows += 1
            useful = nbytes * passes
            run = {'counter': counter, 'load': load, 'useful_bytes': useful, 'returncode': p.returncode,
                   'k_read_dispatch_rows': rows, 'counter_total': total,
                   'ratio_to_useful_bytes': total / useful if total else None,
                   'ratio_if_KB': total * 1024 / useful if total else None,
                   'ratio_if_64B_requests': total * 64 / useful if total else None,
                   'ratio_if_32B_requests': total * 32 / useful if total else None,
                   'stderr_tail': p.stderr[-300:] if p.returncode else ''}
            result['runs'].append(run)
            print(json.dumps(run), flush=True)
    (args.out_dir / 'counter_probe.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--out-dir', type=Path, required=True)
    q.add_argument('--max-counters', type=int, default=14)
    main(q.parse_args())
