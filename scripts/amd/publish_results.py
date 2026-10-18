#!/usr/bin/env python3
"""Publish compact AMD evidence into Git and index the raw evidence by checksum.

Copies summaries, configs, analysis outputs, environment/inventory and small per-family JSONL files
from each result root into results_published/<root-name>/ (tracked), and writes
docs/amd/artifacts.md plus results_published/<root-name>/raw_manifest.sha256 covering every raw
file left outside Git (run.json, logs, rocprof outputs). Raw files are not modified.
"""
import argparse
import hashlib
from pathlib import Path
import shutil

SMALL = ['configs*/*.json', 'j0/environment.json', 'j0/amd_smi_static_gpu.txt', 'j0/amd_smi_topology.txt',
         'j0/rocprofv3_avail.txt', 'environment.json', 'selected_gpu.txt',
         '*/summary.json', '*/summary.txt', 'analysis/*', 'm4_service/service.jsonl', 'm5_pressure/*_a[0-9].json',
         'm2_profile/*/trace/analysis.json', 'm1_profile/*/analysis.json', 'm3_rocprof_compute/*.analysis.txt',
         'm3_rocprof_compute/lane_sets.log', '*/lane*.log', 'tp/lane_*.log', 'm2_counters/lane_*.log',
         'm2_counters_watchdog.log']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    repo = Path(__file__).resolve().parents[2]
    lines = ['# AMD artifact index', '',
             'Compact evidence is tracked under `results_published/`. Raw evidence stays on the cluster',
             '(`/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/...`, or `$WORK/results_raw/...` once moved)',
             'and is identified by the SHA256 manifests listed here. Raw profiler databases can embed environment data;',
             'they are not published.', '']
    for root in args.roots:
        root = root.resolve()
        dest = repo / 'results_published' / root.name
        dest.mkdir(parents=True, exist_ok=True)
        copied = set()
        for pattern in SMALL:
            for f in root.glob(pattern):
                if f.is_file() and f.stat().st_size < 20 << 20:
                    rel = f.relative_to(root)
                    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest / rel)
                    copied.add(rel)
        manifest = dest / 'raw_manifest.sha256'
        total = 0
        with manifest.open('w') as m:
            for f in sorted(p for p in root.rglob('*') if p.is_file()):
                rel = f.relative_to(root)
                if rel in copied:
                    continue
                m.write(f'{sha(f)}  {rel}\n')
                total += f.stat().st_size
        lines += [f'## {root.name}', '', f'- Published compact files: {len(copied)} (under `results_published/{root.name}/`)',
                  f'- Raw files indexed: `results_published/{root.name}/raw_manifest.sha256` ({total / 1e9:.2f} GB)',
                  f'- Raw location: `{root}`', '']
    (repo / 'docs/amd/artifacts.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', type=Path, nargs='+')
    main(p.parse_args())
