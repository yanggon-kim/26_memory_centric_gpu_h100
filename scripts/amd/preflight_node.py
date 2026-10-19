#!/usr/bin/env python3
"""Preflight for an unattended AMD campaign job: verify the allocated node before spending hours on it.

Writes <root>/gpus.tsv (physical index, GPU ID, PCI BDF) and <root>/preflight.json. Exits non-zero
when a check that every lane depends on fails, so the batch job stops early instead of billing for
broken runs. Run in the host module environment (scripts/amd/env.sh) on the compute node.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sh(cmd, timeout=300, env=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


def main(args):
    repo = Path(__file__).resolve().parents[2]
    work = os.environ.get('WORK', '/work1/proj_blaisetine/yanggon')
    checks, fatal = {}, []

    import amd_nvml as n
    n.nvmlInit()
    gpus = []
    for i in range(n.nvmlDeviceGetCount()):
        h = n.nvmlDeviceGetHandleByIndex(i)
        gpus.append({'index': n.nvmlDeviceGetIndex(h), 'uuid': n.nvmlDeviceGetUUID(h), 'name': n.nvmlDeviceGetName(h),
                     'bdf': n.nvmlDeviceGetPciInfo(h).busId,
                     'processes': [p for p in n.processes_detail(h) if p['vram_bytes'] >= 64 << 20 or p['gfx_engine_ns'] > 0]})
    checks['gpus'] = gpus
    if len(gpus) != args.expect_gpus:
        fatal.append(f'expected {args.expect_gpus} GPUs, found {len(gpus)}')
    if any(args.expect_name not in g['name'] for g in gpus):
        fatal.append(f'GPU name mismatch: {sorted({g["name"] for g in gpus})}')
    if any(g['processes'] for g in gpus):
        fatal.append('foreign GPU activity present at start')
    (args.root / 'gpus.tsv').write_text(''.join(f"{g['index']}\t{g['uuid']}\t{g['bdf']}\n" for g in sorted(gpus, key=lambda g: g['index'])))

    import torch
    props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    checks['torch'] = {'devices': len(props), 'arch': sorted({p.gcnArchName for p in props}),
                       'cus': sorted({p.multi_processor_count for p in props})}
    if len(props) != args.expect_gpus or any(not p.gcnArchName.startswith('gfx942') for p in props) \
            or any(p.multi_processor_count != 304 for p in props):
        fatal.append(f"torch view unexpected: {checks['torch']}")

    rc, out, err = sh(f"{repo}/scripts/amd/vllm_in_container.sh -c 'import torch, vllm; "
                      f"print(vllm.__version__, torch.cuda.device_count())'", timeout=600)
    checks['container'] = {'returncode': rc, 'stdout': out.strip()[-200:], 'stderr': err.strip()[-300:]}
    if rc != 0 or not out.strip().endswith(str(args.expect_gpus)):
        fatal.append('vLLM container check failed')

    missing = []
    for cfg in sorted((args.root / 'configs').glob('*_base.json')):
        path = Path(json.loads(cfg.read_text())['model_path'])
        if not any(path.glob('*.safetensors')):
            missing.append(str(path))
    checks['models_missing'] = missing
    if missing:
        fatal.append(f'model snapshots missing: {missing}')

    for b in ('svc_bench_gfx942', 'svc_bench_gfx942_v2'):
        if not os.access(f'{work}/bin/{b}', os.X_OK):
            fatal.append(f'missing binary {b}')

    env = dict(os.environ, HIP_VISIBLE_DEVICES='0')
    d = args.root / 'preflight_pmc'
    rc, out, err = sh(f"timeout -s KILL 120 rocprofv3 --pmc FETCH_SIZE -d {d} -o p --output-format csv -- "
                      f"{work}/bin/svc_bench_gfx942 --mode copy --bytes 67108864 --reps 1 --trials 1 --warmups 0", env=env)
    fetched = None
    try:
        import csv
        f = next(d.rglob('*counter_collection.csv'))
        fetched = sum(float(r['Counter_Value']) for r in csv.DictReader(open(f)) if r['Kernel_Name'].startswith('k_copy'))
    except StopIteration:
        pass
    checks['counter'] = {'returncode': rc, 'fetch_kb_k_copy': fetched, 'expected_kb': 65536}
    if not fetched or not 0.9 * 65536 <= fetched <= 1.2 * 65536:
        fatal.append(f'counter check failed (FETCH_SIZE {fetched} KB for a 64 MiB copy)')

    def fma(mask):
        e = dict(env)
        if mask:
            e['HSA_CU_MASK'] = mask
        rc, out, _ = sh(f"timeout -s KILL 60 {work}/bin/svc_bench_gfx942 --mode fma --iters 20000 --trials 3 --reps 1", env=e)
        return json.loads(out.strip().splitlines()[-1])['useful_tflop_per_s'] if rc == 0 else None
    full, half = fma(''), fma('0:0-151')
    checks['cu_mask'] = {'fma_full_tflops': full, 'fma_152cu_tflops': half}
    if not full or not half or half / full > 0.8:
        fatal.append(f'CU mask check failed ({full} vs {half} TFLOP/s)')

    checks['fatal'] = fatal
    (args.root / 'preflight.json').write_text(json.dumps(checks, indent=1) + '\n')
    print(json.dumps({'fatal': fatal, 'gpus': len(gpus), 'container': checks['container']['stdout'],
                      'counter_kb': fetched, 'cu_mask': checks['cu_mask']}))
    sys.exit(1 if fatal else 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expect-gpus', type=int, default=8)
    p.add_argument('--expect-name', default='MI325X')
    main(p.parse_args())
