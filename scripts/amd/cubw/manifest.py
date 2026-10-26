#!/usr/bin/env python3
"""E0 manifests for a cubw_v1 campaign root: environment.json, source_manifest.json, case_manifest.json.

environment.json   node, GPUs (physical index, UUID, PCI, CU count, arch), driver/ROCm/tool versions,
                   container image digest, monitor/runtime settings. Observation only; nothing is changed.
source_manifest    SHA-256 of every file in the release snapshot the job runs from, the git commit,
                   configs (including derived ones) and model snapshot metadata. Model weights are
                   identified by immutable revision directory, file names and sizes; small files
                   (config, tokenizer, index) are hashed. Weight bytes are NOT rehashed per job.
case_manifest      every planned step with its lane/stage/cell/acquisition/treatment and output path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def sha(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def cmd(c, timeout=120):
    try:
        p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()[-4000:]
    except Exception as error:
        return f'unavailable: {error}'


def main(args):
    root, rel = args.root, args.release
    plan = json.loads((root / 'launch_plan.json').read_text())
    env = {'hostname': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
           'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'), 'kernel': platform.release(),
           'gpus_tsv': (root / 'gpus.tsv').read_text() if (root / 'gpus.tsv').exists() else None,
           'preflight': json.loads((root / 'preflight.json').read_text()) if (root / 'preflight.json').exists() else None,
           'rocm': cmd('cat /opt/rocm/.info/version 2>/dev/null; hipcc --version 2>&1 | head -3'),
           'amd_smi_version': cmd('amd-smi version'),
           'amd_smi_static_gpu0': cmd('amd-smi static -g 0 --asic --board --vbios --limit --driver 2>&1 | head -80'),
           'rocprofv3_host': cmd('rocprofv3 --version'),
           'container': {'image': os.environ.get('AMD_VLLM_SIF', 'default (scripts/amd/vllm_in_container.sh)'),
                         'sha256_record': cmd(f"cat {os.environ.get('WORK', '')}/containers/sha256.txt")},
           'bench_sha256': {p.name: sha(p) for p in Path(os.environ.get('WORK', '.'), 'bin').glob('svc_bench_gfx942*')
                            if p.is_file()},
           'settings_scope': 'observed only; no clock, power, driver, firmware or partition changes'}
    (root / 'environment.json').write_text(json.dumps(env, indent=1) + '\n')
    src = {'release_dir': str(rel), 'git_commit': (rel / 'RELEASE_COMMIT').read_text().strip()
           if (rel / 'RELEASE_COMMIT').exists() else None, 'files': {}}
    for p in sorted(rel.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            src['files'][str(p.relative_to(rel))] = sha(p)
    src['configs'] = {str(p.relative_to(root)): sha(p) for d in ('configs', 'configs_energy')
                      for p in sorted((root / d).glob('*.json'))}
    models = {}
    for p in sorted((root / 'configs').glob('*_base.json')):
        c = json.loads(p.read_text())
        mp = Path(c['model_path'])
        files = {f.name: f.stat().st_size for f in sorted(mp.iterdir()) if f.is_file() or f.is_symlink()}
        small = {f.name: sha(f) for f in sorted(mp.iterdir())
                 if f.suffix in ('.json', '.txt', '.model', '.py') or f.name.startswith('tokenizer')}
        models[c['model_id']] = {'revision': c['model_revision'], 'path': str(mp), 'file_sizes': files,
                                 'small_file_sha256': small, 'weights_rehashed': False}
    src['models'] = models
    src['plan_sha256'] = plan.get('plan_sha256')
    (root / 'source_manifest.json').write_text(json.dumps(src, indent=1) + '\n')
    cases = [{k: s.get(k) for k in ('id', 'lane', 'stage', 'experiment', 'kind', 'cell', 'acquisition', 'block', 'model',
                                     'decoder_cus', 'treatment', 'mode', 'config', 'out', 'requires', 'est_s', 'timeout_s')}
             for s in plan['steps']]
    (root / 'case_manifest.json').write_text(json.dumps({'plan_sha256': plan.get('plan_sha256'), 'cases': cases}, indent=1) + '\n')
    print(f"manifests: {len(src['files'])} release files, {len(src['configs'])} configs, {len(models)} models, {len(cases)} cases")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--root', type=Path, required=True)
    q.add_argument('--release', type=Path, required=True)
    main(q.parse_args())
