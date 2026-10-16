#!/usr/bin/env python3
"""Environment and supervised commands for the separately scoped H200 study."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PACKAGE / 'h200/20261016_job14073758'
WORKSPACE = Path('/tmp/h200_research_jkim4112_job14073758')
CUDA = Path('/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-13.0.1-ne4hey4ce57lmd34nixeikyfribdqln3')
UUID = 'GPU-a613c806-89b1-05d1-f71c-3d52ab46d246'
PYTHON = WORKSPACE / 'runtime/bin/python'
LAUNCH_CUTOFF = datetime.datetime.fromisoformat('2026-10-16T15:03:00-04:00').timestamp()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def environment():
    env = os.environ.copy()
    env.update(CUDA_HOME=str(CUDA), CUDA_INSTALL_PATH=str(CUDA), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES=UUID, HF_HOME=str(WORKSPACE/'cache/huggingface'),
               HF_HUB_CACHE=str(WORKSPACE/'models/hub'), HF_HUB_DISABLE_TELEMETRY='1',
               PIP_CACHE_DIR=str(WORKSPACE/'cache/pip'), XDG_CACHE_HOME=str(WORKSPACE/'cache'),
               TORCHINDUCTOR_CACHE_DIR=str(WORKSPACE/'cache/torchinductor'),
               TRITON_CACHE_DIR=str(WORKSPACE/'cache/triton'), CUDA_CACHE_PATH=str(WORKSPACE/'cache/cuda'),
               MAX_JOBS='4', TORCHINDUCTOR_COMPILE_THREADS='4', OMP_NUM_THREADS='4',
               DO_NOT_TRACK='1', VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false')
    env['PATH'] = str(PYTHON.parent) + os.pathsep + str(CUDA/'bin') + os.pathsep + env['PATH']
    # Avoid the inherited CUDA 12.9 module overriding wheel CUDA 13 libraries.
    for key in ('LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH'):
        env[key] = os.pathsep.join(p for p in env.get(key, '').split(os.pathsep)
                                   if p and 'cuda-12.' not in p)
    for key in ('LD_PRELOAD', 'CUDA_INJECTION64_PATH', 'NVBIT_INSTRUMENTATION_ENABLED'):
        env.pop(key, None)
    return env


def run(root, label, command, timeout=900, gpu=False, env=None):
    root = Path(root)
    if time.time() >= LAUNCH_CUTOFF:
        return {'label': label, 'status': 'pending', 'reason': 'allocation closeout cutoff reached'}
    limit = min(timeout, max(1, int(LAUNCH_CUTOFF-time.time())))
    log = root/'logs'/(label+'.log')
    if log.exists():
        raise FileExistsError('attempt label already exists: '+str(log))
    if gpu:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                              '--format=csv,noheader'], text=True).strip()
        if processes:
            raise RuntimeError('GPU is not idle before '+label+': '+processes)
    started = time.time()
    row = {'label': label, 'command': [str(c) for c in command], 'started_utc':
           datetime.datetime.now(datetime.timezone.utc).isoformat(), 'timeout_s': limit,
           'log': str(log.relative_to(root))}
    row['script_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (PACKAGE/'scripts').glob('*.py')}
    print('START', label, flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(row['command'], cwd=PACKAGE, env=env or environment(),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        row['pid'] = process.pid
        dump(root/'results/current_process.json', row)
        try:
            code = process.wait(timeout=limit)
            row.update(exit_code=code, status='complete' if code == 0 else 'failed')
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            row.update(status='timeout', exit_code=process.returncode)
    row['elapsed_s'] = time.time()-started
    row['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
    with (root/'results/experiment_log.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    dump(root/'results/current_process.json', {'status': 'idle', 'last_attempt': label})
    print('END', label, row['status'], round(row['elapsed_s'], 2), flush=True)
    return row


def python_command(script, *arguments):
    return [PYTHON, PACKAGE/'scripts'/script, *arguments]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--label', required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    outcome = run(args.root, args.label, command, args.timeout, args.gpu)
    sys.exit(0 if outcome['status'] == 'complete' else 1)
