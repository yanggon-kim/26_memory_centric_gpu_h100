#!/usr/bin/env python3
"""Create a fresh allocation-bound campaign from durable evidence, without Codex."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
GPU_STEPS = {'J5', 'J8', 'J7', 'J10', 'J6', 'J11_gpu'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def evidence_path(name):
    path = (PACKAGE / name).resolve()
    if not path.is_relative_to(PACKAGE / 'h200'):
        raise ValueError('Evidence must be inside h200/')
    return path


def previous_session(plan, job_id):
    if plan['previous_session'] != 'latest':
        previous = evidence_path(plan['previous_session'])
    else:
        candidates = []
        for path in (PACKAGE / 'h200').glob('*/session.json'):
            settings = read(path)
            number = int(settings['slurm_job_id'])
            if (int(plan['minimum_previous_job_id']) <= number < int(job_id)
                    and (path.parent / 'results/shoreline_case_manifest.json').exists()
                    and (path.parent / 'results/campaign_queue.json').exists()):
                candidates.append((number, path.parent))
        if not candidates:
            raise RuntimeError('No durable predecessor satisfies this continuation plan')
        previous = max(candidates)[1]
    if int(read(previous / 'session.json')['slurm_job_id']) >= int(job_id):
        raise ValueError('Predecessor must be an earlier allocation')
    return previous


def reused_jobs(previous):
    """A finished command is a reusable disposition, not a passed scientific gate."""
    reused = {}
    visited = set()
    while previous not in visited:
        visited.add(previous)
        ledger = previous / 'results/campaign_queue.json'
        if ledger.exists():
            for row in read(ledger)['steps']:
                if (row['id'] in GPU_STEPS and row['status'] == 'command_complete'
                        and row.get('exit_code') == 0):
                    reused.setdefault(row['id'], {'path': str(ledger.relative_to(PACKAGE)),
                        'sha256': digest(ledger), 'disposition': row,
                        'scientific_gate': 'not inferred from command exit'})
        # The original October 16 namespace predates versioned session.json.
        # Its ledger can be retained, but it cannot supply another predecessor.
        if not (previous / 'session.json').exists():
            break
        settings = read(previous / 'session.json')
        # Inherited skips already carry their original, hash-bound ledger.
        for key, proof in settings.get('queue_reused_jobs', {}).items():
            if key not in GPU_STEPS:
                raise ValueError('Unknown inherited queue step')
            if digest(evidence_path(proof['path'])) != proof['sha256']:
                raise ValueError('Inherited queue ledger changed')
            reused.setdefault(key, proof)
        if not settings.get('old_session'):
            break
        previous = evidence_path(settings['old_session'])
    return reused


def allocation(state, hostname, now, buffer_s, minimum_s):
    fields = dict(x.split('=', 1) for x in state.split() if '=' in x)
    if fields.get('JobState') != 'RUNNING' or fields.get('NodeList') != hostname.split('.')[0]:
        raise RuntimeError('This batch step is not on the running allocated node')
    end = dt.datetime.fromisoformat(fields['EndTime']).astimezone()
    cutoff = end - dt.timedelta(seconds=buffer_s)
    if (cutoff - now).total_seconds() < minimum_s:
        raise RuntimeError('Insufficient budget before the launch cutoff')
    return end, cutoff


def checked(command, log, env=None, timeout=1800):
    print('BATCH_STAGE', log.stem, flush=True)
    with log.open('x') as stream:
        subprocess.run([str(x) for x in command], cwd=PACKAGE, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def prepare_evidence(previous, root):
    audit = []
    for name in ('prior_capture_evidence.json', 'evidence_reuse.json'):
        source = previous / 'results' / name
        data = read(source)
        if name.startswith('prior'):
            pairs = [(evidence_path(row['path']), row['sha256']) for row in data['evidence']]
        else:
            pairs = [(evidence_path(row['original_session']) / row[field], row[sha])
                     for row in data['entries']
                     for field, sha in (('artifact', 'sha256'),
                                        ('qualification_artifact', 'qualification_sha256'))]
        for path, expected in pairs:
            if digest(path) != expected:
                raise RuntimeError('Historical evidence hash mismatch: ' + str(path))
        shutil.copyfile(source, root / 'results' / name)
        audit.append({'path': str(source.relative_to(PACKAGE)), 'sha256': digest(source)})
    # Verify the immediate predecessor's excluded raw files, retaining its ledger untouched.
    inventory = previous / 'results/checkpoint_raw_inventory.json'
    for row in read(inventory)['raw_artifacts']:
        path = evidence_path(row['path'])
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise RuntimeError('Predecessor raw artifact missing or changed: ' + str(path))
    audit.append({'path': str(inventory.relative_to(PACKAGE)), 'sha256': digest(inventory),
                  'verified_raw_count': len(read(inventory)['raw_artifacts'])})
    write(root / 'results/batch_evidence_audit.json', {'verified': audit,
          'scope': 'historical outputs and raw integrity; no cross-allocation timing or clock joins'})


def main(args):
    plan = read(args.plan)
    job_id = os.environ.get('SLURM_JOB_ID', '')
    if not job_id.isdigit() or not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise RuntimeError('Run via sbatch/srun with a Slurm GPU binding')
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('The pinned runtime requires Python 3.10')
    hostname = socket.gethostname()
    state = subprocess.check_output(['scontrol', 'show', 'job', '-o', job_id], text=True, timeout=20)
    end, cutoff = allocation(state, hostname, dt.datetime.now().astimezone(),
                            plan['closeout_buffer_s'], plan['minimum_launch_budget_s'])
    previous = previous_session(plan, job_id)
    active = subprocess.check_output(['squeue', '--noheader', '--me', '--format=%A'], text=True, timeout=20)
    if read(previous / 'session.json')['slurm_job_id'] in active.split():
        raise RuntimeError('Predecessor allocation is still scheduled; do not start a competing continuation')
    workspace = Path('/tmp') / f'h200_research_{os.getuid()}_job{job_id}'
    root = PACKAGE / 'h200' / f'{dt.datetime.now():%Y%m%d}_job{job_id}'
    # Never reuse process outcomes from another allocation, nor overwrite this attempt.
    root.mkdir(exist_ok=False)
    for folder in ('environment/source_snapshots', 'results', 'logs', 'configs', 'runs', 'profiles', 'analysis'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    workspace.mkdir(exist_ok=False)
    outcome = {'status': 'preparing', 'job_id': job_id, 'previous_session': str(previous.relative_to(PACKAGE)),
               'plan_sha256': digest(args.plan), 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write(root / 'results/batch_bootstrap.json', outcome)
    try:
        cuda = Path(plan['cuda_home'])
        for tool in ('nvcc', 'ncu', 'nsys'):
            if not (cuda / 'bin' / tool).is_file():
                raise RuntimeError('Required pinned CUDA tool unavailable: ' + tool)
        storage = {name: shutil.disk_usage(path)._asdict() for name, path in [('tmp', workspace), ('shared', root)]}
        write(root / 'environment/batch_storage.json', storage)
        for name in ('tmp', 'shared'):
            if storage[name]['free'] < plan[f'minimum_{name}_free_gib'] * 1024**3:
                raise RuntimeError('Insufficient ' + name + ' free space')
        prepare_evidence(previous, root)
        settings = read(previous / 'session.json')
        if not settings.get('retained_trace_blocker'):
            raise RuntimeError('This bootstrap requires the reviewed trace blocker; no automatic trace-tool build')
        for field, sha in [('path', 'sha256'), ('archive', 'archive_sha256')]:
            if digest(evidence_path(settings['retained_trace_blocker'][field])) != settings['retained_trace_blocker'][sha]:
                raise RuntimeError('Retained trace evidence mismatch')
        for key in ('gpu_uuid', 'frozen_source_root', 'frozen_source_identity', 'sm_pilot_reviews', 'queue_reused_jobs'):
            settings.pop(key, None)
        settings.update(plan['session_overrides'])
        settings.update(slurm_job_id=job_id, expected_hostname=hostname, workspace=str(workspace),
                        python=str(workspace / 'runtime/bin/python'), cuda_home=str(cuda),
                        allocation_end=end.isoformat(), launch_cutoff=cutoff.isoformat(),
                        closeout_buffer_s=plan['closeout_buffer_s'], old_session=str(previous.relative_to(PACKAGE)),
                        accel_root=str(workspace / 'tools/accel-sim-framework'),
                        queue_reused_jobs=reused_jobs(previous))
        write(root / 'session.json', settings)
        manifest = {'schema_version': 1, 'jobs': [
            {'id': f'J{i}', 'status': 'planned', 'scientific_gate': 'pending',
             'prior_evidence': str((previous / 'STATUS.json').relative_to(PACKAGE))} for i in range(12)]}
        write(root / 'results/shoreline_case_manifest.json', manifest)
        write(root / 'STATUS.json', {'status': 'bootstrap_pending', 'jobs': manifest['jobs']})
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE='1', PIP_NO_INPUT='1', PIP_NO_CACHE_DIR='1',
                   CUDA_HOME=str(cuda), CUDA_INSTALL_PATH=str(cuda),
                   PATH=str(cuda / 'bin') + os.pathsep + env['PATH'])
        for key in ('LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH'):
            env[key] = os.pathsep.join(p for p in env.get(key, '').split(os.pathsep) if p and 'cuda-12.' not in p)
        for key in ('PYTHONPATH', 'LD_PRELOAD', 'CUDA_INJECTION64_PATH', 'NVBIT_INSTRUMENTATION_ENABLED'):
            env.pop(key, None)
        python = Path(settings['python'])
        checked([sys.executable, '-m', 'venv', python.parent.parent], root / 'logs/batch_venv.log', env)
        checked([python, '-m', 'pip', 'install', '-r', PACKAGE / plan['runtime_lock']],
                root / 'logs/batch_install.log', env, timeout=3600)
        # Query exactly the single device inherited from the GPU-bound Slurm step.
        query = """import json,torch
assert torch.cuda.device_count()==1, 'expected exactly one Slurm-visible GPU'
p=torch.cuda.get_device_properties(0)
assert 'H200' in p.name and torch.cuda.get_device_capability(0)==(9,0)
assert torch.version.cuda=='13.0', 'runtime CUDA drift'
print(json.dumps({'uuid':str(p.uuid),'name':p.name,'sm_count':p.multi_processor_count}))
"""
        device = json.loads(subprocess.check_output([str(python), '-c', query], env=env, text=True, timeout=120))
        uuid = device['uuid']
        settings['gpu_uuid'] = uuid if uuid.startswith('GPU-') else 'GPU-' + uuid
        write(root / 'environment/batch_bound_device.json', device)
        write(root / 'session.json', settings)
        write(root / 'environment/batch_plan.json', plan)
        # The selected native-only scope must not silently discard missing B8 traffic.
        if settings['sm_components'] == ['native'] and 'J5' not in settings['queue_reused_jobs']:
            traffic_proofs = []
            for fraction in (100, 75, 50, 30):
                for acq in range(3):
                    name = f'J5_bridge_eager_b8_c4096_full_f{fraction}_acq{acq:02d}_ticks_only_v4_reader_v3.json'
                    candidates = [previous / 'profiles' / name]
                    # The latest namespace may itself have reused the completed traffic.
                    candidates += [evidence_path(plan['traffic_completed_session']) / 'profiles' / name]
                    proof = next((p for p in candidates if p.exists() and read(p).get('status') == 'complete'), None)
                    if proof is None:
                        raise RuntimeError('Native-only plan lacks completed B8 traffic: ' + name)
                    traffic_proofs.append({'path': str(proof.relative_to(PACKAGE)), 'sha256': digest(proof),
                                           'scope': 'prior instrumented physical traffic; no current timing join'})
            write(root / 'results/batch_reused_traffic.json', {'evidence': traffic_proofs})
        checked([python, PACKAGE / 'scripts/freeze_shoreline.py', '--session', root / 'session.json', '--activate'],
                root / 'logs/batch_freeze.log', env)
        settings = read(root / 'session.json')
        env['H200_SESSION_CONFIG'] = str(root / 'session.json')
        # Set cache/auth/device environment using the same contract as every acquisition.
        sys.path.insert(0, str(PACKAGE / 'scripts'))
        os.environ.clear()
        os.environ.update(env)
        from h200_session import environment
        env.update(environment())
        outcome['status'] = 'running'
        write(root / 'results/batch_bootstrap.json', outcome)
        checked([python, PACKAGE / 'scripts/checkpoint_h200.py', '--root', root,
                 '--message', f'Batch {job_id}: publish fresh allocation and automatic continuation plan', '--push'],
                root / 'logs/batch_initial_checkpoint.log', env)
        checked([python, PACKAGE / 'scripts/continue_shoreline.py', '--session', root / 'session.json'],
                root / 'logs/batch_campaign.log', env, timeout=max(1, int((end - dt.datetime.now().astimezone()).total_seconds())))
        queue = read(root / 'results/campaign_queue.json')
        if queue['status'] != 'commands_finished_scientific_gates_separate':
            raise RuntimeError('Campaign stopped: ' + queue['status'])
        outcome['status'] = 'commands_finished_scientific_gates_separate'
    except BaseException as error:
        outcome.update(status='needs_diagnosis', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        outcome['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(root / 'results/batch_bootstrap.json', outcome)
        # Best effort only: SIGKILL/preemption may prevent any final publication.
        try:
            checked([sys.executable, PACKAGE / 'scripts/checkpoint_h200.py', '--root', root,
                     '--message', f'Batch {job_id}: {outcome["status"]}', '--push'],
                    root / 'logs/batch_final_checkpoint.log', timeout=120)
        except Exception as error:
            print('FINAL_PUBLICATION_UNAVAILABLE', type(error).__name__, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    main(parser.parse_args())
