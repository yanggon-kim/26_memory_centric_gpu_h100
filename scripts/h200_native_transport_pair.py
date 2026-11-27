#!/usr/bin/env python3
"""Bounded, separately frozen native transport component on an idle owned pair.

This launcher does not implement model KV generation/injection or serving.
It never submits a Slurm job or terminates an unrelated process.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from h200_priority_campaign import PACKAGE, load, write, sha, publish, limitation

SOURCE_AT_IMPORT = Path(__file__).read_bytes()
ROLES = ('producer', 'consumer')
LABEL = 'native_known_data_transport'


def validate_sources(sessions, currents, job, hostname, uid, state):
    assert len(sessions) == len(currents) == 2
    assert len({s['gpu_uuid'] for s in sessions}) == 2
    assert len({s['python'] for s in sessions}) == 1
    assert len({s['cuda_home'] for s in sessions}) == 1
    assert all(s['slurm_job_id'] == job and s['expected_hostname'] == hostname for s in sessions)
    assert all(c.get('status') == 'idle' for c in currents)
    fields = dict(p.split('=', 1) for p in state.split() if '=' in p)
    assert fields['JobId'] == job and fields['JobState'] == 'RUNNING'
    assert fields['UserId'].endswith('('+str(uid)+')')
    assert fields['NodeList'] == hostname.split('.')[0]
    assert 'gres/gpu:h200=2' in fields['AllocTRES'].split(',')


def choose_ports():
    # Hold both sockets while choosing so the two ports cannot be identical.
    # Native engines require their own sockets: a later bind race fails closed.
    with contextlib.ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in ROLES]
        for s in sockets:
            s.bind(('127.0.0.1', 0))
        return [s.getsockname()[1] for s in sockets]


def validate_returned(outcomes):
    assert len(outcomes) == 2
    for row in outcomes:
        assert row.get('status') in ('resident_curve_disposition',
            'prefill_acquisition_disposition', 'partial_scientific_gates_separate')
        if row['status'] != 'partial_scientific_gates_separate':
            assert row.get('complete_acquisition_set') is True


def stop_owned_role(process, root):
    """Bound cleanup to the still-live supervisor and its direct GPU child."""
    if process.poll() is not None:
        return
    current = root/'results/current_process.json'
    if current.exists():
        child = load(current).get('pid')
        if isinstance(child, int):
            try:
                fields = Path(f'/proc/{child}/stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[1]) == process.pid and os.getpgid(child) == child:
                    os.killpg(child, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        # Recheck ancestry after the grace interval; never use
                        # a stale PID to terminate another process.
                        fields = Path(f'/proc/{child}/stat').read_text().rsplit(')', 1)[1].split()
                        if int(fields[1]) == process.pid and os.getpgid(child) == child:
                            os.killpg(child, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main(args):
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    args.root = args.root.resolve()
    assert args.root.is_relative_to(PACKAGE/'h200')
    paths = [p.resolve() for p in args.source_session]
    sessions = [load(p) for p in paths]
    job = os.environ.get('SLURM_JOB_ID')
    state = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)],
                                    text=True, timeout=25)
    validate_sources(sessions, [load(p.parent/'results/current_process.json') for p in paths],
                     job, socket.gethostname(), os.getuid(), state)
    validate_returned([load(p.parent/'results/worker_outcome.json') for p in paths])
    fields = dict(p.split('=', 1) for p in state.split() if '=' in p)
    timezone = dt.datetime.fromisoformat(sessions[0]['launch_cutoff']).tzinfo
    assert timezone is not None
    parse = lambda text: dt.datetime.fromisoformat(text).replace(tzinfo=timezone).timestamp()
    cutoff = min(parse(fields['EndTime'])-900,
                 *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
    # Aim to finish before first-hour preemption eligibility; it is not a lease.
    cutoff = min(cutoff, parse(fields['StartTime'])+3600-120)
    assert cutoff-time.time() >= 720, 'complete paired component budget does not fit'
    processes = []
    with contextlib.ExitStack() as stack:
        for old in sessions:
            lock = stack.enter_context((Path(old['workspace'])/'producer.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import pynvml as nv
        nv.nvmlInit()
        try:
            for old in sessions:
                assert not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByUUID(old['gpu_uuid']))
        finally:
            nv.nvmlShutdown()
        from h200_resident_campaign import verify_published_core
        for old in sessions:
            gate = old['published_core_gate']
            assert sha(gate['review']) == gate['review_sha256']
            verify_published_core(Path(gate['review']))
        args.root.mkdir(exist_ok=False)
        for name in ('environment', 'results', 'analysis', 'shared', 'logs'):
            (args.root/name).mkdir()
        (args.root/'analysis/report.md').write_text('# Native transport component probe\n\nSynthetic known BF16 blocks only. No model KV generation/injection, loaded serving, clean link latency, physical traffic or quantitative bandwidth qualification.\n')
        write(args.root/'results/idle_admission.json', {
            'job_id': job, 'state': state, 'source_sessions': [
                {'path': str(p), 'sha256': sha(p),
                 'worker_outcome_sha256': sha(p.parent/'results/worker_outcome.json'),
                 'idle_record_sha256': sha(p.parent/'results/current_process.json')} for p in paths],
            'protected_cutoff_unix_s': cutoff, 'required_budget_s': 720,
            'scope': 'Both original producer locks held; both selected UUIDs idle; published broad-curve eligibility rechecked'})
        publish(args.root, 'H200 native component: publish idle pair admission and launcher source')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PACKAGE, text=True).strip()
        for role, old in zip(ROLES, sessions):
            root = args.root/role
            for name in ('environment', 'results', 'logs', 'analysis'):
                (root/name).mkdir(parents=True, exist_ok=True)
            session = {k: v for k, v in old.items() if k not in ('frozen_source_root', 'frozen_source_identity')}
            session.update(workspace=old['workspace']+'_native_'+args.root.name,
                           source_commit=commit, protocol='synthetic_native_transport_component_v1')
            Path(session['workspace']).mkdir(exist_ok=False)
            write(root/'session.json', session)
            subprocess.run([old['python'], PACKAGE/'scripts/freeze_shoreline.py', '--session',
                            root/'session.json', '--activate'], check=True, timeout=120)
        write(args.root/'results/post_preparation_budget.json', {
            'available_s': cutoff-time.time(), 'required_s': 600,
            'scope': 'Recheck complete paired probe, supervision and cleanup budget after publication/freezing'})
        assert cutoff-time.time() >= 600, 'complete paired probe no longer fits after preparation; no GPU producer launched'
        ports = choose_ports()
        deadline = min(cutoff, time.time()+540)
        pair = {'job_id': job, 'hostname': socket.gethostname(), 'host': '127.0.0.1',
                'roles': {role: {'gpu_uuid': old['gpu_uuid'], 'port': port}
                          for role, old, port in zip(ROLES, sessions, ports)},
                'synthetic_payload': True, 'request_prefix': 'h200_'+job+'_'+str(time.time_ns()),
                'shared_directory': str(args.root/'shared'), 'deadline_unix_s': deadline}
        pair_path = args.root/'pair.json'; write(pair_path, pair)
        assert len({load(args.root/role/'session.json')['frozen_source_identity'] for role in ROLES}) == 1
        try:
            for role in ROLES:
                root = args.root/role; session = load(root/'session.json')
                scripts = Path(session['frozen_source_root'])/'scripts'
                env = os.environ.copy()
                for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'CUDA_INJECTION64_PATH', 'NVBIT_INSTRUMENTATION_ENABLED', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH'):
                    env.pop(key, None)
                env.update(H200_SESSION_CONFIG=str(root/'session.json'), PYTHONNOUSERSITE='1',
                           CUDA_VISIBLE_DEVICES=session['gpu_uuid'])
                for key, name in (('CC', 'h200_host_cc.sh'), ('CXX', 'h200_host_cxx.sh')):
                    (scripts/name).chmod(0o700); env[key] = str(scripts/name)
                command = [session['python'], scripts/'h200_session.py', '--root', root,
                           '--label', LABEL, '--gpu', '--timeout', '500', '--',
                           session['python'], scripts/'h200_native_transport_probe.py',
                           '--pair-config', pair_path, '--role', role, '--gpu-uuid', session['gpu_uuid'],
                           '--output', root/'runs/component']
                log = stack.enter_context((args.root/'logs'/(role+'_supervisor.log')).open('w'))
                process = subprocess.Popen(list(map(str, command)), cwd=PACKAGE, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((process, root))
            for process, root in processes:
                process.wait(timeout=max(1, deadline+40-time.time()))
            rows = []
            for process, root in processes:
                proc_path = root/'results/processes'/(LABEL+'.json')
                run_path = root/'runs/component/run.json'
                proc, run = load(proc_path), load(run_path)
                assert process.returncode == 0 and proc['status'] == 'complete'
                assert not proc['monitor']['failure'] and run['status'] == 'known_data_transport_probe_passed'
                assert len(run['trials']) == 6 and all(t['status'] == 'known_data_transport_passed' for t in run['trials'])
                rows.append({'process': str(proc_path), 'process_sha256': sha(proc_path),
                             'run': str(run_path), 'run_sha256': sha(run_path)})
            write(args.root/'results/pair_disposition.json', {'status': 'component_captured_pending_CPU_review',
                  'roles': rows, 'model_KV_verified': False, 'serving_qualified': False,
                  'quantitative_performance_qualified': False})
        except BaseException as error:
            write(args.root/'results/pair_disposition.json', {'status': 'failed', 'error': str(error),
                  'traceback': traceback.format_exc(), 'model_KV_verified': False, 'serving_qualified': False})
            limitation(args.root, 'native_transport_component', str(error), 'results/pair_disposition.json',
                       'Paired immutable-source supervised roles with known blocks and bounded watchdogs',
                       'Inspect preserved partial roles before one bounded repair; model KV and serving remain pending')
            raise
        finally:
            for process, root in processes:
                stop_owned_role(process, root)
            publish(args.root, 'H200 native transport component: publish bounded paired disposition')


def launch(args):
    try:
        main(args)
    except BaseException as error:
        # Preparation/freeze failures also need an immediate local and global
        # record, before any retry. Invalid ownership fails before root creation.
        if (args.root/'analysis/report.md').exists() and not (args.root/'results/pair_disposition.json').exists():
            write(args.root/'results/pair_disposition.json', {'status': 'failed_preparation',
                'error': str(error), 'traceback': traceback.format_exc(),
                'model_KV_verified': False, 'serving_qualified': False})
            limitation(args.root, 'native_transport_pair_preparation', str(error),
                'results/pair_disposition.json', 'Idle owned pair and immutable-source preparation',
                'Inspect preparation evidence before a bounded retry; no transport result inferred')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session', type=Path, action='append', required=True)
    parser.add_argument('--root', type=Path, required=True)
    launch(parser.parse_args())
