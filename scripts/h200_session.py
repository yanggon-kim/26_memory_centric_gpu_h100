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
SESSION = {}
if os.environ.get('H200_SESSION_CONFIG'):
    SESSION = json.loads(Path(os.environ['H200_SESSION_CONFIG']).read_text())
    DEFAULT_ROOT = Path(os.environ['H200_SESSION_CONFIG']).resolve().parent
    WORKSPACE = Path(SESSION['workspace'])
    CUDA = Path(SESSION['cuda_home'])
    UUID = SESSION['gpu_uuid']
    PYTHON = Path(SESSION['python'])
    LAUNCH_CUTOFF = datetime.datetime.fromisoformat(SESSION['launch_cutoff']).timestamp()
NUMERICAL_PACKAGE = Path(SESSION.get('frozen_source_root', PACKAGE))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def environment():
    env = os.environ.copy()
    # Reuse the user's existing HF authentication without moving or logging it.
    token_path=Path(env.get('HF_TOKEN_PATH',str(Path(env.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token')))
    if token_path.is_file():env['HF_TOKEN_PATH']=str(token_path)
    env.update(CUDA_HOME=str(CUDA), CUDA_INSTALL_PATH=str(CUDA), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES=UUID, HF_HOME=str(WORKSPACE/'cache/huggingface'),
               HF_HUB_CACHE=str(WORKSPACE/'models/hub'), HF_HUB_DISABLE_TELEMETRY='1',
               PIP_CACHE_DIR=str(WORKSPACE/'cache/pip'), XDG_CACHE_HOME=str(WORKSPACE/'cache'),
               TORCHINDUCTOR_CACHE_DIR=str(WORKSPACE/'cache/torchinductor'),
               TRITON_CACHE_DIR=str(WORKSPACE/'cache/triton'), CUDA_CACHE_PATH=str(WORKSPACE/'cache/cuda'),
               MAX_JOBS='4', TORCHINDUCTOR_COMPILE_THREADS='4', OMP_NUM_THREADS='4',
               DO_NOT_TRACK='1', VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false')
    env['PATH'] = str(PYTHON.parent) + os.pathsep + str(CUDA/'bin') + os.pathsep + env['PATH']
    if SESSION.get('accel_root'):
        env['H200_ACCEL_ROOT']=SESSION['accel_root']
    # Avoid the inherited CUDA 12.9 module overriding wheel CUDA 13 libraries.
    for key in ('LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH'):
        env[key] = os.pathsep.join(p for p in env.get(key, '').split(os.pathsep)
                                   if p and 'cuda-12.' not in p)
    for key in ('LD_PRELOAD', 'CUDA_INJECTION64_PATH', 'NVBIT_INSTRUMENTATION_ENABLED'):
        env.pop(key, None)
    return env


def verify_allocation():
    if not SESSION:
        return
    import socket
    assert os.environ.get('SLURM_JOB_ID') == SESSION['slurm_job_id'], 'allocation changed'
    assert socket.gethostname() == SESSION['expected_hostname'], 'allocation node changed'
    failures=[]
    for attempt in range(3):
        try:
            state=subprocess.check_output(['scontrol','show','job','-o',SESSION['slurm_job_id']],
                text=True,stderr=subprocess.STDOUT,timeout=15)
            break
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            failures.append({'attempt':attempt,'type':type(error).__name__})
            if attempt==2:raise RuntimeError('allocation status unavailable after three bounded read-only queries; no GPU launch authorized') from error
            time.sleep(1)
    assert 'JobState=RUNNING' in state and 'NodeList='+socket.gethostname().split('.')[0] in state
    return {'job_id':SESSION['slurm_job_id'],'hostname':socket.gethostname(),'state':'RUNNING',
            'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transient_query_failures':failures}


class ProcessMonitor:
    """Sample only the selected physical device and reject foreign ancestry."""
    def __init__(self, root, label, process):
        import threading
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(UUID)
        self.process = process
        self.stop = threading.Event()
        self.failure = None
        self.path = Path(root)/'results/monitors'/(label+'.jsonl')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def owned(self, pid):
        seen = set()
        while pid > 1 and pid not in seen:
            if pid == self.process.pid:
                return True
            seen.add(pid)
            try:
                # CUDA/NVML may report a thread ID. Resolve its thread group
                # before following process parents; a thread's PPID can skip its own group leader.
                status=Path(f'/proc/{pid}/status').read_text()
                tgid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('Tgid:')))
                if tgid!=pid:
                    pid=tgid
                    continue
                # /proc comm may contain spaces; fields after its final ')' are stable.
                fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                pid = int(fields[1])
            except (OSError, ValueError):
                return None  # process disappeared between NVML and ancestry lookup
        return False

    def sample(self):
        nv = self.nvml
        with self.path.open('w') as stream:
            while not self.stop.is_set():
                try:
                    pids = [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(self.handle)]
                    ancestry = {str(p): self.owned(p) for p in pids}
                    identities={}
                    for p in pids:
                        try:
                            status=Path(f'/proc/{p}/status').read_text().splitlines()
                            identities[str(p)]={k:next((s.split(':',1)[1].strip() for s in status if s.startswith(k+':')),None)
                                                for k in ('Name','Tgid','PPid','Uid')}
                            identities[str(p)].update(session_id=os.getsid(p),process_group=os.getpgid(p))
                        except (OSError,ValueError):identities[str(p)]={'unavailable':True}
                    row = {'unix_ns': time.time_ns(), 'gpu_uuid': UUID, 'pids': pids,
                           'owned_ancestry': ancestry,
                           'process_identities':identities,
                           'sm_clock_mhz': nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_SM),
                           'memory_clock_mhz': nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_MEM),
                           'temperature_c': nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU),
                           'power_w': nv.nvmlDeviceGetPowerUsage(self.handle)/1000}
                    if any(v is not True for v in ancestry.values()):
                        self.failure = 'foreign GPU process detected' if any(v is False for v in ancestry.values()) else 'GPU process ancestry unavailable'
                        row['rejection'] = self.failure
                        chains={}
                        for candidate,owned in ancestry.items():
                            if owned is True:continue
                            pid=int(candidate);seen=set();chain=[]
                            while pid>1 and pid not in seen and len(chain)<16:
                                seen.add(pid)
                                try:
                                    lines=Path(f'/proc/{pid}/status').read_text().splitlines()
                                    item={k:next((s.split(':',1)[1].strip() for s in lines if s.startswith(k+':')),None)
                                          for k in ('Name','Tgid','PPid','Uid')}
                                    item['pid']=pid;chain.append(item)
                                    tgid=int(item['Tgid']);pid=tgid if tgid!=pid else int(item['PPid'])
                                except (OSError,ValueError):break
                            chains[candidate]=chain
                        row['rejected_process_ancestry']=chains
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                except Exception as error:
                    self.failure = 'isolation monitor error: '+str(error)
                if self.failure:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
                self.stop.wait(SESSION.get('monitor_period_s', 0.05))

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return {'source': str(self.path), 'failure': self.failure,
                'coverage': 'sampled process ancestry and hardware state; arbitrarily short interference cannot be excluded'}


def run(root, label, command, timeout=900, gpu=False, env=None):
    allocation_proof=verify_allocation()
    root = Path(root)
    if time.time() >= LAUNCH_CUTOFF:
        return {'label': label, 'status': 'pending', 'reason': 'allocation closeout cutoff reached'}
    limit = min(timeout, max(1, int(LAUNCH_CUTOFF-time.time())))
    log = root/'logs'/(label+'.log')
    if log.exists():
        raise FileExistsError('attempt label already exists: '+str(log))
    if gpu:
        if SESSION:
            import fcntl
            producer_lock=(WORKSPACE/'producer.lock').open('a')
            fcntl.flock(producer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            import pynvml
            pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(UUID)
            required=SESSION.get('idle_required_s',10)
            idle_start=time.monotonic();deadline=time.monotonic()+60;checks=[]
            while time.monotonic()-idle_start<required:
                pids=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
                checks.append({'unix_ns':time.time_ns(),'pids':pids})
                if pids:idle_start=time.monotonic()
                if time.monotonic()>deadline:
                    dump(root/'results/idle_checks'/(label+'.json'),{'passed':False,'samples':checks})
                    raise RuntimeError('selected GPU did not remain idle; no foreign process terminated')
                time.sleep(.25)
            dump(root/'results/idle_checks'/(label+'.json'),{'passed':True,'required_idle_s':required,'samples':checks})
        else:
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                  '--format=csv,noheader'], text=True).strip()
            if processes:
                raise RuntimeError('GPU is not idle before '+label+': '+processes)
    started = time.time()
    row = {'label': label, 'command': [str(c) for c in command], 'started_utc':
           datetime.datetime.now(datetime.timezone.utc).isoformat(), 'timeout_s': limit,
           'log': str(log.relative_to(root))}
    row['allocation_verification']=allocation_proof
    row['script_sha256']={}
    snapshots=root/'environment/source_snapshots'
    snapshots.mkdir(parents=True,exist_ok=True)
    row['numerical_source_root']=str(NUMERICAL_PACKAGE)
    row['source_freeze_identity']=SESSION.get('frozen_source_identity')
    row['orchestration_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (PACKAGE/'scripts').glob('shoreline_*.py') if p.name in ('shoreline_jobs.py','shoreline_campaign.py')}
    for name in ('h200_session.py','checkpoint_h200.py','shoreline_queue.py'):
        source=PACKAGE/'scripts'/name
        if source.exists():row['orchestration_source_sha256'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    for source in (NUMERICAL_PACKAGE/'scripts').glob('*.py'):
        content=source.read_bytes();digest=hashlib.sha256(content).hexdigest()
        row['script_sha256'][source.name]=digest
        destination=snapshots/(digest+'.py')
        if not destination.exists():destination.write_bytes(content)
    for name,digest in row['orchestration_source_sha256'].items():
        target=snapshots/(digest+'.py')
        if not target.exists():target.write_bytes((PACKAGE/'scripts'/name).read_bytes())
    row['script_snapshot_root']='environment/source_snapshots'
    print('START', label, flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(row['command'], cwd=PACKAGE, env=env or environment(),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        row['pid'] = process.pid
        monitor = ProcessMonitor(root, label, process) if gpu and SESSION else None
        dump(root/'results/processes'/(label+'.json'), row)
        if gpu:dump(root/'results/current_process.json', row)
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
        if monitor:
            row['monitor'] = monitor.finish()
            if monitor.failure:
                row.update(status='rejected', reason=monitor.failure)
    row['elapsed_s'] = time.time()-started
    row['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
    with (root/'results/experiment_log.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    dump(root/'results/processes'/(label+'.json'), row)
    if gpu:dump(root/'results/current_process.json', {'status': 'idle', 'last_attempt': label})
    print('END', label, row['status'], round(row['elapsed_s'], 2), flush=True)
    return row


def python_command(script, *arguments):
    return [PYTHON, NUMERICAL_PACKAGE/'scripts'/script, *arguments]


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
