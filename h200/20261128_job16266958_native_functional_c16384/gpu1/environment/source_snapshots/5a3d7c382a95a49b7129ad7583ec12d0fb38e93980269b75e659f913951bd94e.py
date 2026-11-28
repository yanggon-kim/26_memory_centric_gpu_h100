#!/usr/bin/env python3
"""Bounded idle-spare transition after the observed native host-linker stall."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from h200_priority_campaign import PACKAGE,Worker,load,write,publish,sha

SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def validate_transition(session,root,current,command,owner,children,gpu_pids):
    assert owner==os.getuid()
    assert current['status']=='idle', 'previous producer still active'
    assert not children and not gpu_pids, 'previous worker/selected GPU is not idle'
    assert any(Path(x).name=='h200_priority_campaign.py' for x in command)
    assert command[command.index('--worker')+1]=='1'
    assert Path(command[command.index('--root')+1]).resolve()==root.resolve()
    assert Path(command[command.index('--session')+1]).resolve()==(root/'session.json').resolve()
    assert session['gpu_uuid'].startswith('GPU-')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',type=Path,required=True);p.add_argument('--previous-pid',type=int,required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    old=load(a.source_session);prior=a.source_session.resolve().parent
    assert os.environ['SLURM_JOB_ID']==old['slurm_job_id']
    assert socket.gethostname()==old['expected_hostname']
    state=subprocess.check_output(['scontrol','show','job','-o',old['slurm_job_id']],text=True)
    assert 'JobState=RUNNING' in state and 'gres/gpu:h200=2' in state
    assert f'({os.getuid()})' in state
    assert not (prior.parent/'gpu0/results/eager_core_disposition.json').exists(), 'review next boundary before transitioning'
    import pynvml
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(old['gpu_uuid'])
    proc=Path('/proc')/str(a.previous_pid)
    command=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
    children=proc.joinpath('task',str(a.previous_pid),'children').read_text().split()
    pids=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
    current=load(prior/'results/current_process.json')
    validate_transition(old,prior,current,command,proc.stat().st_uid,children,pids)
    lock=(Path(old['workspace'])/'producer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.kill(a.previous_pid,signal.SIGTERM)
    for _ in range(30):
        if not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('old idle worker has not exited')
    assert not pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    pynvml.nvmlShutdown();lock.close()
    a.root=a.root.resolve();a.root.mkdir(exist_ok=False)
    for folder in ('environment','results','logs','configs','runs','profiles','analysis'):(a.root/folder).mkdir()
    (a.root/'analysis/report.md').write_text('# Spare-device native-linker repair cohort\n\nSame allocated UUID, new frozen source cohort. Original native counter pilot timed out before any trial while the host linker wrapper waited. Explicit same-GCC/native-linker selection is an experiment-local build choice; installed tools and the other GPU worker are unchanged. No new GPU allocation.\n')
    write(a.root/'results/idle_transition.json',{'previous_pid':a.previous_pid,'validated_command':command,
          'prior_current_process':current,'gpu_uuid':old['gpu_uuid'],'prior_gpu_pids':pids,
          'status':'owned_idle_worker_stopped_before_new_cohort','job_id':old['slurm_job_id'],
          'scope':'intentional CPU-worker transition after producer timeout, not cancellation of the Slurm allocation'})
    session=dict(old);session['workspace']=old['workspace']+'_native_linker';Path(session['workspace']).mkdir(exist_ok=False)
    session['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    session['spare_scheduler_pilot']=True
    session['native_counter_timeout_s']=180
    for key in ('frozen_source_root','frozen_source_identity'):session.pop(key,None)
    path=a.root/'session.json';write(path,session)
    subprocess.run([session['python'],PACKAGE/'scripts/freeze_shoreline.py','--session',path,'--activate'],check=True)
    worker=Worker(a.root,path)
    for variable,name in (('CC','h200_host_cc.sh'),('CXX','h200_host_cxx.sh')):
        wrapper=worker.script/name;wrapper.chmod(0o700);os.environ[variable]=str(wrapper)
        (a.root/'environment'/name).write_bytes(wrapper.read_bytes())
    write(a.root/'environment/host_compiler_selection.json',{'CC':os.environ['CC'],'CXX':os.environ['CXX'],
          'wrapper_sha256':{n:sha(worker.script/n) for n in ('h200_host_cc.sh','h200_host_cxx.sh')},
          'scope':'same GCC12.3 frontend, explicit-B/usr/bin native linker; separate cohort and compiler diagnostics'})
    worker.execute('host_cc_version',[os.environ['CC'],'--version'],gpu=False,timeout=30)
    selected=worker.execute('host_linker_selection',[os.environ['CC'],'-print-prog-name=ld'],gpu=False,timeout=30)
    assert selected['status']=='complete'
    assert (a.root/'logs/host_linker_selection.log').read_text().strip()=='/usr/bin/ld'
    worker.execute('host_ld_version',['/usr/bin/ld','--version'],gpu=False,timeout=30)
    outcome=__import__('h200_eager_first').secondary(worker)
    write(a.root/'results/worker_outcome.json',outcome)
    publish(a.root,'H200 idle-spare linker repair cohort: preserve all outcomes and limits')


if __name__=='__main__':main()
