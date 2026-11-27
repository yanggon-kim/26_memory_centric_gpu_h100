#!/usr/bin/env python3
"""Resume only a budget-stopped curve on its unchanged allocated UUID/source.

This changes analytical scheduling allowance, never numerical programs,
acquisition counts, trial counts, correctness, clocks or isolation gates.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from h200_priority_campaign import PACKAGE,Worker,load,write,sha,publish,analyze_curve,limitation,FRACTIONS

SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def allowance(observed,completed,available):
    assert observed and all(t>0 for t in observed) and 0<completed<3
    estimate=(3-completed)*(max(observed)+60)
    return {'classification':'analytical scheduling estimate only','completed_acquisitions':completed,
            'observed_acquisition_wall_s':list(observed),'remaining_bundle_estimate_s':estimate,
            'available_before_closeout_s':available,'admitted':estimate<=available,
            'method':'maximum observed whole acquisition plus60s each;15minute CPU-closeout reserve; actual Slurm end and all scientific gates unchanged'}


def captured_acquisition(records,acquisition):
    selected=[r for r in records if r['engine']=='eager' and not r['pilot'] and r['acquisition']==acquisition]
    if len(selected)!=9 or not all(r['status']=='complete' and len(r.get('step_ms',[]))==10 for r in selected):return False
    stock=[r for r in selected if r['stock']]
    green=[r for r in selected if not r['stock']]
    return (len(stock)==1 and not stock[0].get('suffix') and
            {(r['fraction'],r.get('suffix','')) for r in green}==
            {(f,'') for f in FRACTIONS}|{(1.,'_interleaved'),(1.,'_end')})


def stop_idle_parent(pid,root,uuid,deadline):
    import pynvml
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(uuid)
    proc=Path('/proc')/str(pid)
    command=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
    assert proc.stat().st_uid==os.getuid()
    assert any(Path(x).name=='h200_priority_campaign.py' for x in command)
    assert command[command.index('--worker')+1]=='0'
    assert Path(command[command.index('--root')+1]).resolve()==root.resolve()
    assert Path(command[command.index('--session')+1]).resolve()==(root/'session.json').resolve()
    while time.time()<deadline:
        current=load(root/'results/current_process.json')
        if current.get('status')!='idle' or pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            time.sleep(.5);continue
        paused=False;terminated=False
        try:
            os.kill(pid,signal.SIGSTOP);paused=True
            children=proc.joinpath('task',str(pid),'children').read_text().split()
            current=load(root/'results/current_process.json')
            if children or current.get('status')!='idle' or pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                continue
            # Preserve the original runtime build choices, including an unset
            # CC/CXX. Never apply the spare's compiler repair to this curve.
            inherited=dict(x.split('=',1) for x in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in x)
            selected={}
            for key in ('PATH','CC','CXX','CUDA_HOME','CUDA_INSTALL_PATH','OMP_NUM_THREADS','MAX_JOBS','TORCHINDUCTOR_COMPILE_THREADS','PYTHONNOUSERSITE','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','CUDA_MODULE_LOADING','PYTORCH_CUDA_ALLOC_CONF','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG'):
                if key in inherited:os.environ[key]=inherited[key];selected[key]=inherited[key]
                else:os.environ.pop(key,None);selected[key]=None
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);terminated=True
            return {'status':'owned_idle_orchestrator_transition','previous_pid':pid,'command':command,
                    'current_process':current,'runtime_environment':selected,'gpu_uuid':uuid,
                    'scope':'no GPU child or active selected-device process stopped; original frozen numerical cohort retained'}
        finally:
            if paused and not terminated:
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
            if terminated:pynvml.nvmlShutdown()
        time.sleep(.5)
    pynvml.nvmlShutdown()
    raise RuntimeError('no safe idle boundary before bounded transition deadline')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous-pid',type=int,required=True);a=p.parse_args();a.root=a.root.resolve()
    original_session=load(a.root/'session.json')
    assert os.environ['SLURM_JOB_ID']==original_session['slurm_job_id']
    for name in ('h200_priority_campaign.py','h200_session.py'):
        assert sha(PACKAGE/'scripts'/name)==sha(Path(original_session['frozen_source_root'])/'scripts'/name), 'numerical/original supervisor source changed'
    budget=a.root/'results/budget_eager_b1_c4096.json'
    while True:
        if budget.exists():
            initial=load(budget)
            if initial.get('stages') and not initial['stages'][-1]['admitted']:break
        gate=a.root/'results/eager_core_disposition.json'
        if gate.exists() and load(gate).get('stop')!='budget':
            print('Original curve continued or finished; no transition required',flush=True);return
        time.sleep(1)
    completed=len(initial.get('observed_acquisition_wall_s',[]))
    assert completed in (1,2)
    original_bytes=budget.read_bytes()
    session=dict(original_session)
    session['closeout_buffer_s']=900
    session['launch_cutoff']=(dt.datetime.fromisoformat(session['allocation_end'])-dt.timedelta(minutes=15)).isoformat()
    session['operational_resume']={'original_session_sha256':sha(a.root/'session.json'),
        'reason':'critical raw evidence already persistent; verified previous CPU archives/closeouts take seconds; prioritize complete current comparison',
        'changes':'30minute reserve reduced to15minutes; no scientific or numerical setting changed'}
    resume_session=a.root/'session_finish_curve.json';write(resume_session,session)
    worker=Worker(a.root,resume_session);worker.hs.verify_allocation()
    plan=allowance(initial['observed_acquisition_wall_s'],completed,
                   worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time())
    write(a.root/'results/eager_curve_resume_budget.json',{'status':'reviewed','original_budget_sha256':sha(budget),'stages':[plan]})
    if not plan['admitted'] or plan['available_before_closeout_s']-plan['remaining_bundle_estimate_s']<30:
        print('Measured remaining bundle does not fit; original worker left unchanged',flush=True);return
    safe_wait=min(180,plan['available_before_closeout_s']-plan['remaining_bundle_estimate_s']-30)
    try:transition=stop_idle_parent(a.previous_pid,a.root,worker.uuid,time.time()+safe_wait)
    except RuntimeError as error:
        write(a.root/'results/eager_curve_resume_transition.json',{'status':'no_safe_transition_original_unchanged','error':str(error)})
        return
    write(a.root/'results/eager_curve_resume_transition.json',transition)
    os.environ['H200_SESSION_CONFIG']=str(resume_session)
    (a.root/'results/eager_curve_budget_before_resume.json').write_bytes(original_bytes)
    gate=a.root/'results/eager_core_disposition.json'
    if gate.exists():gate.rename(a.root/'results/eager_core_disposition_before_resume.json')
    from h200_review_allocation import case_records
    records,issues,interrupted=case_records(a.root)
    assert not issues, str(issues)
    worker.records=records;worker.rebind_step=True;worker.capture_bind=False
    proof=next(r for r in records if r['pilot'] and r['stock'] and r['engine']=='eager')
    worker.expected=load(proof['run'])['cases'][0]['trials'][0]['output_ids']
    assert all(r['source_identity']==worker.settings['frozen_source_identity'] and r['gpu_uuid']==worker.uuid for r in records)
    for acquisition in range(completed):
        assert captured_acquisition(records,acquisition), 'original acquisition incomplete'
    state={'status':'resuming_original_cohort','original_budget_sha256':sha(a.root/'results/eager_curve_budget_before_resume.json'),
           'source_freeze_identity':worker.settings['frozen_source_identity'],'gpu_uuid':worker.uuid,
           'stages':[],'scientific_gates':'unchanged3acquisitionsx10trials, exact outputs, all NSYS/SM/source/clock/isolation gates'}
    with (a.root/'analysis/report.md').open('a') as stream:
        stream.write('\n- Scheduling-only continuation: original remaining-bundle estimate declined; use the maximum observed acquisition plus60s per remaining acquisition and a15minute CPU-closeout reserve before the actual Slurm end. Same UUID, original frozen numerical sources, build environment and all scientific gates retained. Critical raw files already persist on shared storage. See `../session_finish_curve.json`, `../results/eager_curve_resume_transition.json` and `../results/eager_curve_resume_budget.json`.\n')
    publish(a.root,'H200 core priority: preserve measured broad acquisition before scheduling-only continuation')
    observed=list(initial['observed_acquisition_wall_s']);stop='budget'
    for acquisition in range(completed,3):
        plan=allowance(observed,acquisition,worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time())
        state['stages'].append(plan);write(a.root/'results/eager_curve_resume_budget.json',state)
        if not plan['admitted']:break
        began=time.monotonic()
        worker.case('eager',1.,acquisition,stock=True)
        worker.case('eager',1.,acquisition)
        for index,fraction in enumerate(FRACTIONS[1:]):
            worker.case('eager',fraction,acquisition)
            if index==1:worker.case('eager',1.,acquisition,suffix='_interleaved')
        worker.case('eager',1.,acquisition,suffix='_end')
        analyze_curve(a.root,worker.records,'eager')
        observed.append(time.monotonic()-began);state['observed_acquisition_wall_s']=observed
        write(a.root/'results/eager_curve_resume_budget.json',state)
        publish(a.root,f'H200 original eager cohort: completed broad acquisition{acquisition} with all gates retained')
    result=analyze_curve(a.root,worker.records,'eager')
    complete=all(captured_acquisition(worker.records,a) for a in range(3))
    if len(observed)==3:stop=(None if result['complete_qualified_curve'] else 'quality') if complete else 'producer'
    write(gate,{'status':'eager_curve_continuation_disposition','complete_acquisition_set':complete,
                'complete_qualified_eager_curve':result['complete_qualified_curve'],'stop':stop,
                'allow_context_expansion':complete,'scope':'same original numerical cohort; no cross-allocation/device pooling'})
    if complete and not result['complete_qualified_curve']:
        limitation(a.root,'eager_curve_resumed_quality','Complete acquisition set captured but scientific qualification remains failed',
                    'results/curve_eager_b1_c4096.json','All trials/references and original clock/source gates retained; no favorable resampling',
                    'Inspect specific quality failures; no NVIDIA threshold without qualification')
    publish(a.root,'H200 original eager curve continuation: publish complete or explicit partial disposition')


if __name__=='__main__':main()
