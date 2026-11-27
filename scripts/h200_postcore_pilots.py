#!/usr/bin/env python3
"""Use an owned, finished core worker's idle GPU for bounded functional pilots."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from h200_priority_campaign import PACKAGE,Worker,load,write,publish,limitation,config,valid_tokens,sha


def validate_idle(core,current,outcome,gpu_pids):
    assert core.get('complete_acquisition_set') is True, 'broad repeated curve must be captured first'
    assert outcome.get('status')=='resident_curve_disposition', 'original worker has not returned'
    assert current.get('status')=='idle' and not gpu_pids, 'selected core GPU is not idle'
    assert not outcome.get('conditioned_followup') or outcome['conditioned_followup']['status'] in (
        'deferred_budget','orchestrator_finished','orchestrator_failed'), 'conditioned worker may still be active'


def pilots(worker):
    from h200_resident_campaign import build_environment
    build_environment(worker)
    inv=worker.execute('inventory',[worker.py,worker.script/'inventory.py','--gpu-uuid',worker.uuid,
        '--check-cuda','--output',worker.root/'environment/inventory.json'],timeout=180)
    assert inv['status']=='complete', 'post-core inventory failed'
    results=[]
    for label,script,context,extra,estimate in (
        ('P3a_functional_policies','h200_serving_policy_pilot.py',4096,[],540),
        ('P4a_functional_prefill','h200_prefill_resident.py',1024,['--pilot','--acquisition','0'],440)):
        available=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time()
        admission={'id':label,'estimate_s':estimate,'margin_s':60,'available_s':available,
            'admitted':available>=estimate+60,
            'basis':'about twice observed prior functional producer+strict-reference time; analytic scheduling only, new source/UUID requires its own actual gates'}
        write(worker.root/'results'/(label+'_budget.json'),admission)
        if not admission['admitted']:
            results.append({'id':label,'status':'deferred_budget','available_s':available})
            write(worker.root/'results/functional_pilots.json',{'results':results})
            continue  # The smaller independent prefill pilot may still fit.
        cfg=config(worker.root,'graph',context,3,1);output=worker.root/'runs'/label
        proc=worker.execute(label,[worker.py,worker.script/script,'--config',cfg,'--gpu-uuid',worker.uuid,
            '--output',output]+extra,timeout=600)
        path=output/'run.json'
        passed=proc['status']=='complete' and path.is_file() and bool(valid_tokens(load(path)))
        if passed:passed=worker.reference(path,label)
        row={'id':label,'status':'functional_passed' if passed else 'blocked_functional_or_reference',
            'producer_status':proc['status'],'strict_reference_passed':bool(passed),
            'context':context,'batch':1,'gpu_uuid':worker.uuid,'source_identity':worker.settings['frozen_source_identity'],
            'run':str(path),'run_sha256':sha(path) if path.exists() else None,
            'qualified_quantitative_comparison':False,
            'scope':'bounded instrumented functional proof only; not3acquisitions, loaded serving, tuning, SLOs, energy or physical traffic'}
        if passed:
            receipt=output/('policy_pilot.json' if label.startswith('P3') else 'prefill_resident.json')
            data=load(receipt)
            assert data['status']==('three_functional_policies_passed' if label.startswith('P3') else 'complete')
            row.update(receipt=str(receipt),receipt_sha256=sha(receipt))
        else:
            limitation(worker.root,label,'Bounded post-core functional/reference pilot failed',
                str(path),'Versioned adapter; original completed core retained;600second producer limit and strict dynamic/StaticCache gate',
                'Inspect preserved failure before bounded repair; do not promote a functional pilot to a quantitative family result')
        results.append(row);write(worker.root/'results/functional_pilots.json',{'results':results})
        with (worker.root/'analysis/report.md').open('a') as stream:
            stream.write(f"\n## {label}\n\n{row['status']}. {row['scope']}. Evidence: `{row['run']}`; strict reference: {row['strict_reference_passed']}.\n")
        publish(worker.root,'H200 post-core '+label+': publish functional gate and remaining quantitative scope')
    return {'status':'functional_pilots_disposition','results':results}


def main(a):
    old=load(a.source_session);prior=a.source_session.resolve().parent
    assert os.environ.get('SLURM_JOB_ID')==old['slurm_job_id']
    assert socket.gethostname()==old['expected_hostname']
    state=subprocess.check_output(['scontrol','show','job','-o',old['slurm_job_id']],text=True)
    assert 'JobState=RUNNING' in state and f'({os.getuid()})' in state and 'gres/gpu:h200=2' in state
    import pynvml
    pynvml.nvmlInit()
    try:
        handle=pynvml.nvmlDeviceGetHandleByUUID(old['gpu_uuid'])
        gpu_pids=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
    finally:pynvml.nvmlShutdown()
    core=load(prior/'results/eager_core_disposition.json')
    outcome=load(prior/'results/worker_outcome.json');current=load(prior/'results/current_process.json')
    validate_idle(core,current,outcome,gpu_pids)
    lock=(Path(old['workspace'])/'producer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.root=a.root.resolve();a.root.mkdir(exist_ok=False)
    for folder in ('environment','results','logs','configs','runs','profiles','analysis'):(a.root/folder).mkdir()
    (a.root/'analysis/report.md').write_text(('# Post-core resident prefill acquisitions\n\n' if a.prefill_curve else '# Post-core functional serving/prefill pilots\n\n')+'Same allocated GPU0, new frozen source/workspace after the original core worker returned. Original core observations are retained separately. The other allocated GPU may independently run context cases; shared host CPU concurrency remains part of this environment. No new allocation, clock/power or system changes.\n')
    write(a.root/'results/idle_transition.json',{'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'gpu_uuid':old['gpu_uuid'],'prior_session':str(a.source_session),'prior_session_sha256':sha(a.source_session),
        'core':core,'worker_outcome':outcome,'current_process':current,'gpu_pids':gpu_pids,
        'scope':'original core returned and selected GPU idle; old producer lock held until post-core controller exits'})
    session=dict(old);session['workspace']=old['workspace']+(f'_prefill_c{a.context}' if a.prefill_curve else '_postcore')
    Path(session['workspace']).mkdir(exist_ok=False)
    session['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip()
    session['protocol']='resident_prefill_3x10_v2' if a.prefill_curve else 'postcore_functional_pilots_v1'
    for key in ('frozen_source_root','frozen_source_identity'):session.pop(key,None)
    path=a.root/'session.json';write(path,session)
    subprocess.run([session['python'],PACKAGE/'scripts/freeze_shoreline.py','--session',path,'--activate'],check=True)
    worker=Worker(a.root,path)
    try:
        if a.prefill_curve:
            from h200_prefill_campaign import capture
            outcome=capture(worker,a.context)
        else:outcome=pilots(worker)
        write(a.root/'results/worker_outcome.json',outcome)
    except BaseException as error:
        write(a.root/'results/worker_outcome.json',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        limitation(a.root,'postcore_controller','Post-core controller failed: '+str(error),
            'results/worker_outcome.json','Owned completed/idle transition; new frozen functional cohort',
            'Inspect first failure and preserve partial results; no quantitative result inferred')
        raise
    finally:
        publish(a.root,'H200 post-core functional controller: durable final disposition')
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prefill-curve',action='store_true')
    p.add_argument('--context',type=int,choices=(1024,),default=1024,help='first repeated prefill comparison only; expand after its review')
    main(p.parse_args())
