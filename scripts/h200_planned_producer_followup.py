#!/usr/bin/env python3
"""Job-specific continuation after a published, archived functional case.

Never submits/cancels a Slurm job, edits original evidence or extends deadlines.
The continuation must be explicitly enabled in a versioned plan for this exact
job. Otherwise standalone functional drivers keep their original behavior.
"""
import datetime as dt
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from h200_priority_campaign import PACKAGE,load,write,sha,publish,limitation

ENTRY='h200_producer_timing_followup.py'
REQUIRED_SECONDS=1920 # controller1860s admission plus60s metadata/publication margin


def admission(plan,job,context,fields,sessions,outcome,certificate,now):
    assert plan['job_id']==job and plan['context']==context and plan['enabled'] is True
    assert plan['entry']==ENTRY and plan['GPUs']==2
    assert fields['JobId']==job and fields['JobState']=='RUNNING'
    assert fields['UserId'].endswith('('+str(os.getuid())+')')
    assert fields['NodeList']==socket.gethostname().split('.')[0]
    assert 'gres/gpu:h200=2' in fields['AllocTRES'].split(',')
    assert len(sessions)==2 and len({s['gpu_uuid'] for s in sessions})==2
    assert all(s['slurm_job_id']==job and s['expected_hostname']==socket.gethostname() for s in sessions)
    assert len({s['frozen_source_identity'] for s in sessions})==1
    assert outcome['status']=='functional_prerequisites_passed'
    assert certificate['status']=='native_model_functional_review_passed' and certificate['job_id']==job
    cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
        *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
    available=cutoff-now
    return {'eligible':available>=REQUIRED_SECONDS,'required_s':REQUIRED_SECONDS,
        'available_s':available,'cutoff_unix_s':cutoff,
        'reason':'complete pilot budget fits' if available>=REQUIRED_SECONDS else 'insufficient protected budget; preserve completed functional case'}


def run_if_planned(context):
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():return {'status':'no_owned_job_plan'}
    plan_path=PACKAGE/'h200/20261127_resident_preparation/results'/('same_allocation_followup_'+job+'.json')
    if not plan_path.exists():return {'status':'no_plan'}
    plan=load(plan_path)
    if not plan.get('enabled'):return {'status':'disabled_plan'}
    if plan.get('job_id')!=job or plan.get('context')!=context:return {'status':'nonmatching_plan'}
    roots=list((PACKAGE/'h200').glob('*_job'+job+'_native_functional_c'+str(context)))
    assert len(roots)==1
    core=roots[0];receipt=core/'results/planned_producer_followup.json'
    result={'status':'checking_admission','GPU_followup_launched':False,'plan':str(plan_path),'plan_sha256':sha(plan_path),
        'core_evidence_scope':'Original functional case, model certificate and raw runs remain unchanged. The verified core archive predates this separate continuation metadata; continuation results use their own sibling root and archive.'}
    try:
        sessions=[load(core/f'gpu{i}/session.json') for i in range(2)]
        outcome=load(core/'results/worker_outcome.json')
        cert=Path(outcome['model_review']);assert sha(cert)==outcome['model_review_sha256']
        assert sys.executable==sessions[0]['python']==sessions[1]['python']
        assert sha(PACKAGE/'scripts'/ENTRY)==plan['entry_sha256']
        for session in sessions:
            scripts=Path(session['frozen_source_root'])/'scripts'
            assert sha(scripts/ENTRY)==plan['entry_sha256']
            assert sha(scripts/Path(__file__).name)==sha(Path(__file__))
        slurm=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
        fields=dict(item.split('=',1) for item in slurm.split() if '=' in item)
        gate=admission(plan,job,context,fields,sessions,outcome,load(cert),time.time())
        result.update(admission=gate,slurm=slurm)
        if not gate['eligible']:
            result['status']='deferred_insufficient_time'
            write(receipt,result)
            limitation(core,'post_functional_producer_budget',gate['reason'],'results/planned_producer_followup.json',
                'Original functional case completed, published and archived first; require32minutes for a complete separately supervised producer pilot',
                'Reuse the qualified functional software evidence within scope; prepare the next complete producer comparison after this allocation closes. Never weaken source/clock/output/isolation gates or start an incomplete contrast.')
            return result
        followup=core.with_name(core.name+'_producer_timing_pilot')
        assert not followup.exists()
        command=[sys.executable,PACKAGE/'scripts'/ENTRY,'--functional-root',core,'--root',followup]
        result.update(status='admitted_pending_controller',followup_root=str(followup),command=list(map(str,command)))
        write(receipt,result)
        with (core/'analysis/report.md').open('a') as stream:
            stream.write('\n## Separate same-allocation producer continuation\n\nOriginal functional comparison is closed, published and archived. A job-specific versioned plan admits the separate stock/native/stock pilot only after complete functional gates and32minutes of protected time; its controller rechecks actual idle ownership and its own31minute bound. Continuation metadata postdates the original core archive. New pilot evidence and archive will use `'+str(followup)+'`.\n')
        publish(core,'H200 continuation: publish closed-core boundary and separate producer admission')
        # The controller still rechecks the actual allocation, idleness,
        # complete time bound and all frozen sources before any GPU role.
        result.update(controller_invoked=True,GPU_followup_launched=None,
            GPU_launch_evidence_scope='See the independent controller admission and monitored process records; controller invocation/exit0 alone never proves a GPU launch or successful measurement')
        process=subprocess.run(list(map(str,command)),cwd=PACKAGE)
        result.update(status='followup_controller_returned' if process.returncode==0 else 'followup_failed_or_partial',
            controller_exit_code=process.returncode)
        if process.returncode!=0:
            limitation(core,'post_functional_producer_outcome','Separately supervised producer continuation failed or was partial',
                str(followup)+'; results/planned_producer_followup.json',
                'Preserved original successful functional case; launched only the source-verified bounded independent controller',
                'Inspect the producer root first failing stage and preserved partial data. Publish actual disposition and select a bounded repair or next eligible case; do not call functional success a producer performance result.')
    except BaseException as error:
        result.update(status='followup_admission_or_execution_failed',error=str(error),traceback=traceback.format_exc())
        write(receipt,result)
        limitation(core,'post_functional_producer_admission',str(error),'results/planned_producer_followup.json',
            'Job-specific plan; original functional stage is already closed and unchanged; source/time/ownership gates remain strict',
            'Inspect the actual receipt before retrying a separate eligible complete pilot; no primary functional evidence is overwritten')
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:
        write(receipt,result)
        publish(core,'H200 continuation: publish actual post-functional producer disposition')
    return result
