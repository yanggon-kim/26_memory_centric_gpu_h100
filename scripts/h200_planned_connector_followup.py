"""Prospective exact-job continuation after a closed producer comparison.

No Slurm submission, deadline extension or change to completed evidence.
Missing/disabled plans preserve the original entrypoint behavior.
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

ENTRY='h200_connector_capture_campaign.py'
REQUIRED=1860
SOURCE_NAMES=(ENTRY,'h200_native_connector_capture.py','h200_connector_capture_gate.py',
    'h200_connector_spans.py','h200_connector_timeline.py','h200_review_connector_capture.py',
    Path(__file__).name,'h200_producer_quant_entrypoint.py')


def completed_comparison_gate(outcome,review,acquisitions,archive):
    assert outcome['status'] in ('qualified_fixed3x10','captured_rejected_fixed3x10')
    assert not outcome.get('review_error')
    assert review['complete'] is True and review['source_gate'] is True and review['same_physical_pair_gate'] is True
    assert not review['errors'] and review['pilot_samples_in_estimator']==0
    expected={(a,condition,role) for a in range(3) for condition,roles in
        (('stock_before',('stock',)),('native',('producer','consumer')),('stock_after',('stock',))) for role in roles}
    assert acquisitions['mode']=='comparison' and acquisitions['planned_acquisitions']==3
    assert acquisitions['warmups']==3 and acquisitions['timed']==10
    keys=[(r['acquisition'],r['condition'],r['role']) for r in acquisitions['captures']]
    assert len(keys)==12 and set(keys)==expected
    keys=[(r['acquisition'],r['condition'],r['role']) for r in review['rows']]
    assert len(keys)==12 and set(keys)==expected
    assert all(len(r['metrics']['submit_to_final_step_return_host_ms'])==10 for r in review['rows'])
    assert archive['status']=='archive_transfer_verified' and archive['every_archive_member_verified'] is True
    assert archive['archive_sha256']==archive['destination_sha256'] and archive['regular_files']>0
    # Clock rejection remains a rejected producer comparison. It does not
    # substitute for, or prohibit, a separately admitted functional profiler pilot.
    return True


def run_if_planned(core,quant):
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():return {'status':'no_owned_job_plan'}
    path=PACKAGE/'h200/20261127_resident_preparation/results'/('connector_followup_'+job+'.json')
    if not path.exists():return {'status':'no_plan'}
    plan=load(path)
    if plan.get('enabled') is not True:return {'status':'disabled_plan'}
    core=Path(core).resolve();quant=Path(quant).resolve()
    assert core.is_relative_to(PACKAGE/'h200') and quant.is_relative_to(PACKAGE/'h200')
    receipt=quant/'results/planned_connector_followup.json'
    if receipt.exists():return {'status':'already_recorded','receipt':str(receipt),'sha256':sha(receipt)}
    result={'status':'checking_optional_connector_admission','GPU_controller_invoked':False,
        'plan':str(path),'plan_sha256':sha(path),
        'scope':'Original producer comparison is already closed/published/archived. New instrumented functional root and archive remain separate; this continuation metadata postdates the original producer archive.'}
    try:
        assert plan['job_id']==job and plan['GPUs']==2 and plan['entry']==ENTRY
        assert set(plan['source_sha256'])==set(SOURCE_NAMES),'complete declared continuation source set required'
        assert plan['context']==load(core/'model/pair.json')['context']
        sessions=[load(core/f'gpu{i}/session.json') for i in (0,1)]
        assert all(s['slurm_job_id']==job for s in sessions)
        for name,digest in plan['source_sha256'].items():
            assert sha(PACKAGE/'scripts'/name)==digest
            for session in sessions:assert sha(Path(session['frozen_source_root'])/'scripts'/name)==digest
        outcome=load(quant/'results/worker_outcome.json');review_path=Path(outcome['review'])
        assert sha(review_path)==outcome['review_sha256']
        archives=list(quant.glob('results/archive_transfer_*.json'));assert len(archives)==1
        archived=load(archives[0]);assert Path(archived['source_root']).resolve()==quant
        completed_comparison_gate(outcome,load(review_path),load(quant/'results/acquisitions.json'),archived)
        manifest=load(archived['source_manifest'])
        assert any(r['path']=='results/producer_quant_CPU_review.json' and r['sha256']==outcome['review_sha256'] for r in manifest['files'])
        raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
        fields=dict(v.split('=',1) for v in raw.split() if '=' in v)
        assert fields['JobState']=='RUNNING' and fields['JobId']==job
        assert fields['NodeList']==socket.gethostname().split('.')[0]
        assert fields['UserId'].endswith('('+str(os.getuid())+')')
        cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
            *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
        result.update(available_s=cutoff-time.time(),required_s=REQUIRED,cutoff_unix_s=cutoff)
        if result['available_s']<REQUIRED:
            result['status']='deferred_complete_connector_pilot_does_not_fit'
            write(receipt,result)
            limitation(quant,'post_quant_connector_budget','Complete instrumented connector pilot does not fit remaining protected time',
                'results/planned_connector_followup.json','Closed and archived producer cohort first; full1860second continuation bound retained',
                'After allocation closeout select a fresh eligible dedicated pilot using new owned-pair/current strict functional gates. Preserve original completed/rejected cohort; no shortened proof or profiler coverage claim.')
            return result
        target=core.with_name(core.name+'_connector_capture');assert not target.exists()
        command=[sys.executable,PACKAGE/'scripts'/ENTRY,'--functional-root',core,'--root',target]
        result.update(status='admitted_pending_independent_controller',root=str(target),command=list(map(str,command)))
        write(receipt,result)
        publish(quant,'H200 connector continuation: publish closed producer boundary and separate instrumented admission')
        result['GPU_controller_invoked']=True;write(receipt,result)
        process=subprocess.run(list(map(str,command)),cwd=PACKAGE)
        result.update(status='controller_returned' if process.returncode==0 else 'controller_failed_or_partial',controller_exit_code=process.returncode,
            qualification='Exit status is not source/functional/CUDA coverage or scientific qualification; inspect separate controller review and archive.')
        if (target/'results/worker_outcome.json').exists():result['outcome']=str(target/'results/worker_outcome.json')
        if process.returncode!=0:
            limitation(quant,'post_quant_connector_outcome','Separate bounded connector controller failed or was partial',
                str(target)+'; results/planned_connector_followup.json','Preserved closed producer cohort; controller independently rechecks source, ownership, idleness and complete time',
                'Inspect separate raw/partial profiler data and exact failed gate; preserve all existing evidence and bound repairs before the next eligible case.')
    except BaseException as error:
        result.update(status='connector_followup_admission_or_execution_failed',error=str(error),traceback=traceback.format_exc())
        write(receipt,result)
        limitation(quant,'post_quant_connector_admission',str(error),'results/planned_connector_followup.json',
            'Exact-job plan; complete original3x10source/physical-pair review and verified archive before separate source/time/idle controller',
            'Preserve the completed producer comparison; inspect failed prerequisite before any future GPU launch. No primary evidence is rewritten or implicitly qualified.')
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:
        write(receipt,result)
        publish(quant,'H200 connector continuation: publish actual separate pilot disposition')
    return result
