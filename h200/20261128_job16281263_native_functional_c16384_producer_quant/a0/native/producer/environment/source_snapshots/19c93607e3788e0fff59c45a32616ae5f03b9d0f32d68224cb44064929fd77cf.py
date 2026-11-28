#!/usr/bin/env python3
"""Owned same-allocation stock/native/stock producer timing pilot.

No Slurm submission. Requires a completed long-context functional campaign.
The first pilot is deliberately separate from the eventual3x10comparison.
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
from h200_priority_campaign import PACKAGE,load,write,sha,publish,limitation
from h200_native_transport_pair import validate_sources,choose_ports
from h200_native_functional_campaign import run_one,run_pair,copy_role
from h200_native_model_followup import copy_runtime_cache

ENTRY='h200_producer_timing_role.py'
LABEL='matched_producer_timing'


def admission(sessions,current,job,hostname,uid,slurm,outcome,pair,certificate):
    validate_sources(sessions,current,job,hostname,uid,slurm)
    assert outcome['status']=='functional_prerequisites_passed'
    assert certificate['status']=='native_model_functional_review_passed' and certificate['job_id']==job
    assert pair['job_id']==job and pair['hostname']==hostname
    assert pair['protocol']=='native_model_KV_long_context_functional_v2'
    assert pair['context'] in (16384,32704) and pair['generated_tokens']==64
    assert [pair['roles'][r]['gpu_uuid'] for r in ('producer','consumer')]==[s['gpu_uuid'] for s in sessions]
    assert len({s['frozen_source_identity'] for s in sessions})==1


def main(args):
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    source=args.functional_root.resolve();root=args.root.resolve()
    assert source.is_relative_to(PACKAGE/'h200') and root.is_relative_to(PACKAGE/'h200')
    assert not root.exists()
    bases=[source/f'gpu{i}' for i in range(2)]
    sessions=[load(base/'session.json') for base in bases]
    job=os.environ.get('SLURM_JOB_ID')
    slurm=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True,timeout=25)
    functional_pair=load(source/'model/pair.json')
    cert_path=source/'model/results/native_model_CPU_review.json';certificate=load(cert_path)
    admission(sessions,[load(base/'results/current_process.json') for base in bases],job,socket.gethostname(),os.getuid(),
        slurm,load(source/'results/worker_outcome.json'),functional_pair,certificate)
    assert sha(cert_path)==load(source/'results/worker_outcome.json')['model_review_sha256']
    fields=dict(item.split('=',1) for item in slurm.split() if '=' in item)
    cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
        *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
    required=1860
    assert cutoff-time.time()>=required,'Complete480+720+480second pilot and180second closeout margin do not fit'
    # The functional cohort must already contain this exact new role. Reusing
    # a prior functional certificate cannot silently add unpinned GPU code.
    for session in sessions:
        for name in (ENTRY,Path(__file__).name,'h200_review_producer_timing.py'):
            assert sha(Path(session['frozen_source_root'])/'scripts'/name)==sha(PACKAGE/'scripts'/name)
    with contextlib.ExitStack() as stack:
        for session in sessions:
            stream=stack.enter_context((Path(session['workspace'])/'producer.lock').open('a'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import pynvml as nv
        nv.nvmlInit()
        try:
            for session in sessions:
                assert not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByUUID(session['gpu_uuid']))
        finally:nv.nvmlShutdown()
        root.mkdir()
        for name in ('environment','results','logs','analysis'):(root/name).mkdir()
        (root/'analysis/report.md').write_text('# Matched native producer timing pilot\n\nSame-GPU stock before/native real transfer/stock after, each3warm+3timed. This pilot is not a3independent-acquisition performance comparison. Producer first step return, all-native-send readiness and consumer host decoder steps remain separate; file IPC is not HTTP/deployed TTFT or serving goodput.\n')
        source_bytes=Path(__file__).read_bytes();(root/'environment/controller_source.py').write_bytes(source_bytes)
        write(root/'environment/admission.json',{'job_id':job,'slurm':slurm,'available_s':cutoff-time.time(),'required_s':required,
            'cutoff_unix_s':cutoff,'functional_certificate':str(cert_path),'functional_certificate_sha256':sha(cert_path),
            'controller_sha256':sha(root/'environment/controller_source.py')})
        pair={key:functional_pair[key] for key in ('job_id','hostname','host','context','generated_tokens',
            'reference_run','reference_run_sha256','strict_HF_reference','strict_HF_reference_sha256')}
        ports=choose_ports()
        pair.update(protocol='matched_native_producer_host_timing_v1',synthetic_payload=False,mode='pilot',acquisition=0,
            warmup_trials=3,timed_trials=3,request_prefix='h200_producer_'+job+'_'+str(time.time_ns()),
            roles={role:{'gpu_uuid':session['gpu_uuid'],'port':port} for role,session,port in zip(('producer','consumer'),sessions,ports)},
            functional_CPU_review=str(cert_path),functional_CPU_review_sha256=sha(cert_path))
        captures=[];outcome={'status':'preparing_producer_timing_pilot','performance_qualified':False,'serving_qualified':False}
        def prepare(base,target,name):
            copy_role(base,target)
            session=load(target/'session.json')
            workspace=Path('/tmp')/('h200_producer_'+str(os.getuid())+'_'+job+'_'+name)
            workspace.mkdir(exist_ok=False)
            receipt=copy_runtime_cache(Path(session['workspace']),workspace)
            session['workspace']=str(workspace);write(target/'session.json',session)
            write(target/'environment/cache_copy.json',receipt)
        try:
            for condition in ('stock_before','native','stock_after'):
                condition_root=root/condition
                for name in ('results','logs','analysis','shared'):(condition_root/name).mkdir(parents=True,exist_ok=True)
                declaration=dict(pair,shared_directory=str(condition_root/'shared'),deadline_unix_s=cutoff)
                if condition=='native':
                    for role,base in zip(('producer','consumer'),bases):prepare(base,condition_root/role,condition+'_'+role)
                    publish(root,'H200 producer pilot: checkpoint completed controls before native pair')
                    records=run_pair(condition_root,declaration,ENTRY,LABEL,'timing',720,cutoff)
                    for role,record in zip(('producer','consumer'),records):
                        captures.append({'condition':condition,'role':role,'base':str(condition_root/role),
                            'run':record['run'],'run_sha256':record['run_sha256'],'process':record['process'],
                            'process_sha256':record['process_sha256']})
                else:
                    prepare(bases[0],condition_root,'control_'+condition)
                    descriptor=condition_root/'pair.json';write(descriptor,declaration)
                    publish(root,'H200 producer pilot: checkpoint frozen same-GPU stock control')
                    process=run_one(condition_root,LABEL,ENTRY,['--pair-config',descriptor,'--role','stock',
                        '--gpu-uuid',sessions[0]['gpu_uuid'],'--output',condition_root/'runs/timing'],480,cutoff)
                    path=condition_root/'runs/timing/run.json';proc_path=condition_root/'results/processes'/(LABEL+'.json')
                    captures.append({'condition':condition,'role':'stock','base':str(condition_root),'run':str(path),
                        'run_sha256':sha(path),'process':str(proc_path),'process_sha256':sha(proc_path)})
                write(root/'results/captures.json',{'mode':'pilot','captures':captures})
                publish(root,'H200 producer pilot: publish completed '+condition+' evidence')
            outcome.update(status='producer_timing_pilot_captured_pending_review',captures=len(captures))
            from h200_review_producer_timing import review
            reviewed=review(root)
            outcome.update(status=reviewed['status'],review=str(root/'results/producer_timing_CPU_review.json'),
                review_sha256=sha(root/'results/producer_timing_CPU_review.json'))
        except BaseException as error:
            outcome.update(status='blocked_or_partial_producer_timing_pilot',error=str(error),traceback=traceback.format_exc())
            write(root/'results/failure.json',outcome)
            limitation(root,'producer_timing_pilot',str(error),'results/failure.json',
                'Owned same-GPU stock/native/stock pilot with source/HF/functional gates; no observer or payload hashes in timed requests',
                'Retain partial cohorts; inspect first failing stage and prepare one bounded repair before a fresh complete pilot')
            raise
        finally:
            write(root/'results/captures.json',{'mode':'pilot','captures':captures})
            outcome.update(next_required='Independent CPU review, original clock/source/isolation/drift gates; then fresh3x10matched acquisitions before performance claims. Independent instrumented connector attribution remains separate.',
                exact_next_command='python -m json.tool '+str(root/'results/worker_outcome.json'))
            write(root/'results/worker_outcome.json',outcome)
            with (root/'analysis/report.md').open('a') as stream:stream.write('\n'+outcome['status']+'. '+outcome['next_required']+'\n')
            publish(root,'H200 producer pilot: preserve completed and unresolved stages')
            try:
                subprocess.run([sessions[0]['python'],PACKAGE/'scripts/archive_h200_priority.py','--root',root,
                    '--destination','/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives'],check=True,timeout=180)
            except Exception as error:
                write(root/'results/archive_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
                limitation(root,'producer_timing_archive',str(error),'results/archive_failure.json',
                    'Bounded archive transfer with original evidence retained',
                    'Retry checksum-verified transfer from preserved originals; no transfer success claimed')
                publish(root,'H200 producer pilot: preserve archive transfer failure')
                raise
            publish(root,'H200 producer pilot: publish verified closed-pilot archive receipt')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--functional-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    try:main(args)
    except BaseException as error:
        root=args.root.resolve()
        if root.is_relative_to(PACKAGE/'h200') and not root.exists():
            (root/'results').mkdir(parents=True);(root/'analysis').mkdir()
            (root/'analysis/report.md').write_text('# Producer timing pilot admission failed\n\nNo GPU role was launched by this controller.\n')
            write(root/'results/admission_failure.json',{'error':str(error),'traceback':traceback.format_exc(),'GPU_roles_launched':0})
            limitation(root,'producer_timing_admission',str(error),'results/admission_failure.json',
                'Require actual owned idle pair, completed long-context functional gates and complete pilot budget before launching',
                'Preserve the active/closed functional job. Inspect the failed admission and prepare a fresh eligible complete cohort; never weaken time/source/isolation gates')
            publish(root,'H200 producer pilot: preserve prelaunch admission limitation')
        raise
