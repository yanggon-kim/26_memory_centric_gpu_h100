#!/usr/bin/env python3
"""Fixed three-acquisition producer comparison on an already owned pair.

No Slurm submission. A completed current-allocation functional cohort provides
correctness/physical-pair prerequisites. An earlier verified pilot supplies
runtime bounds only; its samples never enter this comparison's estimates.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import traceback
from h200_priority_campaign import PACKAGE,load,write,sha,publish,limitation
from h200_native_transport_pair import choose_ports
from h200_native_functional_campaign import run_one,run_pair,copy_role
from h200_native_model_followup import copy_runtime_cache
from h200_producer_timing_followup import admission,ENTRY,LABEL
from h200_producer_quant_budget import budget


def main(args):
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    source=args.functional_root.resolve();root=args.root.resolve();pilot=args.pilot_root.resolve()
    assert all(p.is_relative_to(PACKAGE/'h200') for p in (source,root,pilot)) and not root.exists()
    estimate=budget(pilot)
    assert estimate['all_bounds_within_existing660second_watchdog'],'pilot estimate exceeds existing role watchdog; no extension authorized by this adapter'
    bases=[source/f'gpu{i}' for i in range(2)];sessions=[load(base/'session.json') for base in bases]
    job=os.environ.get('SLURM_JOB_ID')
    raw=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True,timeout=25)
    original=load(source/'model/pair.json');cert=source/'model/results/native_model_CPU_review.json'
    outcome=load(source/'results/worker_outcome.json')
    admission(sessions,[load(base/'results/current_process.json') for base in bases],job,socket.gethostname(),os.getuid(),
        raw,outcome,original,load(cert))
    assert sha(cert)==outcome['model_review_sha256'] and original['context']==estimate['context']
    assert load(Path(original['reference_run']))['engine_settings']==estimate['engine_settings']
    fields=dict(item.split('=',1) for item in raw.split() if '=' in item)
    cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
        *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
    assert cutoff-time.time()>=estimate['required_s'],'complete fixed3x10cohort and closeout budget do not fit'
    names=(ENTRY,Path(__file__).name,'h200_producer_quant_budget.py','h200_review_producer_quant.py','shoreline_statistics.py')
    for session in sessions:
        for name in names:assert sha(Path(session['frozen_source_root'])/'scripts'/name)==sha(PACKAGE/'scripts'/name)
    with contextlib.ExitStack() as stack:
        for session in sessions:
            stream=stack.enter_context((Path(session['workspace'])/'producer.lock').open('a'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import pynvml as nv
        nv.nvmlInit()
        try:
            for session in sessions:assert not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByUUID(session['gpu_uuid']))
        finally:nv.nvmlShutdown()
        root.mkdir()
        for name in ('environment','results','logs','analysis'):(root/name).mkdir()
        (root/'analysis/report.md').write_text('# Fixed3x10matched producer comparison\n\nThree fresh process acquisitions, each stock-before/default native producer+consumer/stock-after, three warmups and ten timed requests per condition. The earlier pilot is used only for runtime admission. Primary ratio anchors the same-acquisition stock-before; stock-after is an independent drift guard. No HTTP/deployed TTFT, end-to-end goodput, physical traffic or primary energy claim.\n')
        (root/'environment/controller_source.py').write_bytes(Path(__file__).read_bytes())
        write(root/'environment/admission.json',{'job_id':job,'slurm':raw,'cutoff_unix_s':cutoff,'available_s':cutoff-time.time(),
            'functional_certificate':str(cert),'functional_certificate_sha256':sha(cert),'runtime_estimate':estimate,
            'scope':'Pilot data estimate startup/timeouts only; no cross-UUID performance pooling or pilot samples in comparison'})
        captures=[];result={'status':'running_fixed3x10','performance_qualified':False,'serving_qualified':False}
        def prepare(base,target,tag):
            copy_role(base,target);session=load(target/'session.json')
            workspace=Path('/tmp')/('h200_quant_'+str(os.getuid())+'_'+job+'_'+tag)
            workspace.mkdir(exist_ok=False)
            receipt=copy_runtime_cache(Path(session['workspace']),workspace)
            session['workspace']=str(workspace);write(target/'session.json',session)
            write(target/'environment/cache_copy.json',receipt)
        def save():write(root/'results/acquisitions.json',{'mode':'comparison','planned_acquisitions':3,'warmups':3,'timed':10,'captures':captures})
        try:
            publish(root,'H200 producer3x10: publish fixed cohort, pilot-only runtime bounds and sources')
            for acquisition in range(3):
                still_required=(3-acquisition)*sum(estimate['phase_timeout_s'].values())+180
                assert cutoff-time.time()>=still_required,'remaining fixed cohort no longer fits protected time'
                acquisition_root=root/('a'+str(acquisition))
                pair={key:original[key] for key in ('job_id','hostname','host','context','generated_tokens',
                    'reference_run','reference_run_sha256','strict_HF_reference','strict_HF_reference_sha256')}
                ports=choose_ports()
                pair.update(protocol='matched_native_producer_host_timing_v1',synthetic_payload=False,mode='comparison',
                    acquisition=acquisition,warmup_trials=3,timed_trials=10,request_prefix='h200_quant_'+job+'_a'+str(acquisition)+'_'+str(time.time_ns()),
                    roles={role:{'gpu_uuid':session['gpu_uuid'],'port':port} for role,session,port in zip(('producer','consumer'),sessions,ports)},
                    functional_CPU_review=str(cert),functional_CPU_review_sha256=sha(cert))
                for condition in ('stock_before','native','stock_after'):
                    current=acquisition_root/condition
                    for name in ('results','logs','analysis','shared'):(current/name).mkdir(parents=True,exist_ok=True)
                    declaration=dict(pair,shared_directory=str(current/'shared'),deadline_unix_s=cutoff)
                    timeout=estimate['phase_timeout_s'][condition]
                    if condition=='native':
                        for role,base in zip(('producer','consumer'),bases):prepare(base,current/role,'a'+str(acquisition)+'_'+condition+'_'+role)
                        records=run_pair(current,declaration,ENTRY,LABEL,'timing',timeout,cutoff)
                        for role,record in zip(('producer','consumer'),records):
                            captures.append(dict(record,acquisition=acquisition,condition=condition,role=role,base=str(current/role)))
                    else:
                        prepare(bases[0],current,'a'+str(acquisition)+'_'+condition)
                        descriptor=current/'pair.json';write(descriptor,declaration)
                        run_one(current,LABEL,ENTRY,['--pair-config',descriptor,'--role','stock',
                            '--gpu-uuid',sessions[0]['gpu_uuid'],'--output',current/'runs/timing'],timeout,cutoff)
                        path=current/'runs/timing/run.json';process=current/'results/processes'/(LABEL+'.json')
                        captures.append({'acquisition':acquisition,'condition':condition,'role':'stock','base':str(current),
                            'run':str(path),'run_sha256':sha(path),'process':str(process),'process_sha256':sha(process)})
                    save()
                publish(root,'H200 producer3x10: publish complete acquisition'+str(acquisition)+' without dropping any trials')
            result['status']='captured_fixed3x10_pending_review'
        except BaseException as error:
            result.update(status='blocked_or_partial_fixed3x10',error=str(error),traceback=traceback.format_exc())
            write(root/'results/failure.json',result)
            limitation(root,'producer_quant_capture',str(error),'results/failure.json; results/acquisitions.json',
                'Prespecified three fresh acquisitions/ten timed per condition; all source/clock/isolation/time gates retained',
                'Preserve complete and partial acquisitions; no ratio unless all three qualify on the same physical pair. Investigate the first failure and prepare a bounded repair or next eligible comparison.')
            raise
        finally:
            save()
            from h200_review_producer_quant import review
            try:
                reviewed=review(root);path=root/'results/producer_quant_CPU_review.json'
                assert not path.exists();write(path,reviewed)
                result.update(review=str(path),review_sha256=sha(path),performance_qualified=reviewed['performance_qualified'])
                if result['status']=='captured_fixed3x10_pending_review':result['status']='qualified_fixed3x10' if reviewed['performance_qualified'] else 'captured_rejected_fixed3x10'
                lines=['\n## Fixed cohort CPU review\n',result['status'],
                    '\nPrimary native/stock-before producer host ratio: '+str(reviewed['qualified_producer_ratio']),
                    '\nAll raw per-condition host timings, clocks, drift, source and correctness dispositions are in `../results/producer_quant_CPU_review.json`. Pilot samples in estimate:0. Primary J/token:NA; serving/goodput remains unmeasured.']
                with (root/'analysis/report.md').open('a') as stream:stream.write('\n'.join(lines)+'\n')
                with (root/'analysis/report.md').open('a') as stream:
                    stream.write('\n| Acquisition | Condition | Role | Raw median submit-to-final-return host ms | Clock gate |\n|---:|---|---|---:|---|\n')
                    for row in reviewed['rows']:
                        stream.write('| '+str(row['acquisition'])+' | '+row['condition']+' | '+row['role']+' | '+str(statistics.median(row['metrics']['submit_to_final_step_return_host_ms']))+' | '+str(row['clock']['passed'])+' |\n')
                    stream.write('\nStock/producer rows return one token; the consumer row spans63outputs and retains its first handoff query and following62host calls separately in JSON. Raw medians remain descriptive when a gate rejects; only the explicit qualified ratio/CI supports a performance comparison.\n')
                if not reviewed['performance_qualified']:
                    limitation(root,'producer_quant_CPU_review','Fixed cohort incomplete or original source/clock/isolation/drift gates rejected; qualified producer ratio NA',
                        'results/producer_quant_CPU_review.json','Preserved every acquisition and timed request; primary denominator fixed before collection; original gates and paired acquisition bootstrap retained',
                        'Inspect exact rejected gates and raw values. Do not pool UUIDs, drop acquisitions or resample for favorable clocks. Bound any methodological repair before another identified cohort.')
            except Exception as error:
                result.update(review_error=str(error),review_traceback=traceback.format_exc())
                limitation(root,'producer_quant_reader',str(error),'results/worker_outcome.json; results/acquisitions.json',
                    'Independent read-only review of actual source/process/output/window records',
                    'Inspect the reader failure and preserved raw evidence; never treat exit0 as performance qualification')
            result.update(exact_next_command=sessions[0]['python']+' '+str(PACKAGE/'scripts/h200_review_producer_quant.py')+' --root '+str(root),
                next_required='Read-only verification of the actual fixed cohort; inspect blocked/rejected gates and preserve all partial acquisitions. After allocation closeout choose a bounded repair or next eligible comparison; never pool physical UUIDs or repeat a complete comparison by default.')
            write(root/'results/worker_outcome.json',result)
            publish(root,'H200 producer3x10: publish complete/rejected/partial fixed cohort and interpretation')
            try:
                subprocess.run([sessions[0]['python'],PACKAGE/'scripts/archive_h200_priority.py','--root',root,
                    '--destination','/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives'],check=True,timeout=180)
            except Exception as error:
                write(root/'results/archive_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
                limitation(root,'producer_quant_archive',str(error),'results/archive_failure.json','Bounded checksum-verified transfer, originals retained',
                    'Resume transfer verification from preserved originals; no successful archive claim')
                publish(root,'H200 producer3x10: preserve archive transfer failure');raise
            publish(root,'H200 producer3x10: publish verified archive receipt')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--functional-root',type=Path,required=True);parser.add_argument('--pilot-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    try:main(args)
    except BaseException as error:
        root=args.root.resolve()
        if root.is_relative_to(PACKAGE/'h200') and not root.exists():
            (root/'results').mkdir(parents=True);(root/'analysis').mkdir()
            (root/'analysis/report.md').write_text('# Producer3x10admission failed\n\nNo new GPU role was launched.\n')
            write(root/'results/admission_failure.json',{'error':str(error),'traceback':traceback.format_exc(),'GPU_roles_launched':0})
            limitation(root,'producer_quant_admission',str(error),'results/admission_failure.json',
                'Current owned functional gates, separately verified pilot runtime estimate, fixed3x10budget and source equality before any role',
                'Keep existing/queued cases and partial results; inspect the failed gate before a new eligible complete cohort')
            publish(root,'H200 producer3x10: preserve prelaunch admission limitation')
        raise
