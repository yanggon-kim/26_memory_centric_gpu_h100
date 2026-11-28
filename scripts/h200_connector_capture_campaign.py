"""Prospective owned-pair NSYS connector pilot; no Slurm submission.

Candidate must be promoted into a future frozen scripts tree before GPU use.
Original functional gates remain mandatory; profiler exit0 is not coverage.
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
import sys
import time
import traceback

from h200_priority_campaign import PACKAGE,load,write,sha,publish,limitation
from h200_native_transport_pair import validate_sources,choose_ports,stop_owned_role
from h200_native_functional_campaign import supervised_command,copy_role
from h200_native_model_followup import copy_runtime_cache

ENTRY='h200_native_connector_capture.py'
LABEL='native_model_KV_functional'
PAIR_BOUND=840
REQUIRED=1800
PROTOCOL='host_API_NVTX_metadata_range_v2'


def admission(sessions,currents,job,hostname,uid,slurm,outcome,certificate,pair,now):
    validate_sources(sessions,currents,job,hostname,uid,slurm)
    assert outcome['status']=='functional_prerequisites_passed'
    assert certificate['status']=='native_model_functional_review_passed' and not certificate['errors']
    assert certificate['job_id']==pair['job_id']==job and pair['hostname']==hostname
    assert pair['protocol']=='native_model_KV_long_context_functional_v2'
    assert pair['context'] in (16384,32704) and pair['generated_tokens']==64
    assert [pair['roles'][r]['gpu_uuid'] for r in ('producer','consumer')]==[s['gpu_uuid'] for s in sessions]
    assert len({s['frozen_source_identity'] for s in sessions})==1
    fields=dict(v.split('=',1) for v in slurm.split() if '=' in v)
    cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
        *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
    assert cutoff-now>=REQUIRED,'complete bounded connector capture/export/review/closeout budget does not fit'
    return {'job_id':job,'slurm':slurm,'available_s':cutoff-now,'required_s':REQUIRED,
        'cutoff_unix_s':cutoff,'role_watchdog_s':660,'pair_supervisor_bound_s':PAIR_BOUND,
        'scope':'Instrumented twelve-request functional pilot, one metadata profiler range, two180sCPU exports, review and publication/archive margin; no runtime or clock setting change'}


def profiled_command(root,pair_path,role,uuid):
    command,env=supervised_command(root,LABEL,ENTRY,['--pair-config',pair_path,
        '--role',role,'--gpu-uuid',uuid,'--output',root/'runs/model'],PAIR_BOUND)
    split=command.index('--')+1
    session=load(root/'session.json');nsys=Path(session['cuda_home'])/'bin/nsys'
    profile=[str(nsys),'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx',
        '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
        '--output='+str(root/'profiles/connector')]
    return command[:split]+profile+command[split:],env


def run_profile_pair(root,pair,cutoff):
    assert cutoff-time.time()>=PAIR_BOUND+60,'complete paired capture bound does not fit'
    pair=dict(pair,deadline_unix_s=min(cutoff,time.time()+PAIR_BOUND))
    path=root/'pair.json';write(path,pair)
    prepared=[(root/role,*profiled_command(root/role,path,role,pair['roles'][role]['gpu_uuid']))
        for role in ('producer','consumer')]
    processes=[]
    with contextlib.ExitStack() as stack:
        try:
            for base,command,env in prepared:
                log=stack.enter_context((root/'logs'/(base.name+'_supervisor.log')).open('x'))
                process=subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append((process,base))
            deadline=min(cutoff,pair['deadline_unix_s']+40)
            while any(p.poll() is None for p,_ in processes):
                assert all(p.poll() in (None,0) for p,_ in processes),'profiled role failed; stop waiting peer'
                if time.time()>=deadline:raise subprocess.TimeoutExpired('connector_profile_pair',PAIR_BOUND)
                time.sleep(.2)
            for process,base in processes:
                assert process.returncode==0
                record=load(base/'results/processes'/(LABEL+'.json'))
                assert record['status']=='complete' and not record['monitor']['failure']
        finally:
            for process,base in processes:stop_owned_role(process,base)


def export_trace(base,cutoff):
    assert cutoff-time.time()>=210,'complete CPU export bound does not fit'
    session=load(base/'session.json');scripts=Path(session['frozen_source_root'])/'scripts'
    report=base/'profiles/connector.nsys-rep';database=base/'profiles/connector.sqlite'
    assert report.is_file() and report.stat().st_size>0 and not database.exists()
    command=[session['python'],scripts/'h200_session.py','--root',base,'--label','connector_nsys_export',
        '--timeout','180','--',Path(session['cuda_home'])/'bin/nsys','export','--type=sqlite',
        '--output='+str(database),report]
    with (base/'logs/export_supervisor.log').open('x') as log:
        subprocess.run(list(map(str,command)),cwd=PACKAGE,check=True,timeout=210,stdout=log,stderr=subprocess.STDOUT)
    assert database.is_file() and database.stat().st_size>0
    return {'report':str(report),'report_sha256':sha(report),'database':str(database),
        'database_sha256':sha(database),'scope':'Export success and nonempty files only; CUDA/NVTX coverage still requires independent reader'}


def main(args):
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    core=args.functional_root.resolve();root=args.root.resolve()
    assert core.is_relative_to(PACKAGE/'h200') and root.is_relative_to(PACKAGE/'h200') and not root.exists()
    sessions=[load(core/f'gpu{i}/session.json') for i in (0,1)]
    slurm=subprocess.check_output(['scontrol','show','job','-o',os.environ.get('SLURM_JOB_ID','')],text=True,timeout=25)
    outcome=load(core/'results/worker_outcome.json');cert=core/'model/results/native_model_CPU_review.json'
    assert sha(cert)==outcome['model_review_sha256']
    pair=load(core/'model/pair.json')
    gate=admission(sessions,[load(core/f'gpu{i}/results/current_process.json') for i in (0,1)],
        os.environ.get('SLURM_JOB_ID'),socket.gethostname(),os.getuid(),slurm,outcome,load(cert),pair,time.time())
    needed=(ENTRY,'h200_connector_capture_gate.py','h200_connector_spans.py',
        'h200_connector_timeline.py','h200_review_connector_capture.py',Path(__file__).name)
    for session in sessions:
        assert sys.executable==session['python']
        for name in needed:assert sha(Path(session['frozen_source_root'])/'scripts'/name)==sha(PACKAGE/'scripts'/name)
    with contextlib.ExitStack() as stack:
        for session in sessions:
            lock=stack.enter_context((Path(session['workspace'])/'producer.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import pynvml as nv
        nv.nvmlInit()
        try:
            for session in sessions:assert not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByUUID(session['gpu_uuid']))
        finally:nv.nvmlShutdown()
        root.mkdir()
        args._created_by_controller=True
        for name in ('environment','results','logs','analysis','shared'):(root/name).mkdir()
        write(root/'environment/admission.json',gate)
        (root/'environment/controller_source.py').write_bytes(Path(__file__).read_bytes())
        (root/'analysis/report.md').write_text('# Instrumented native connector capture\n\nOriginal twelve-request functional protocol, one metadata-only profiler range. No clean producer ratio, deployed TTFT, serving goodput, physical traffic or primary energy claim.\n')
        ports=choose_ports()
        pair.update(connector_attribution_protocol=PROTOCOL,shared_directory=str(root/'shared'),
            request_prefix='h200_connector_'+os.environ['SLURM_JOB_ID']+'_'+str(time.time_ns()))
        result={'status':'preparing_profiled_connector_pair','scientific_qualified':False,'serving_qualified':False,'primary_J_per_token':None}
        try:
            write(root/'environment/nsys_version.json',{'output':subprocess.check_output(
                [Path(sessions[0]['cuda_home'])/'bin/nsys','--version'],text=True,timeout=25)})
            for role,old,port in zip(('producer','consumer'),sessions,ports):
                base=root/role;copy_role(core/('gpu0' if role=='producer' else 'gpu1'),base)
                (base/'profiles').mkdir();new=load(base/'session.json')
                workspace=Path('/tmp')/('h200_connector_'+str(os.getuid())+'_'+os.environ['SLURM_JOB_ID']+'_'+role)
                workspace.mkdir(exist_ok=False)
                write(base/'environment/cache_copy.json',copy_runtime_cache(Path(old['workspace']),workspace))
                new['workspace']=str(workspace);write(base/'session.json',new)
                pair['roles'][role]['port']=port
            publish(root,'H200 connector capture: publish owned idle pair and bounded instrumented scope')
            run_profile_pair(root,pair,gate['cutoff_unix_s'])
            traces={role:export_trace(root/role,gate['cutoff_unix_s']) for role in ('producer','consumer')}
            write(root/'results/traces.json',traces)
            from h200_review_native_model import review as functional_review
            checked=functional_review(root)
            assert checked['status']=='native_model_functional_review_passed' and not checked['errors']
            from h200_review_connector_capture import review
            reviewed=review(root)
            review_path=root/'results/connector_capture_CPU_review.json'
            assert not review_path.exists();write(review_path,reviewed)
            result.update(status=reviewed['status'],review=str(root/'results/connector_capture_CPU_review.json'),
                review_sha256=sha(root/'results/connector_capture_CPU_review.json'))
            if not reviewed['structure_passed']:
                limitation(root,'connector_capture_coverage','Independent source/functional/NSYS structure or launch association incomplete',
                    'results/connector_capture_CPU_review.json','Retained all raw kernels, unassigned requests and ambiguous/outside-window launches; no exit0-only qualification',
                    'Inspect exact rejected/partial rows, bound any repair and retain unmatched activity. Structural coverage alone never establishes layer compute, clean producer penalty, deployed goodput or energy.')
        except BaseException as error:
            result.update(status='blocked_or_partial_connector_capture',error=str(error),traceback=traceback.format_exc())
            write(root/'results/failure.json',result)
            limitation(root,'connector_capture',str(error),'results/failure.json',
                'Owned idle current pair, exact frozen source and complete1800second admission; one metadata range and unchanged twelve-request strict functional proof',
                'Retain all partial reports/databases and raw events. Inspect first failed source/correctness/coverage gate before a bounded repair or next eligible case; no exit0-only profiler claim.')
            raise
        finally:
            result.update(exact_next_command=sessions[0]['python']+' '+str(PACKAGE/'scripts/h200_review_connector_capture.py')+' --root '+str(root),
                next_required='Read-only verification of actual source/functional/capture structure; inspect unmatched activity and calibrate instrumentation before quantitative compute/producer claims. Preserve all partial outcomes and original archives.')
            write(root/'results/worker_outcome.json',result)
            with (root/'analysis/report.md').open('a') as f:f.write('\nActual disposition: '+result['status']+'. Independent launch coverage and calibration remain separate from clean measurements.\n')
            publish(root,'H200 connector capture: preserve actual completed and unresolved stages')
            try:
                subprocess.run([sessions[0]['python'],PACKAGE/'scripts/archive_h200_priority.py','--root',root,
                    '--destination','/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives'],check=True,timeout=180)
            except Exception as error:
                write(root/'results/archive_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
                limitation(root,'connector_capture_archive',str(error),'results/archive_failure.json',
                    'Bounded verified closed-unit archive; original reports/databases retained',
                    'Retry verified transfer from preserved originals; no archive success claimed.')
                publish(root,'H200 connector capture: preserve archive transfer failure')
                raise
            publish(root,'H200 connector capture: publish verified closed-unit archive receipt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--functional-root',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    try:main(args)
    except BaseException as error:
        root=args.root.resolve()
        if root.is_relative_to(PACKAGE/'h200') and (not root.exists() or getattr(args,'_created_by_controller',False)) and not (root/'results/worker_outcome.json').exists():
            (root/'results').mkdir(parents=True,exist_ok=True);(root/'analysis').mkdir(exist_ok=True)
            (root/'analysis/report.md').write_text('# Connector capture admission/setup failure\n\nNo GPU role launched; original current/queued comparison preserved.\n')
            write(root/'results/admission_failure.json',{'error':str(error),'traceback':traceback.format_exc(),'GPU_roles_launched':0})
            limitation(root,'connector_capture_admission',str(error),'results/admission_failure.json',
                'Require current strict functional/source proof, owned idle pair and full1800second bound before GPU capture',
                'Inspect failed gate and preserve active/queued work; never use unpinned code or shorten correctness trials.')
            publish(root,'H200 connector capture: preserve prelaunch admission/setup failure')
        raise
