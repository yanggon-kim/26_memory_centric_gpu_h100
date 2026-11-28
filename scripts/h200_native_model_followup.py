#!/usr/bin/env python3
"""Same-allocation native model prerequisite after reviewed terminal cases.

Reuse only a source-verified same-pair component. New stock/strict references,
separate metadata controls and intrusive payload proofs remain mandatory.
No deployment latency, producer penalty or serving qualification is inferred.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time
import traceback
from h200_priority_campaign import PACKAGE,load,write,sha,publish,limitation,config,valid_tokens,strict_reference_record
from h200_native_transport_pair import validate_sources,validate_returned,verify_terminal_reference_block,choose_ports
from h200_native_functional_campaign import run_one,run_pair,copy_role


def copy_runtime_cache(source, target):
    """Copy only compilation/registry caches, with byte checks, before a GPU run."""
    names=('vllm','torchinductor','triton','cuda')
    assert all(not (source/'cache'/name).is_symlink() for name in names)
    entries=[p for name in names for p in (source/'cache'/name).rglob('*')]
    assert all(not p.is_symlink() and p.stat().st_uid==os.getuid() for p in entries)
    files=[p for p in entries if p.is_file()]
    size=sum(p.stat().st_size for p in files)
    assert size<=4*1024**3 and shutil.disk_usage(target).free>=size+512*1024**2
    rows=[]
    for name in names:
        old=source/'cache'/name
        if old.exists():shutil.copytree(old,target/'cache'/name)
    for path in files:
        relative=path.relative_to(source/'cache');copied=target/'cache'/relative
        digest=sha(path);assert sha(copied)==digest
        rows.append({'path':str(relative),'bytes':path.stat().st_size,'sha256':digest})
    return {'source_workspace':str(source),'destination_workspace':str(target),'bytes':size,'files':rows,
        'scope':'Only owned closed compilation/registry caches; no HF credentials, weights, outputs or correctness qualification copied. Fresh independent reference/output gates remain mandatory.'}


def main(args):
    def interrupted(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    args.root=args.root.resolve();args.component=args.component.resolve()
    assert args.root.is_relative_to(PACKAGE/'h200') and args.component.is_relative_to(PACKAGE/'h200')
    paths=[p.resolve() for p in args.source_session];sessions=[load(p) for p in paths]
    job=os.environ.get('SLURM_JOB_ID')
    raw=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True,timeout=25)
    validate_sources(sessions,[load(p.parent/'results/current_process.json') for p in paths],job,socket.gethostname(),os.getuid(),raw)
    outcomes=[load(p.parent/'results/worker_outcome.json') for p in paths]
    terminal=[verify_terminal_reference_block(p.parent,s) if o.get('status')=='blocked_resident_pilot' else None
              for p,s,o in zip(paths,sessions,outcomes)]
    validate_returned(outcomes,terminal)
    fields=dict(item.split('=',1) for item in raw.split() if '=' in item)
    cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
               *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
    cached=bool(getattr(args,'cache_source_session',None))
    stock_limit=600 if cached else 300
    required=stock_limit+240+720+180
    assert cutoff-time.time()>=required, 'complete stock/reference/model plus180second staging/publication budget does not fit'
    cache_session=load(args.cache_source_session) if cached else None
    if cached:
        assert cache_session['slurm_job_id']==job and cache_session['expected_hostname']==socket.gethostname()
        assert Path(args.cache_source_session).resolve() in paths
        cache_outcome=load(Path(args.cache_source_session).resolve().parent/'results/worker_outcome.json')
        assert cache_outcome['status']=='prefill_acquisition_disposition' and cache_outcome['complete_acquisition_set'] is True
    from h200_review_native_transport import review as component_review
    reviewed=component_review(args.component)
    assert reviewed['status']=='known_data_transport_functional_review_passed'
    assert reviewed['job_id']==job and set(reviewed['gpu_uuids'])=={s['gpu_uuid'] for s in sessions}
    assert [r['gpu_uuid'] for r in reviewed['roles']]==[s['gpu_uuid'] for s in sessions], 'component direction must match model roles'
    review_path=args.component/'results/native_component_CPU_review.json'
    with contextlib.ExitStack() as stack:
        for session in sessions:
            lock=stack.enter_context((Path(session['workspace'])/'producer.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import pynvml as nv
        nv.nvmlInit()
        try:
            for session in sessions:
                assert not nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByUUID(session['gpu_uuid']))
        finally: nv.nvmlShutdown()
        args.root.mkdir(exist_ok=False)
        for name in ('environment','results','logs','configs','analysis'):(args.root/name).mkdir()
        (args.root/'analysis/report.md').write_text('# Same-allocation real-model native KV prerequisite\n\nOriginal cases returned and were reviewed. Reuse only the verified same-pair component; new full64stock and strict dynamic/unchanged StaticCache references precede metadata-only controls and separate intrusive source/receive/injection proofs. Serialized functional handoff only; no producer timing, serving, goodput, physical-byte or energy qualification.\n')
        write(args.root/'environment/idle_admission.json',{'job_id':job,'slurm':raw,
            'original_sessions':[{'path':str(p),'sha256':sha(p)} for p in paths],
            'terminal_reference_block_reviews':terminal,'native_component_review':str(review_path),
            'native_component_review_sha256':sha(review_path),'required_s':required,'stock_reference_limit_s':stock_limit,'available_s':cutoff-time.time(),
            'cutoff_unix_s':cutoff,'first_preemption_eligible':fields.get('PreemptEligibleTime'),
            'scope':'Actual scheduler end minus15minute reserve; first-hour preemption may interrupt sooner. Both original locks held, original workers returned, selected UUIDs idle. No allocation or unrelated process stopped.'})
        write(args.root/'environment/orchestrator_source.json',{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))})
        (args.root/'environment/orchestrator_source.py').write_bytes(Path(__file__).read_bytes())
        subprocess.run([sessions[0]['python'],PACKAGE/'scripts/checkpoint_h200.py',
            '--root',args.root,'--root',args.component,
            '--message','H200 native model follow-up: publish reviewed component and terminal-case admission','--push'],
            check=True,timeout=300)
        bases=[]
        for index,old in enumerate(sessions):
            base=args.root/f'gpu{index}'
            for name in ('environment','results','logs','analysis','configs','runs'):(base/name).mkdir(parents=True,exist_ok=True)
            session={k:v for k,v in old.items() if k not in ('frozen_source_root','frozen_source_identity')}
            session.update(workspace=old['workspace']+'_model_'+args.root.name,
                protocol='native_model_functional_cached_startup_v2' if cached else 'native_model_functional_followup_v1',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip())
            Path(session['workspace']).mkdir(exist_ok=False)
            write(base/'session.json',session)
            subprocess.run([session['python'],PACKAGE/'scripts/freeze_shoreline.py','--session',base/'session.json','--activate'],check=True,timeout=120)
            if cached:
                copied=copy_runtime_cache(Path(cache_session['workspace']),Path(session['workspace']))
                copied.update(source_session=str(args.cache_source_session),source_session_sha256=sha(args.cache_source_session))
                write(base/'environment/runtime_cache_copy.json',copied)
            bases.append(base)
        assert len({load(b/'session.json')['frozen_source_identity'] for b in bases})==1
        outcome={'status':'preparing','quantitative_performance_qualified':False,'serving_qualified':False}
        active=args.root
        try:
            reference_root=bases[0];active=reference_root
            cfg=config(reference_root,'graph',128,3,1)
            source=reference_root/'runs/P0_native_model_reference/run.json'
            run_one(reference_root,'P0_native_model_reference','controlled_vllm_experiments.py',[
                '--config',cfg,'--gpu-uuid',sessions[0]['gpu_uuid'],'--output',source.parent,'--mode','timing','--generated-tokens','64'],stock_limit,cutoff)
            assert valid_tokens(load(source))
            hf=reference_root/'results/P0_native_model_reference_hf.json'
            run_one(reference_root,'P0_native_model_reference_hf','hf_full_reference.py',[
                '--source-run',source,'--gpu-uuid',sessions[0]['gpu_uuid'],'--output',hf],240,cutoff)
            assert strict_reference_record(load(hf),sha(source))
            publish(args.root,'H200 native model follow-up: publish fresh stock and strict dualHF reference')
            model=args.root/'model';active=model
            for name in ('results','analysis','shared','logs'):(model/name).mkdir(parents=True,exist_ok=True)
            (model/'analysis/report.md').write_text('# Native model functional comparison\n\nMetadata-only controls precede intrusive payload checks. Serialized handoff is not deployed serving timing.\n')
            for role,base in zip(('producer','consumer'),bases):copy_role(base,model/role)
            ports=choose_ports()
            pair={'protocol':'native_model_KV_functional_v1','job_id':job,'hostname':socket.gethostname(),'host':'127.0.0.1',
                'synthetic_payload':False,'context':128,'generated_tokens':64,
                'roles':{role:{'gpu_uuid':s['gpu_uuid'],'port':port} for role,s,port in zip(('producer','consumer'),sessions,ports)},
                'request_prefix':'h200_model_'+job+'_'+str(time.time_ns()),'shared_directory':str(model/'shared'),
                'reference_run':str(source),'reference_run_sha256':sha(source),'strict_HF_reference':str(hf),'strict_HF_reference_sha256':sha(hf),
                'native_component_CPU_review':str(review_path),'native_component_CPU_review_sha256':sha(review_path)}
            run_pair(model,pair,'h200_native_model_pilot.py','native_model_KV_functional','model',720,cutoff)
            from h200_review_native_model import review as model_review
            checked=model_review(model);assert checked['status']=='native_model_functional_review_passed'
            outcome.update(status='native_model_functional_prerequisites_passed',review=str(model/'results/native_model_CPU_review.json'),
                review_sha256=sha(model/'results/native_model_CPU_review.json'))
        except BaseException as error:
            outcome.update(status='blocked_or_partial_native_model',error=str(error),traceback=traceback.format_exc())
            write(args.root/'results/failure.json',outcome)
            limitation(args.root,'native_model_followup',str(error),'results/failure.json',
                'Same-pair verified component, new full64stock/strict dualHF and bounded supervised native model phases',
                'Preserve every partial stage and inspect first failure before one bounded repair; no serving or timing promotion')
            if active!=args.root:
                write(active/'results/failure.json',outcome)
                limitation(active,'native_model_stage',str(error),'results/failure.json',
                    'Same fixed outputs and full phase/payload gates; no weakened assertions','Inspect this failed stage before a new identified attempt')
            raise
        finally:
            outcome.update(next_required='Matched clean producer/connector timing and live consumer comparisons remain pending; functional success is not performance qualification.')
            write(args.root/'results/worker_outcome.json',outcome)
            with (args.root/'analysis/report.md').open('a') as stream:stream.write('\nDisposition: '+outcome['status']+'. '+outcome['next_required']+'\n')
            publish(args.root,'H200 native model follow-up: publish complete and unresolved functional stages')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',type=Path,action='append',required=True)
    p.add_argument('--component',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache-source-session',type=Path,help='Explicit bounded startup repair: owned completed prefill cache,600second stock bound; no output/source qualification transfer')
    main(p.parse_args())
