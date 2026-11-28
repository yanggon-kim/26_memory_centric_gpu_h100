#!/usr/bin/env python3
"""Standalone owned two-H200 functional prerequisites for native serving.

No submission, clock changes or numerical deployment qualification. Run only
through the dedicated Slurm entrypoint after the current comparison closes.
"""
import argparse
import contextlib
import datetime as dt
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import traceback
from h200_priority_campaign import PACKAGE, load, write, sha, config, publish, limitation, valid_tokens, strict_reference_record
from h200_native_transport_pair import choose_ports, stop_owned_role

PYTHON = '/storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python'
CUDA = '/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-13.0.1-ne4hey4ce57lmd34nixeikyfribdqln3'


def remaining(cutoff):
    return cutoff-time.time()


def role_environment(root):
    session = load(root/'session.json'); scripts = Path(session['frozen_source_root'])/'scripts'
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','CUDA_INJECTION64_PATH','NVBIT_INSTRUMENTATION_ENABLED'):
        env.pop(key,None)
    env.update(H200_SESSION_CONFIG=str(root/'session.json'),PYTHONNOUSERSITE='1',
        CUDA_VISIBLE_DEVICES=session['gpu_uuid'],OMP_NUM_THREADS='4',MAX_JOBS='4',TORCHINDUCTOR_COMPILE_THREADS='4')
    for variable,name in (('CC','h200_host_cc.sh'),('CXX','h200_host_cxx.sh')):
        (scripts/name).chmod(0o700); env[variable] = str(scripts/name)
    return session,scripts,env


def supervised_command(root,label,entry,arguments,timeout):
    session,scripts,env = role_environment(root)
    command = [session['python'],scripts/'h200_session.py','--root',root,'--label',label,
        '--gpu','--timeout',str(timeout),'--',session['python'],scripts/entry,*arguments]
    return list(map(str,command)),env


def run_one(root,label,entry,arguments,timeout,cutoff):
    assert remaining(cutoff) >= timeout+30, 'complete '+label+' budget does not fit'
    command,env = supervised_command(root,label,entry,arguments,timeout)
    with (root/'logs'/(label+'_supervisor.log')).open('x') as log:
        process = subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            process.wait(timeout=min(timeout+45,max(1,remaining(cutoff))))
            assert process.returncode == 0, label+' supervisor failed'
            record = load(root/'results/processes'/(label+'.json'))
            assert record['status'] == 'complete' and not record['monitor']['failure']
            return record
        finally:
            stop_owned_role(process,root)


def copy_role(base,root):
    for name in ('environment','results','logs','analysis','runs'):
        (root/name).mkdir(parents=True,exist_ok=True)
    write(root/'session.json',load(base/'session.json'))
    # Source snapshots are written separately by each actual H200Session run.


def run_pair(root,pair,entry,label,output_name,timeout,cutoff):
    assert remaining(cutoff) >= timeout+60, 'complete paired '+label+' budget does not fit'
    pair['deadline_unix_s'] = min(cutoff,time.time()+timeout)
    pair_path = root/'pair.json'; write(pair_path,pair)
    processes = []
    with contextlib.ExitStack() as stack:
        try:
            for role in ('producer','consumer'):
                subroot = root/role
                command,env = supervised_command(subroot,label,entry,[
                    '--pair-config',pair_path,'--role',role,'--gpu-uuid',pair['roles'][role]['gpu_uuid'],
                    '--output',subroot/'runs'/output_name],timeout)
                log = stack.enter_context((root/'logs'/(role+'_supervisor.log')).open('x'))
                process = subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append((process,subroot))
            deadline = min(cutoff,pair['deadline_unix_s']+40)
            while any(process.poll() is None for process,subroot in processes):
                assert all(process.poll() in (None,0) for process,subroot in processes), 'native role failed; stop its waiting peer immediately'
                if time.time()>=deadline:
                    raise subprocess.TimeoutExpired(label,timeout)
                time.sleep(.2)
            records = []
            for process,subroot in processes:
                assert process.returncode == 0, 'paired native role failed: '+subroot.name
                process_path = subroot/'results/processes'/(label+'.json')
                native_path = subroot/'runs'/output_name/'run.json'
                records.append({'process':str(process_path),'process_sha256':sha(process_path),
                    'run':str(native_path),'run_sha256':sha(native_path)})
            return records
        finally:
            for process,subroot in processes:
                stop_owned_role(process,subroot)


def stage(root,name,command,limit,cutoff):
    assert remaining(cutoff)>30
    began = time.time()
    with (root/'logs'/(name+'.log')).open('x') as stream:
        result = subprocess.run(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,
            timeout=min(limit,max(1,remaining(cutoff)-15)))
    write(root/'results'/(name+'.json'),{'exit_code':result.returncode,'elapsed_s':time.time()-began,
        'command':list(map(str,command)),'classification':'environment preparation; no model result'})
    assert result.returncode == 0, name+' failed'


def start_spare_counter(root,base,cutoff):
    assert remaining(cutoff)>=360, 'complete optional counter bound does not fit'
    session,scripts,env=role_environment(base)
    command=[session['python'],PACKAGE/'scripts/h200_native_counter_campaign.py','--root',base,'--session',base/'session.json']
    log=(root/'logs/spare_counter_coordinator.log').open('x')
    try:
        process=subprocess.Popen(list(map(str,command)),cwd=PACKAGE,env=env,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:
        log.close();raise
    return {'process':process,'log':log,'started':time.time(),'base':base,'command':list(map(str,command))}


def finish_spare_counter(root,task,cutoff):
    process=task['process'];base=task['base']
    result={'command':task['command'],'scientific_gate':'see independent native counter disposition; never inferred from coordinator exit0'}
    try:
        process.wait(timeout=max(1,min(360-(time.time()-task['started']),remaining(cutoff)-30)))
        path=base/'results/native_counter_disposition.json'
        result.update(exit_code=process.returncode,status='finished' if process.returncode==0 and path.exists() else 'failed_or_missing_disposition',
            disposition=str(path),disposition_sha256=sha(path) if path.exists() else None)
    except subprocess.TimeoutExpired:
        result.update(status='bounded_counter_coordinator_timeout')
    finally:
        stop_owned_role(process,base);task['log'].close()
    write(root/'results/optional_native_counter.json',result)
    if result['status']!='finished':
        limitation(root,'optional_native_counter_coordinator',result['status'],'results/optional_native_counter.json',
            'Independent single-UUID coordinator with360second total bound during the other GPUs stock/HFreference; owned-child cleanup only',
            'Inspect preserved native/profiler evidence; independent model functional gates remain mandatory and eligible')
    publish(root,'H200 independent known-payload counter: publish actual disposition before paired model phases')
    return result


def main(args):
    def interrupted(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    assert sys.executable == PYTHON and os.environ.get('SLURM_JOB_ID','').isdigit()
    job = os.environ['SLURM_JOB_ID']
    state = subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
    fields = dict(item.split('=',1) for item in state.split() if '=' in item)
    assert fields['JobState'] == 'RUNNING' and fields['UserId'].endswith('('+str(os.getuid())+')')
    assert fields['NodeList'] == socket.gethostname().split('.')[0] and 'gres/gpu:h200=2' in fields['AllocTRES'].split(',')
    end = dt.datetime.fromisoformat(fields['EndTime']).astimezone()
    start = dt.datetime.fromisoformat(fields['StartTime']).astimezone()
    cutoff = min((end-dt.timedelta(seconds=900)).timestamp(),start.timestamp()+3600-120)
    context=getattr(args,'context',128)
    assert context in (128,16384,32704) and context+64<=32768
    suffix='' if context==128 else '_c'+str(context)
    root = PACKAGE/'h200'/f'{dt.datetime.now():%Y%m%d}_job{job}_native_functional{suffix}'
    for name in ('environment','results','logs','analysis','configs'):
        (root/name).mkdir(parents=True,exist_ok=False)
    (root/'analysis/report.md').write_text('# Native model KV functional prerequisites\n\nNo deployment timing or serving qualification. Preserve known-data component, metadata-only model controls and intrusive payload proof separately. At initial setup, no GPU measurements had been made.\n')
    outcome = {'status':'preparing','job_id':job,'quantitative_qualified':False,'serving_qualified':False}
    active_root = root
    counter_task=None
    write(root/'STATUS.json',outcome)
    try:
        progress_path = PACKAGE/'h200/20261127_resident_preparation/results/campaign_progress.json'
        progress = load(progress_path)
        assert args.preceding_job in progress['closed_job_ids'], 'preceding comparison must be closed and recorded before native follow-up'
        write(root/'environment/case_boundary.json',{'preceding_job':args.preceding_job,
            'campaign_manifest':str(progress_path),'campaign_manifest_sha256':sha(progress_path),
            'scope':'Prior allocation closeout is necessary; the submitting agent must also review the current scientific comparison disposition before selecting this prerequisite job'})
        write(root/'environment/slurm.json',{'state':state,'cutoff_unix_s':cutoff,
            'scope':'Actual EndTime minus15minute closeout, additionally aim to finish2minutes before first-hour preemption eligibility; no guaranteed lease'})
        visibility = load(os.environ['H200_VISIBILITY_MANIFEST'])
        assert visibility['slurm_job_id'] == job and visibility['hostname'] == socket.gethostname()
        assert visibility['selected_cuda_visible_devices'] == os.environ['CUDA_VISIBLE_DEVICES']
        write(root/'environment/allocated_visibility.json',visibility)
        uuids = visibility['gpu_uuids']; assert len(set(uuids)) == 2
        stage(root,'runtime_check',[PYTHON,'-u','-c',
            "import torch,vllm,transformers; assert torch.__version__.split('+')[0]=='2.11.0'; assert vllm.__version__=='0.22.1'; assert transformers.__version__=='4.57.6'; assert torch.version.cuda=='13.0'; assert torch.cuda.device_count()==2; print(torch.__version__,vllm.__version__,transformers.__version__,torch.version.cuda); print([(str(torch.cuda.get_device_properties(i).uuid),torch.cuda.get_device_properties(i).name,torch.cuda.get_device_properties(i).multi_processor_count) for i in range(2)])"],600,cutoff)
        stage(root,'runtime_manifest',[PYTHON,'-m','pip','freeze'],90,cutoff)
        budgets = {}
        for location in (PACKAGE,Path('/storage/scratch1/6/jkim4112/models/h200_priority_hub'),
                         Path('/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives')):
            usage = shutil.disk_usage(location)
            budgets[str(location)] = {'total_bytes':usage.total,'free_bytes':usage.free}
        write(root/'environment/storage.json',budgets)
        from h200_resident_campaign import verify_published_core
        gate = verify_published_core(PACKAGE/'h200/20261127_job16228671_priority/gpu0/results/resident_curve_CPU_review.json')
        assert remaining(cutoff)>=2400, 'complete functional pipeline budget no longer fits after staging'
        write(root/'results/admission.json',{'available_s':remaining(cutoff),'required_s':2400,
            'basis':'Bounded component540s, stock600s, reference240s, model720s plus preparation/publication/cleanup margin. The earlier300second cold stock bound timed out during initialization on job16238357; no numerical gate is changed.'})
        write(root/'environment/topology.json',{'command':['nvidia-smi','topo','-m'],
            'output':subprocess.check_output(['nvidia-smi','topo','-m'],text=True,timeout=25)})
        params = Path('/proc/driver/nvidia/params')
        write(root/'environment/profiler_permissions.json',{'driver_params':params.read_text() if params.exists() else None,
            'counter_permission_exercised':False,'scope':'Functional protocol uses no profiler counters; driver setting observed only, no permission or performance-counter claim'})
        bases = []
        for index,uuid in enumerate(uuids):
            base = root/f'gpu{index}'
            for name in ('environment','results','logs','analysis','configs','runs'):
                (base/name).mkdir(parents=True,exist_ok=True)
            session = {'schema_version':1,'slurm_job_id':job,'expected_hostname':socket.gethostname(),
                'python':PYTHON,'cuda_home':CUDA,'gpu_uuid':uuid,'allocation_end':end.isoformat(),
                'launch_cutoff':dt.datetime.fromtimestamp(cutoff).astimezone().isoformat(),
                'closeout_buffer_s':900,'idle_required_s':10,'monitor_period_s':.05,
                'workspace':f'/tmp/h200_native_{os.getuid()}_{job}_{index}',
                'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip(),
                'protocol':'native_functional_prerequisites_v1','published_core_gate':gate,'seed':20261013}
            Path(session['workspace']).mkdir(exist_ok=False)
            write(base/'session.json',session)
            subprocess.run([PYTHON,PACKAGE/'scripts/freeze_shoreline.py','--session',base/'session.json','--activate'],check=True,timeout=120)
            bases.append(base)
        publish(root,'H200 native prerequisites: publish allocated pair and frozen source before producers')
        component = root/'transport'
        for name in ('results','analysis','shared','logs'):
            (component/name).mkdir(parents=True,exist_ok=True)
        (component/'analysis/report.md').write_text('# Native known-data transport component\n\nSynthetic BF16 blocks only; not model KV or serving.\n')
        active_root = component
        for role,base in zip(('producer','consumer'),bases): copy_role(base,component/role)
        ports = choose_ports()
        pair = {'job_id':job,'hostname':socket.gethostname(),'host':'127.0.0.1',
            'roles':{role:{'gpu_uuid':uuid,'port':port} for role,uuid,port in zip(('producer','consumer'),uuids,ports)},
            'request_prefix':'h200_'+job+'_'+str(time.time_ns()),'shared_directory':str(component/'shared'),'synthetic_payload':True}
        write(root/'STATUS.json',dict(outcome,status='running_component',stage='known-data native transport'))
        records = run_pair(component,pair,'h200_native_transport_probe.py','native_known_data_transport','component',540,cutoff)
        write(component/'results/pair_disposition.json',{'status':'component_captured_pending_CPU_review',
            'roles':records,'model_KV_verified':False,'serving_qualified':False,'quantitative_performance_qualified':False})
        from h200_review_native_transport import review as review_component
        reviewed = review_component(component); assert reviewed['status']=='known_data_transport_functional_review_passed'
        publish(root,'H200 native component: publish independently reviewed known-data pair')
        outcome.update(status='component_passed_model_pending')
        if getattr(args,'spare_native_counter',False):
            try:
                counter_task=start_spare_counter(root,bases[1],cutoff)
            except (OSError,AssertionError) as error:
                write(root/'results/optional_native_counter_start_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
                limitation(root,'optional_native_counter_start',str(error),'results/optional_native_counter_start_failure.json',
                    'Bounded optional spare-device control; required model functional stages are independent',
                    'Inspect the preserved startup error; continue required model gates without claiming counters')
        reference_root = bases[0]; cfg = config(reference_root,'graph',context,3,1)
        active_root = reference_root
        write(root/'STATUS.json',dict(outcome,status='running_stock_reference',stage=f'new C{context} full64stock and strict dualHF'))
        source = reference_root/'runs/P0_native_model_reference/run.json'
        run_one(reference_root,'P0_native_model_reference','controlled_vllm_experiments.py',[
            '--config',cfg,'--gpu-uuid',uuids[0],'--output',source.parent,'--mode','timing','--generated-tokens','64'],600,cutoff)
        assert valid_tokens(load(source))
        hf = reference_root/'results/P0_native_model_reference_hf.json'
        run_one(reference_root,'P0_native_model_reference_hf','hf_full_reference.py',[
            '--source-run',source,'--gpu-uuid',uuids[0],'--output',hf],240,cutoff)
        assert strict_reference_record(load(hf),sha(source))
        if counter_task is not None:
            finish_spare_counter(root,counter_task,cutoff);counter_task=None
        model = root/'model'
        for name in ('results','analysis','shared','logs'):
            (model/name).mkdir(parents=True,exist_ok=True)
        (model/'analysis/report.md').write_text('# Real-model native KV functional pilot\n\nMetadata-only controls followed by intrusive payload proofs. Serialized handoff, not deployment timing or serving.\n')
        active_root = model
        for role,base in zip(('producer','consumer'),bases): copy_role(base,model/role)
        ports = choose_ports()
        pair.update(protocol='native_model_KV_functional_v1' if context==128 else 'native_model_KV_long_context_functional_v2',synthetic_payload=False,context=context,generated_tokens=64,
            roles={role:{'gpu_uuid':uuid,'port':port} for role,uuid,port in zip(('producer','consumer'),uuids,ports)},
            request_prefix='h200_model_'+job+'_'+str(time.time_ns()),shared_directory=str(model/'shared'),
            reference_run=str(source),reference_run_sha256=sha(source),strict_HF_reference=str(hf),strict_HF_reference_sha256=sha(hf),
            native_component_CPU_review=str(component/'results/native_component_CPU_review.json'),
            native_component_CPU_review_sha256=sha(component/'results/native_component_CPU_review.json'))
        write(root/'STATUS.json',dict(outcome,status='running_native_model_functional',stage='metadata controls then intrusive payload proofs'))
        run_pair(model,pair,'h200_native_model_pilot.py','native_model_KV_functional','model',720,cutoff)
        from h200_review_native_model import review as review_model
        reviewed = review_model(model); assert reviewed['status']=='native_model_functional_review_passed'
        outcome.update(status='functional_prerequisites_passed',model_review=str(model/'results/native_model_CPU_review.json'),
            model_review_sha256=sha(model/'results/native_model_CPU_review.json'))
    except BaseException as error:
        outcome.update(status='blocked_or_partial_functional',error=str(error),traceback=traceback.format_exc())
        write(root/'results/failure.json',outcome)
        limitation(root,'native_functional_campaign',str(error),'results/failure.json',
            'Bounded known-data component, strict stock reference and separate model phase/payload controls; no gate relaxed',
            'Inspect the first failing stage, retain all partial data and prepare one bounded repair before a new paired cohort')
        if active_root != root:
            write(active_root/'results/failure.json',outcome)
            limitation(active_root,'native_functional_stage',str(error),'results/failure.json',
                'Bounded supervised functional stage; every partial role retained',
                'Inspect this comparison before a new bounded cohort; no timing or serving conclusion')
        raise
    finally:
        if counter_task is not None:
            finish_spare_counter(root,counter_task,cutoff);counter_task=None
        outcome.update(next_command=f'{PYTHON} -m json.tool {root}/results/worker_outcome.json',
            next_required_work='Inspect the first failing stage; after functional gates pass, implement matched clean producer/connector timing controls before loaded serving. This exact next command inspects evidence; no unimplemented experiment is presented as ready.',
            completed_scope='Only the component/model functional stages explicitly passed by their CPU readers count; no producer-penalty, full-versus-restricted live worker or goodput comparison is completed here')
        write(root/'results/worker_outcome.json',outcome); write(root/'STATUS.json',outcome)
        with (root/'analysis/report.md').open('a') as stream:
            stream.write('\n## Handoff\n\n'+outcome['status']+'. '+outcome['completed_scope']+'\n\nNext: '+outcome['next_command']+'.\n')
        publish(root,'H200 native prerequisites: publish completed and unresolved functional stages')
        try:
            subprocess.run([PYTHON,PACKAGE/'scripts/archive_h200_priority.py','--root',root,
                '--destination','/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives'],check=True,timeout=180)
        except Exception as error:
            write(root/'results/archive_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
            limitation(root,'native_archive_transfer',str(error),'results/archive_failure.json',
                'Bounded checksum-verified archive transfer; originals retained',
                'Resume archive verification from preserved originals; do not claim successful transfer')
            publish(root,'H200 native prerequisites: preserve archive transfer limitation')
            raise
        publish(root,'H200 native prerequisites: publish verified archive receipt and handoff')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-job',default='16238357')
    parser.add_argument('--context',type=int,choices=(128,16384,32704),default=128)
    parser.add_argument('--spare-native-counter',action='store_true',help='Independent known512MiB DRAM/L2 counter control on GPU1while GPU0runs its required stock/HFreference')
    main(parser.parse_args())
