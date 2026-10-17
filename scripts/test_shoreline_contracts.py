#!/usr/bin/env python3
"""Negative and boundary contracts for new statistical/replay qualification."""
import math
from shoreline_statistics import latency_bounds,sensitivity,hierarchical_interval,paired_ratio


def rejected(function,*args):
    try:function(*args)
    except (AssertionError,ValueError):return
    raise AssertionError('invalid inputs accepted')


def main():
    from trace_vllm_decode import ordinary_control_guard
    ordinary_control_guard({},3,10)
    rejected(ordinary_control_guard,{'CUDA_INJECTION64_PATH':'unit-test-tracer.so'},3,10)
    rejected(ordinary_control_guard,{'LD_PRELOAD':'unit-test-tracer.so'},3,10)
    rejected(ordinary_control_guard,{'NVBIT_INSTRUMENTATION_ENABLED':'1'},3,10)
    rejected(ordinary_control_guard,{},2,10)
    rejected(ordinary_control_guard,{},3,9)
    from shoreline_kernel_identity import attention_role
    assert attention_role('void flash::prepare_varlen_num_blocks_kernel<int>()')=='attention_scheduler'
    assert attention_role('void vllm::reshape_and_cache_flash_kernel<BF16>()')=='kv_update'
    assert attention_role('void cutlass::device_kernel<flash::FlashAttnFwdSm90<params>>')=='attention_forward'
    assert attention_role('void cutlass::device_kernel<flash::FlashAttnFwdCombine<params>>')=='attention_combine'
    assert attention_role('void flashinfer::sampling_kernel<params>()') is None
    from shoreline_kernel_analysis import static_opcodes
    assert static_opcodes('Header\n0x1234 @!P0 LDG.E R1, [R2]\n0x1244 HGMMA.64X8X16.F32.BF16 R0\n0x1254 EXIT\n')=={'LDG.E':1,'HGMMA.64X8X16.F32.BF16':1,'EXIT':1}
    assert static_opcodes('Warning: no loaded source\n')=={}
    import tempfile,json
    from pathlib import Path
    from shoreline_evidence import qualify_replays
    import os,threading
    from types import SimpleNamespace
    from h200_session import ProcessMonitor
    monitor=ProcessMonitor.__new__(ProcessMonitor);monitor.process=SimpleNamespace(pid=os.getpid())
    assert monitor.owned(os.getpid()) is True
    observed=[]
    def thread_identity():observed.append(monitor.owned(threading.get_native_id()))
    thread=threading.Thread(target=thread_identity);thread.start();thread.join()
    assert observed==[True], 'owned CUDA thread IDs must resolve through Tgid'
    assert monitor.owned(1) is False
    from unittest.mock import patch
    import run_trace,signal,subprocess
    child=SimpleNamespace(pid=987654321,poll=lambda:None)
    waited=[]
    def wait_child(timeout=None):
        waited.append(timeout)
        if timeout:raise subprocess.TimeoutExpired('unit-test-owned-child',timeout)
    child.wait=wait_child
    with patch.object(run_trace.os,'killpg') as kill:
        run_trace.stop_owned_child(child)
        assert kill.call_args_list==[__import__('unittest').mock.call(child.pid,signal.SIGTERM),__import__('unittest').mock.call(child.pid,signal.SIGKILL)]
        assert waited==[5,None]
    try:run_trace.terminate_wrapper(signal.SIGTERM,None)
    except SystemExit as error:assert error.code==143
    else:raise AssertionError('wrapper ignored termination')
    import shoreline_queue
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'environment/source_snapshots').mkdir(parents=True);(root/'logs').mkdir();(root/'results').mkdir()
        session=root/'session.json';session.write_text(json.dumps({'launch_cutoff':'2000-01-01T00:00:00+00:00'}))
        calls=[]
        with patch.object(shoreline_queue.subprocess,'call',side_effect=lambda command,**kw: calls.append(command) or 0):
            shoreline_queue.main(SimpleNamespace(session=session,start_at='J2'))
        assert [Path(cmd[1]).name for cmd in calls]==['shoreline_analyze.py','plot_shoreline.py','checkpoint_h200.py']
        state=json.loads((root/'results/campaign_queue.json').read_text())
        assert state['status']=='launch_cutoff_reached' and state['steps'][0]['status']=='pending'
        assert all(item['exit_code']==0 for item in state['cpu_closeout'])
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'results').mkdir();(root/'logs').mkdir();(root/'environment/source_snapshots').mkdir(parents=True)
        session=root/'session.json';session.write_text(json.dumps({'launch_cutoff':'2100-01-01T00:00:00+00:00'}))
        calls=[]
        with patch.object(shoreline_queue.subprocess,'call',side_effect=lambda command,**kw: calls.append(command) or 0):
            shoreline_queue.main(SimpleNamespace(session=session,start_at='J11_semantic_pilot'))
        tracing=[command for command in calls if '--job' in command and command[command.index('--job')+1]=='J11']
        assert len(tracing)==2 and '--semantic-pilot-only' in tracing[0] and '--semantic-pilot-only' not in tracing[1]
    from shoreline_treatment_analysis import provenance
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'results/processes').mkdir(parents=True);(root/'environment').mkdir()
        run=root/'run.json';run.write_text('{}')
        import hashlib
        identity='1'*64;manifest=root/'environment'/('source_freeze_'+identity[:12]+'.json')
        manifest.write_text(json.dumps({'identity':identity}))
        process=root/'results/processes/fixture.json'
        process.write_text(json.dumps({'status':'complete','monitor':{'failure':None},'source_freeze_identity':identity}))
        row={'id':'fixture','run_source':'run.json','run_sha256':hashlib.sha256(run.read_bytes()).hexdigest()}
        assert provenance(root,row)==identity
        assert provenance(root,{**row,'run_sha256':'0'*64}) is None
        process.write_text(json.dumps({'status':'rejected','source_freeze_identity':identity}))
        assert provenance(root,row) is None
    import shoreline_reuse
    with tempfile.TemporaryDirectory() as directory:
        package=Path(directory);old=package/'h200/old';current=package/'h200/new'
        old.mkdir(parents=True);(current/'results').mkdir(parents=True)
        artifact=old/'run.json';qualification=old/'qualification.json'
        artifact.write_text(json.dumps({'status':'complete','cases':[{'output_digests_identical':True,
            'trials':[{'output_ids':[[1,2,3]]}]*3}]}));qualification.write_text('{}')
        import hashlib
        entry={'key':'fixture','original_session':'h200/old','artifact':'run.json',
               'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
               'qualification_artifact':'qualification.json','qualification_sha256':hashlib.sha256(qualification.read_bytes()).hexdigest(),
               'role':'exact-output-reference-only','clock_timing_join_permitted':False}
        registry=current/'results/evidence_reuse.json'
        def registry_entry(value):registry.write_text(json.dumps({'entries':[value]}))
        with patch.object(shoreline_reuse,'__file__',str(package/'scripts/shoreline_reuse.py')):
            registry_entry(entry);assert shoreline_reuse.output_reference(current,'fixture')==artifact
            assert shoreline_reuse.output_reference(current,'missing') is None
            registry_entry({**entry,'sha256':'0'*64});rejected(shoreline_reuse.output_reference,current,'fixture')
            registry_entry({**entry,'clock_timing_join_permitted':True});rejected(shoreline_reuse.output_reference,current,'fixture')
            registry_entry({**entry,'artifact':'../../escape'});rejected(shoreline_reuse.output_reference,current,'fixture')
            registry.write_text(json.dumps({'entries':[entry,entry]}));rejected(shoreline_reuse.output_reference,current,'fixture')
    import h200_session,subprocess,socket
    from shoreline_jobs import functional_smoke_reference
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'configs').mkdir();(root/'results/qualifications').mkdir(parents=True)
        cfg={'model_id':'fixture','model_revision':'1'*40,'seed':20261013,'precision':'BF16'}
        (root/'configs/fixture_smoke.json').write_text(json.dumps(cfg))
        def smoke(mode,token=7,context=2048):
            out=root/'runs'/mode;out.mkdir(parents=True,exist_ok=True);path=out/'run.json'
            path.write_text(json.dumps({'status':'complete','config':cfg,'cases':[{'batch_size':1,'context_length':context,
                'generated_tokens':64,'prompt_sha256':'fixed','trials':[{'output_ids':[[token]*64]}]*3}]}))
            (root/'results/qualifications'/f'fixture_{mode}_b1_c2048_smoke.json').write_text(json.dumps({
                'status':'passed','command_status':'complete','output':str(out.relative_to(root)),
                'run_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
            return path
        smoke('hf_eager');path=smoke('eager')
        assert functional_smoke_reference(root,'fixture','eager',1,2048)==path
        path.write_text('{}');rejected(functional_smoke_reference,root,'fixture','eager',1,2048)
        smoke('eager',token=8);rejected(functional_smoke_reference,root,'fixture','eager',1,2048)
        smoke('eager',context=1024);rejected(functional_smoke_reference,root,'fixture','eager',1,2048)
    host=socket.gethostname();session={'slurm_job_id':'unit-test-only','expected_hostname':host}
    with patch.object(h200_session,'SESSION',session),patch.dict(os.environ,{'SLURM_JOB_ID':'unit-test-only'}),patch.object(h200_session.time,'sleep'):
        with patch.object(h200_session.subprocess,'check_output',side_effect=[subprocess.TimeoutExpired('scontrol',15),'JobState=RUNNING EndTime=2030-01-01T00:00:00 NodeList='+host.split('.')[0]]) as query:
            assert h200_session.verify_allocation()['state']=='RUNNING' and query.call_count==2
    import datetime
    scheduled=datetime.datetime.fromisoformat('2030-01-01T00:00:00').timestamp()
    assert h200_session.allocation_cutoff('EndTime=2030-01-01T00:00:00',scheduled+3600)==(scheduled-1800,scheduled)
    assert h200_session.allocation_cutoff('EndTime=2030-01-01T00:00:00',scheduled-7200)==(scheduled-7200,scheduled)
    rejected(h200_session.allocation_cutoff,'EndTime=Unknown',scheduled)
    with patch.object(h200_session,'SESSION',session),patch.dict(os.environ,{'SLURM_JOB_ID':'unit-test-only'}),patch.object(h200_session.time,'sleep'):
        with patch.object(h200_session.subprocess,'check_output',return_value='JobState=COMPLETED NodeList='+host.split('.')[0]) as query:
            rejected(h200_session.verify_allocation);assert query.call_count==1
        with patch.object(h200_session.subprocess,'check_output',side_effect=subprocess.TimeoutExpired('scontrol',15)) as query:
            try:h200_session.verify_allocation()
            except RuntimeError:pass
            else:raise AssertionError('unavailable allocation authorized launch')
            assert query.call_count==3
    with tempfile.TemporaryDirectory() as directory,patch.object(h200_session,'SESSION',{}),patch.object(h200_session,'LAUNCH_CUTOFF',0),patch.object(h200_session.subprocess,'Popen') as launch:
        assert h200_session.run(Path(directory),'cutoff_fixture',['never-run'],gpu=False)['status']=='pending'
        launch.assert_not_called()
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);base=root/'capture';base.mkdir()
        out=base/'pass000';out.mkdir()
        pub=base/'pass000.json'
        expected={'prompt_sha256':'fixed','trials':[{'output_ids':[[1,2,3]]}]}
        (out/'run.json').write_text(json.dumps({'status':'complete','cases':[{
            'prompt_sha256':'fixed','trials':[{'output_ids':[[1,2,3]]}]*3}]}))
        (out/'selected_interval.json').write_text(json.dumps({'status':'complete','ticks':[4,5,6]}))
        pub.write_text(json.dumps({'status':'complete','output':str(out)}))
        assert qualify_replays(base,expected,root)['passed']
        pub.write_text(json.dumps({'status':'running','output':str(out)}))
        assert not qualify_replays(base,expected,root)['passed']
        pub.write_text(json.dumps({'status':'complete','output':str(out)}))
        (base/'pass001').mkdir()
        assert not qualify_replays(base,expected,root)['passed']
    # A trace reader must establish actual intersections and reject missing kernels.
    import sqlite3
    from shoreline_timeline import native_overlap
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'native.sqlite';db=sqlite3.connect(path)
        db.executescript('CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT); CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER);')
        db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'transfer'),(2,'arithmetic')])
        for condition,ratio in enumerate((1,2,4,8,16,32)):
            for trial in range(10):
                start=(condition*10+trial)*10000
                db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?)',(start,start+9000,f'overlap_ratio{ratio}_concurrent'))
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1)',[(start+10+j*100,start+90+j*100) for j in range(4)])
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,2)',[(start+20+j*100,start+80+j*100) for j in range(ratio)])
        db.commit()
        proof=native_overlap(path)
        assert len(proof['conditions'])==60 and all(r['same_pass_overlap_ms']>0 for r in proof['conditions'])
        db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=1');db.commit();db.close()
        rejected(native_overlap,path)
    from shoreline_accounting import useful
    import struct
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        config={'model_type':'qwen2','num_attention_heads':2,'num_key_value_heads':1,'hidden_size':8,
                'num_hidden_layers':2,'tie_word_embeddings':True}
        (root/'config.json').write_text(json.dumps(config))
        header={'model.embed_tokens.weight':{'shape':[16,8],'dtype':'BF16','data_offsets':[0,256]}}
        payload=json.dumps(header).encode();(root/'model.safetensors').write_bytes(struct.pack('<Q',len(payload))+payload)
        u=useful(root,1,8)
        from shoreline_accounting import headers
        raw=(root/'config.json').read_text()
        manifest=root/'headers.json'
        manifest.write_text(json.dumps({'schema_version':2,'model_revision':'1'*40,
            'model_config_json_text':raw,'model_config_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'model_config':config,'safetensors_headers':headers(root)}))
        assert useful(manifest,1,8)==u, 'durable header accounting must equal actual checkpoint headers'
        damaged=json.loads(manifest.read_text());damaged['model_config_sha256']='0'*64
        manifest.write_text(json.dumps(damaged));rejected(useful,manifest,1,8)
        assert u['ticks'][0]['historical_kv_bytes']==2*2*1*11*1*4*2
        assert u['U_window_unique']-u['U_window_initial_external_main']==2*2*1*2*1*4*2
        assert u['ticks'][0]['required_weight_bytes']==256,'tied output storage counted once per tick'
        config['sliding_window']=4;(root/'config.json').write_text(json.dumps(config))
        rejected(useful,root,1,8)
    from trace_raw_audit import audit,instruction
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);path=root/'kernel.trace';prefix='0 '*11
        wait=prefix+'0000 00000001 0 SYNCS.PHASECHK.TRANS64.TRYWAIT 0 8 1 7ffe00001234 0 0 NoVal\n'
        arrive=prefix+'0010 00000001 0 SYNCS.ARRIVE 0 8 1 7ffe00001234 0 0 NoVal\n'
        path.write_text('-binary version = 90\n-enable lineinfo = 0\n'+wait)
        assert audit(root)['status']=='rejected_wait_pruning_required'
        path.write_text('-binary version = 90\n-enable lineinfo = 0\n'+wait+arrive)
        a=audit(root)
        assert a['status']=='passed_no_wait_pruning' and a['files'][0]['raw_active_lane_record_contributions']==2
        rejected(instruction,prefix+'0000 ffffffff 0 SYNCS.ARRIVE 0 8 3 7ffe00001234 0 0 NoVal')
    b=latency_bounds(10,2,8)
    assert b['O_interval']==[0,2]
    for e in b['coupled_endpoints']:
        assert math.isclose(10,2+8-e['X']+e['O'])
        assert 0<=e['X']<=2
    assert not latency_bounds(1,2,8)['feasible']
    rejected(latency_bounds,10,-1,2)
    losing=sensitivity(b,1,.7,.2)
    assert not losing['memory_service_improves'] and losing['M_new']>8
    assert sensitivity(b,1,1.2,0)['memory_service_improves']
    assert sensitivity(b,1,1,0)['same_residual_same_eta_time']==[10,10]
    rejected(hierarchical_interval,[[1]*10]*2)
    rejected(hierarchical_interval,[[1]*9]*3)
    rejected(paired_ratio,[[1]*10]*3,[[0]*10]*3)
    equal=paired_ratio([[1]*10]*3,[[1]*10]*3,draws=100)
    assert equal['ci95']==[1,1] and equal['passes_5_percent_noninferiority']
    losing=paired_ratio([[1.1]*10]*3,[[1]*10]*3,draws=100)
    assert not losing['passes_5_percent_noninferiority']
    print('PASS: acquisition sufficiency, positive rates, coupled bounds, impossible model, losing bandwidth and noninferiority controls')


if __name__=='__main__':main()
