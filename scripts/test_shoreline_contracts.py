#!/usr/bin/env python3
"""Negative and boundary contracts for new statistical/replay qualification."""
import math
from shoreline_statistics import latency_bounds,sensitivity,hierarchical_interval,paired_ratio


def rejected(function,*args):
    try:function(*args)
    except (AssertionError,ValueError):return
    raise AssertionError('invalid inputs accepted')


def main():
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
    host=socket.gethostname();session={'slurm_job_id':'unit-test-only','expected_hostname':host}
    with patch.object(h200_session,'SESSION',session),patch.dict(os.environ,{'SLURM_JOB_ID':'unit-test-only'}),patch.object(h200_session.time,'sleep'):
        with patch.object(h200_session.subprocess,'check_output',side_effect=[subprocess.TimeoutExpired('scontrol',15),'JobState=RUNNING NodeList='+host.split('.')[0]]) as query:
            assert h200_session.verify_allocation()['state']=='RUNNING' and query.call_count==2
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
