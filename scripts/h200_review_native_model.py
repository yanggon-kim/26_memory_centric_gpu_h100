#!/usr/bin/env python3
"""Review paired model KV functional phases; no deployment timing inference."""
import argparse
import json
from pathlib import Path
import traceback
from h200_priority_campaign import load, write, sha, valid_tokens, strict_reference_record, limitation
from h200_native_model_pilot import validate_scope, validate_queries, CONNECTOR_SHA, ENGINE_SHA
from h200_native_pd_contract import validate_layers, phase_contract


def verified_process(base, label, entry, session):
    process_path=base/'results/processes'/(label+'.json');process=load(process_path)
    assert process['status']=='complete' and process['exit_code']==0
    assert process['source_freeze_identity']==session['frozen_source_identity']
    assert process['monitor'] and not process['monitor']['failure']
    monitor_path=Path(process['monitor']['source'])
    samples=[json.loads(line) for line in monitor_path.read_text().splitlines()]
    assert samples and any(row['pids'] for row in samples)
    assert all(row['gpu_uuid']==session['gpu_uuid'] and not row.get('rejection') and
        all(value is True for value in row['owned_ancestry'].values()) for row in samples)
    entry_sha=process['script_sha256'][entry]
    snapshot=base/'environment/source_snapshots'/(entry_sha+'.py')
    assert sha(snapshot)==entry_sha
    log=base/'logs'/(label+'.log');assert sha(log)==process['log_sha256']
    idle=base/'results/idle_checks'/(label+'.json')
    assert load(idle)['passed'] is True and load(idle)['required_idle_s']>=10
    return {'process':str(process_path),'process_sha256':sha(process_path),
        'monitor':str(monitor_path),'monitor_sha256':sha(monitor_path),
        'log':str(log),'log_sha256':sha(log),'entry_snapshot':str(snapshot),'entry_sha256':entry_sha,
        'idle_check':str(idle),'idle_check_sha256':sha(idle)}


def host_pool_accounting(source,target,payload_bytes):
    """Actual native pool APIs; delayed releases stay with their owning request."""
    assert source['host_pool_events']==[]
    groups={name:[] for name in ('store_tensor','load_tensor','free')}
    for event in target['host_pool_events']:
        assert event['request_id']==target['request_id'] and event['status']=='complete'
        assert event['host_start_unix_ns']<=event['host_end_unix_ns']
        assert type(event['address']) is int and event['address']>0
        groups[event['operation']].append(event)
    stores={e['address']:e for e in groups['store_tensor']}
    loads={e['address']:e for e in groups['load_tensor']}
    released={e['address']:e for e in groups['free']}
    assert len(stores)==len(groups['store_tensor']) and len(loads)==len(groups['load_tensor'])
    assert len(released)==len(groups['free']) and stores.keys()==loads.keys() and released.keys()<=stores.keys()
    for addr,stored in stores.items():
        assert 0<stored['requested_tensor_bytes']==loads[addr]['requested_tensor_bytes']<=stored['allocated_block_bytes']
        assert loads[addr]['host_start_unix_ns']>=stored['host_end_unix_ns']
        if addr in released:
            assert released[addr]['allocated_block_bytes_before']==stored['allocated_block_bytes']
            assert released[addr]['host_start_unix_ns']>=loads[addr]['host_end_unix_ns']
    total=sum(e['requested_tensor_bytes'] for e in stores.values());assert total<=payload_bytes
    retained=[{'address':addr,'request_id':target['request_id'],'requested_tensor_bytes':e['requested_tensor_bytes']}
              for addr,e in stores.items() if addr not in released]
    return {'native_host_spill_tensor_bytes':total,
        'native_host_restore_tensor_bytes':sum(e['requested_tensor_bytes'] for e in loads.values()),
        'retained_before_owned_cleanup':retained,
        'physical_HBM_or_link_bytes':None,
        'scope':'Actual native pool store/load/free API payload counts; delayed free events remain assigned to their original request. Outstanding bytes are observed before owned engine/process cleanup, not evidence of a post-exit leak. Host call spans include native copies/blocking and instrumentation; no clean latency or GPU-staging peak claim.'}


def check_pair_trials(producer, consumer, expected, prompt, pair):
    assert len(producer) == len(consumer) == 12
    contract = phase_contract(prompt,expected[0])
    payload_bytes = contract['analytical_requested_KV_tensor_bytes']
    summaries = []
    for index,(source,target) in enumerate(zip(producer,consumer)):
        payload = index >= 6
        mode = 'payload_proof' if payload else 'metadata_control'
        for row in (source,target):
            assert row['status'] == 'functional_phase_passed'
            assert row['mode'] == mode and row['payload_proof'] is payload
            assert row['trial'] == index%6 and row['warmup'] is (index%6 < 3)
            assert row['internal_request_id'] == row['request_id']
        rid = source['request_id']; assert target['request_id'] == rid
        validate_queries(source['scheduler_rows'],'producer',rid,context=len(prompt))
        validate_queries(target['scheduler_rows'],'consumer',rid,context=len(prompt))
        assert len(source['outputs']) == 1 and source['outputs'][0]['output_ids'] == expected[:1]
        assert len(target['outputs']) == 63
        for step,row in enumerate(target['outputs']):
            assert row['output_ids'] == expected[1:step+2]
        assert expected[:1]+target['outputs'][-1]['output_ids'] == expected
        assert source['received'] == source['injected'] == [] and target['sent'] == []
        sent,received = source['sent'],target['received']
        assert len(sent) == len(received) == 64
        assert {r['layer_index'] for r in sent} == {r['layer_index'] for r in received} == set(range(64))
        sends = {r['tensor_id']:r for r in sent}; recvs = {r['tensor_id']:r for r in received}
        assert len(sends) == len(recvs) == 64 and sends.keys() == recvs.keys()
        assert set(target['native_receive_completions']) == set(sends)
        for identifier,sent_row in sends.items():
            received_row = recvs[identifier]
            assert identifier.startswith(rid+'#') and sent_row['completed'] is True
            for key in ('layer_index','shape','dtype','contiguous','nbytes'):
                assert sent_row[key] == received_row[key]
            assert sent_row['dtype'] == 'torch.bfloat16' and sent_row['contiguous'] is True
            assert sent_row['remote_address'] == f"127.0.0.1:{pair['roles']['consumer']['port']}"
            assert received_row['remote_address'] == f"127.0.0.1:{pair['roles']['producer']['port']}"
            if payload: assert sent_row['sha256'] == received_row['sha256']
            else: assert 'sha256' not in sent_row and 'sha256' not in received_row
        assert sum(r['nbytes'] for r in sent) == payload_bytes
        if payload:
            checked = validate_layers(sent,target['injected'],rid,payload_bytes)
            assert target['payload_gate'] == checked
        else:
            assert target['injected'] == []
        summary={'request_id':rid,'mode':mode,'trial':index%6,'warmup':index%6<3,
            'actual_requested_KV_tensor_bytes':payload_bytes,'canonical_outputs':64,
            'consumer_original_prompt_queries':0,'consumer_canonical_decode_queries':63}
        if pair.get('protocol')=='native_model_KV_long_context_functional_v2':
            summary['native_host_pool']=host_pool_accounting(source,target,payload_bytes)
        summaries.append(summary)
    return summaries


def review(root):
    root = Path(root).resolve(); pair = load(root/'pair.json'); errors = []; records = []; summaries = []
    reference_provenance=[]
    try:
        reference_path = Path(pair['reference_run']); reference = load(reference_path)
        assert sha(reference_path) == pair['reference_run_sha256'] and valid_tokens(reference)
        hf = load(pair['strict_HF_reference'])
        assert sha(pair['strict_HF_reference']) == pair['strict_HF_reference_sha256']
        assert strict_reference_record(hf,sha(reference_path))
        assert hf['gpu_uuid'] == reference['hardware']['uuid'] == pair['roles']['producer']['gpu_uuid']
        base=reference_path.parents[2];reference_session=load(base/'session.json')
        assert reference_session['gpu_uuid']==pair['roles']['producer']['gpu_uuid']
        assert reference_session['slurm_job_id']==pair['job_id'] and reference_session['expected_hostname']==pair['hostname']
        label=reference_path.parent.name
        assert Path(pair['strict_HF_reference'])==base/'results'/(label+'_hf.json')
        for suffix,entry in (('', 'controlled_vllm_experiments.py'),('_hf','hf_full_reference.py')):
            reference_provenance.append(verified_process(base,label+suffix,entry,reference_session))
        assert all(len(arm[kind])==3 for arm in (hf,hf['static_cache_gate']) for kind in ('warmups','trials'))
        assert hf['static_cache_gate']['adapter_sha256']==load(base/'results/processes'/(label+'_hf.json'))['script_sha256']['software_decode_experiments.py']
        assert not hf.get('deterministic_reference_diagnostic'), 'native functional prerequisite requires its declared default strict oracle'
        component = load(pair['native_component_CPU_review'])
        assert sha(pair['native_component_CPU_review']) == pair['native_component_CPU_review_sha256']
        assert component['status'] == 'known_data_transport_functional_review_passed' and not component['errors']
        assert component['optional_dependencies']['msgpack']=='1.1.1'
        assert component['job_id'] == pair['job_id'] and set(component['gpu_uuids']) == {r['gpu_uuid'] for r in pair['roles'].values()}
        for item in component['roles']:
            for name in ('run','process','monitor','session'):
                assert sha(item[name]) == item[name+'_sha256']
        runs = []; identities = []
        for role in ('producer','consumer'):
            subroot = root/role; session_path = subroot/'session.json'; session = load(session_path)
            validate_scope(pair,session,role,session['gpu_uuid'],pair['job_id'],pair['hostname'])
            process_path = subroot/'results/processes/native_model_KV_functional.json'; process = load(process_path)
            assert process['status'] == 'complete' and process['exit_code'] == 0 and not process['monitor']['failure']
            assert process['source_freeze_identity'] == session['frozen_source_identity']
            entry_sha = process['script_sha256']['h200_native_model_pilot.py']
            assert sha(subroot/'environment/source_snapshots'/(entry_sha+'.py')) == entry_sha
            assert sha(subroot/'logs/native_model_KV_functional.log') == process['log_sha256']
            monitor_path = Path(process['monitor']['source'])
            samples = [json.loads(line) for line in monitor_path.read_text().splitlines()]
            assert samples and any(row['pids'] for row in samples)
            assert all(row['gpu_uuid'] == session['gpu_uuid'] and not row.get('rejection')
                and all(v is True for v in row['owned_ancestry'].values()) for row in samples)
            assert load(subroot/'results/current_process.json')['status'] == 'idle'
            run_path = subroot/'runs/model/run.json'; run = load(run_path)
            assert run['status'] == 'native_model_functional_captured_pending_pair_review'
            assert run['role'] == role and run['gpu_uuid'] == session['gpu_uuid'] == run['hardware']['uuid']
            assert run['session_sha256'] == sha(session_path) and run['pair_sha256'] == sha(root/'pair.json')
            assert run['source_identity'] == session['frozen_source_identity']
            assert run['connector_sha256'] == CONNECTOR_SHA and run['native_engine_sha256'] == ENGINE_SHA
            assert run['reference_run_sha256'] == sha(reference_path)
            assert run['optional_dependencies']==component['optional_dependencies']
            assert run['versions']==reference['versions']
            assert session['frozen_source_identity']==reference_session['frozen_source_identity']
            assert run['quantitative_performance_qualified'] is False and run['serving_qualified'] is False
            assert run['model_KV_generated'] == (role=='producer')
            assert run['model_KV_injected_verified'] == (role=='consumer')
            cleanup = load(subroot/'runs/model/cleanup.json')
            assert cleanup['status'] == 'complete' and cleanup['distributed_initialized_after'] is False
            runs.append(run); identities.append(session['frozen_source_identity'])
            records.append({'role':role,'gpu_uuid':session['gpu_uuid'],
                'session':str(session_path),'session_sha256':sha(session_path),
                'run':str(run_path),'run_sha256':sha(run_path),'process':str(process_path),
                'process_sha256':sha(process_path),'monitor':str(monitor_path),'monitor_sha256':sha(monitor_path)})
        assert len(set(identities)) == 1
        case = reference['cases'][0]; assert (case['batch_size'],case['context_length'],case['generated_tokens']) == (1,pair['context'],64)
        summaries = check_pair_trials(runs[0]['trials'],runs[1]['trials'],case['trials'][0]['output_ids'][0],case['prompt_ids'][0],pair)
        if pair['protocol']=='native_model_KV_long_context_functional_v2':
            assert all(r['unassigned_host_pool_events']==[] for r in runs)
            retained=[item for s in summaries for item in s['native_host_pool']['retained_before_owned_cleanup']]
            key=lambda item:(item['request_id'],item['address'])
            assert sorted(retained,key=key)==sorted(runs[1]['host_pool_retained_before_owned_cleanup'],key=key)
            assert runs[0]['host_pool_retained_before_owned_cleanup']==[]
    except (OSError,ValueError,KeyError,AssertionError,TypeError,IndexError) as error:
        errors.append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    result = {'status':'native_model_functional_review_passed' if not errors else 'failed_CPU_review',
        'classification':'CPU review of real-model native functional handoff',
        'job_id':pair['job_id'],'roles':records,'errors':errors,'requests':summaries,
        'reference_process_provenance':reference_provenance,
        'functional_model_KV_verified':not errors,'quantitative_performance_qualified':False,
        'serving_qualified':False,'primary_J_per_token':None,'physical_HBM_or_NVLink_bytes':None,
        'implementation_sha256':sha(Path(__file__)),
        'scope':'Metadata-only functional controls and separate intrusive payload proofs, each3warm/3diagnostic on one owned pair/process cohort. Producer waits complete sends before admitting consumer. Exact combined64outputs and actual query/transfer/injection evidence only; no native deployment timing, independent-acquisition comparison, tuned or loaded serving, physical traffic or energy qualification.'}
    write(root/'results/native_model_CPU_review.json',result)
    with (root/'analysis/report.md').open('a') as stream:
        stream.write('\n## Native model functional CPU review\n\n'+result['status']+'. '+result['scope']+'\n')
    if errors:
        limitation(root,'native_model_CPU_review','Native model phase/source/payload/output CPU gate failed',
            'results/native_model_CPU_review.json','Retain metadata-only controls separately from intrusive payload proof',
            'Inspect the first error before a bounded repair; no deployment inference')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    result = review(parser.parse_args().root); print({'status':result['status'],'serving_qualified':False})
