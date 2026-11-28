#!/usr/bin/env python3
"""Review the synthetic native component without promoting it to model KV."""
import argparse
import hashlib
from pathlib import Path
import struct
import traceback
from h200_priority_campaign import load, write, sha, limitation
from h200_native_transport_probe import SHAPE, LAYER_BYTES, ENGINE_SHA, validate_pair


def check_trials(producer, consumer):
    assert len(producer) == len(consumer) == 6
    digests = {value: hashlib.sha256(struct.pack('<f', float(value))[2:]*
                                   (LAYER_BYTES//2)).hexdigest() for value in range(128)}
    for index, (sent, received) in enumerate(zip(producer, consumer)):
        assert sent['trial'] == received['trial'] == index
        assert sent['warmup'] == received['warmup'] == (index < 3)
        assert sent['request_id'] == received['request_id']
        assert sent['status'] == received['status'] == 'known_data_transport_passed'
        assert sent['all_native_send_completions_observed'] is True
        assert len(sent['layers']) == len(received['layers']) == 64
        for layer, (source, target) in enumerate(zip(sent['layers'], received['layers'])):
            identifier = sent['request_id']+f'#model.layers.{layer}.self_attn.attn'
            for row in (source, target):
                assert row['layer_index'] == layer and row['tensor_id'] == identifier
                assert row['shape'] == list(SHAPE) and row['dtype'] == 'torch.bfloat16'
                assert row['nbytes'] == LAYER_BYTES
                assert row['sha256'] == digests[(index*64+layer) % 128]
            assert target['received_sha256'] == source['sha256']
            assert target['known_values_verified'] is True


def review(root):
    root = Path(root).resolve(); errors = []; records = []
    pair = load(root/'pair.json')
    try:
        disposition = load(root/'results/pair_disposition.json')
        assert disposition['status'] == 'component_captured_pending_CPU_review'
        assert not disposition['model_KV_verified'] and not disposition['serving_qualified']
        sessions = []; runs = []
        for role, bound in zip(('producer', 'consumer'), disposition['roles']):
            subroot = root/role; session_path = subroot/'session.json'; session = load(session_path)
            validate_pair(pair,session,role,session['gpu_uuid'],pair['job_id'],pair['hostname'])
            process_path = subroot/'results/processes/native_known_data_transport.json'
            run_path = subroot/'runs/component/run.json'
            assert Path(bound['process']) == process_path and Path(bound['run']) == run_path
            assert sha(process_path) == bound['process_sha256'] and sha(run_path) == bound['run_sha256']
            process = load(process_path); run = load(run_path)
            assert process['status'] == 'complete' and process['exit_code'] == 0
            assert process['source_freeze_identity'] == session['frozen_source_identity']
            assert not process['monitor']['failure']
            monitor_path = Path(process['monitor']['source']); samples = [
                __import__('json').loads(line) for line in monitor_path.read_text().splitlines()]
            assert samples and any(row['pids'] for row in samples)
            assert all(row['gpu_uuid'] == session['gpu_uuid'] and not row.get('rejection')
                and all(value is True for value in row['owned_ancestry'].values()) for row in samples)
            assert sha(subroot/'logs/native_known_data_transport.log') == process['log_sha256']
            idle = load(subroot/'results/idle_checks/native_known_data_transport.json')
            assert idle['passed'] is True and idle['required_idle_s'] >= 10
            assert load(subroot/'results/current_process.json')['status'] == 'idle'
            entry_sha = process['script_sha256']['h200_native_transport_probe.py']
            assert sha(subroot/'environment/source_snapshots'/(entry_sha+'.py')) == entry_sha
            assert run['status'] == 'known_data_transport_probe_passed'
            assert run['role'] == role and run['gpu_uuid'] == session['gpu_uuid']
            assert run['session_sha256'] == sha(session_path)
            assert run['pair_config_sha256'] == sha(root/'pair.json')
            assert run['numerical_source_identity'] == session['frozen_source_identity']
            assert run['native_engine_sha256'] == ENGINE_SHA
            assert run['optional_dependencies']['msgpack']=='1.1.1'
            assert run['optional_dependencies']['package_file_sha256']
            assert run['synthetic_payload'] is True
            assert run['model_KV_generated'] is False and run['model_KV_injected'] is False
            assert run['qualified_performance'] is False and run['physical_HBM_or_NVLink_bytes'] is None
            assert run['requested_tensor_bytes_per_trial'] == 64*LAYER_BYTES
            assert run['hardware']['uuid'] == session['gpu_uuid'] and 'H200' in run['hardware']['name']
            sessions.append(session); runs.append(run)
            records.append({'role':role,'gpu_uuid':session['gpu_uuid'],
                'session':str(session_path),'session_sha256':sha(session_path),
                'run':str(run_path),'run_sha256':sha(run_path),
                'process':str(process_path),'process_sha256':sha(process_path),
                'monitor':str(monitor_path),'monitor_sha256':sha(monitor_path)})
        assert len({s['frozen_source_identity'] for s in sessions}) == 1
        assert runs[0]['optional_dependencies']==runs[1]['optional_dependencies']
        check_trials(runs[0]['trials'],runs[1]['trials'])
    except (OSError,ValueError,KeyError,AssertionError,TypeError,IndexError) as error:
        errors.append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    result = {'status':'known_data_transport_functional_review_passed' if not errors else 'failed_CPU_review',
        'classification':'CPU review of synthetic native transport component',
        'job_id':pair['job_id'],'gpu_uuids':[r['gpu_uuid'] for r in pair['roles'].values()],
        'implementation_sha256':sha(Path(__file__)),
        'pair_sha256':sha(root/'pair.json'),'roles':records,'errors':errors,
        'optional_dependencies':runs[0].get('optional_dependencies') if not errors else None,
        'model_KV_verified':False,'serving_qualified':False,'quantitative_performance_qualified':False,
        'physical_HBM_or_NVLink_bytes':None,
        'scope':'Six known-data transfer trials per role:3warm/3diagnostic, one process per device. Independently recomputed BF16 payload hashes, paired completion and source/isolation checks only. No model KV injection, serving, clean latency or calibrated bandwidth.'}
    write(root/'results/native_component_CPU_review.json',result)
    with (root/'analysis/report.md').open('a') as stream:
        stream.write('\n## Native component CPU review\n\n'+result['status']+'. '+result['scope']+'\n')
    if errors:
        limitation(root,'native_component_CPU_review','Paired synthetic component failed provenance/payload/completion checks',
            'results/native_component_CPU_review.json','Independent known BF16 hashes and paired raw process/isolation review',
            'Inspect first error before a bounded repair; model and serving pilots remain ineligible')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    result = review(parser.parse_args().root)
    print({'status':result['status'],'model_KV_verified':False,'serving_qualified':False})
