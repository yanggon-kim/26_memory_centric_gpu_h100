#!/usr/bin/env python3
"""Prospective real-model native KV role; functional, never a serving result.

Requires a separately supervised paired controller, a validated native component
and a strict full-output stock reference. First run metadata-only controls, then
separate intrusive source/received/injected payload proofs. No installed edits.
"""
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import traceback
from h200_native_pd_contract import request_id, phase_contract

CONNECTOR_SHA = '13e2e1b5baa8cd503c68a0729fe50124ee36d41320b63c94335bd7d0e04266a9'
ENGINE_SHA = 'bcbb73c586bc82d51c84e89e6b0dca9dc2354fa7a48856c1ae3a9f7dc45fd17a'


def validate_scope(pair, session, role, uuid, job, hostname):
    assert pair['protocol'] == 'native_model_KV_functional_v1'
    assert pair['synthetic_payload'] is False
    assert pair['job_id'] == session['slurm_job_id'] == job
    assert pair['hostname'] == session['expected_hostname'] == hostname
    assert set(pair['roles']) == {'producer', 'consumer'} and role in pair['roles']
    assert len({r['gpu_uuid'] for r in pair['roles'].values()}) == 2
    assert pair['roles'][role]['gpu_uuid'] == session['gpu_uuid'] == uuid
    assert pair['host'] == '127.0.0.1'
    request_id(pair['request_prefix'], pair['host'], pair['roles']['producer']['port'],
               pair['host'], pair['roles']['consumer']['port'])
    assert pair['generated_tokens'] == 64 and pair['context'] == 128


def validate_queries(rows, role, rid, context=128):
    assert rows and all(not r['any_preempted_request'] for r in rows)
    queries = [q for row in rows for q in row['requests']]
    assert len(queries) == (1 if role == 'producer' else 63)
    assert all(q['request_id'] == rid for q in queries)
    if role == 'producer':
        assert queries[0]['computed_tokens_before_step'] == 0
        assert queries[0]['scheduled_input_queries'] == context
        assert queries[0]['original_prompt_tokens'] == context
    else:
        assert [q['computed_tokens_before_step'] for q in queries] == list(range(context, context+63))
        assert all(q['scheduled_input_queries'] == 1 and q['original_prompt_tokens'] == context+1 for q in queries)
        assert sum(q['prompt_input_queries'] for q in queries) == 1
        assert sum(q['post_prompt_input_queries'] for q in queries) == 62


def tensor_record(tensor_id, tensor, payload):
    assert tensor.is_cuda and tensor.is_contiguous() and str(tensor.dtype) == 'torch.bfloat16'
    match = re.search(r'\.layers\.(\d+)\.', tensor_id)
    assert match is not None
    row = {'tensor_id': tensor_id, 'layer_index': int(match.group(1)),
           'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
           'contiguous': True, 'nbytes': tensor.numel()*tensor.element_size()}
    if payload:
        import torch
        row['sha256'] = hashlib.sha256(tensor.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    return row


class NativeLedger:
    """Instance hooks; payload mode deliberately adds GPU ordering and copies."""
    def __init__(self, connector):
        self.connector = connector
        self.native = connector.p2p_nccl_engine
        self.active = None
        self.originals = {name: getattr(self.native, name) for name in ('send_tensor', 'recv_tensor', 'get_finished')}
        self.load = connector.start_load_kv

        def send(identifier, tensor, remote):
            if self.active is not None:
                self.active['sent'].append(tensor_record(identifier, tensor, self.active['payload_proof']))
            return self.originals['send_tensor'](identifier, tensor, remote)

        def recv(identifier, remote):
            tensor = self.originals['recv_tensor'](identifier, remote)
            if self.active is not None:
                assert tensor is not None, 'native receive returned no payload'
                self.active['received'].append(tensor_record(identifier, tensor, self.active['payload_proof']))
            return tensor

        def finished(ids, layers):
            if self.active is not None:
                rid = self.active['request_id']
                self.active['native_receive_completions'] = sorted(set(self.active['native_receive_completions']) |
                    self.native.recv_request_id_to_tensor_ids.get(rid, set()))
            return self.originals['get_finished'](ids, layers)

        def loaded(context, **kwargs):
            if self.active is None or not self.active['payload_proof'] or connector.is_producer:
                return self.load(context, **kwargs)
            metadata = connector._get_connector_metadata()
            result = self.load(context, **kwargs)
            received = {r['tensor_id']: r for r in self.active['received']}
            for req in metadata.requests:
                for name, layer in context.no_compile_layers.items():
                    cache = getattr(layer, 'kv_cache', None)
                    if cache is None:
                        continue
                    identifier = req.request_id+'#'+name
                    source = received[identifier]
                    if cache.shape[1] == 2:
                        injected = cache[req.block_ids[:source['shape'][0]], ...]
                    else:
                        assert cache.shape[0] == 2
                        injected = cache[:, req.block_ids[:source['shape'][1]], ...]
                    row = tensor_record(identifier, injected, True)
                    assert row['shape'] == source['shape'] and row['sha256'] == source['sha256']
                    row.update(injected=True, completed=True)
                    self.active['injected'].append(row)
            return result

        self.native.send_tensor = send
        self.native.recv_tensor = recv
        self.native.get_finished = finished
        connector.start_load_kv = loaded

    def close(self):
        for name, method in self.originals.items():
            setattr(self.native, name, method)
        self.connector.start_load_kv = self.load


def main(args):
    from software_decode_experiments import dump
    from h200_native_transport_probe import wait_json, all_native_sends_completed
    session_path = Path(os.environ['H200_SESSION_CONFIG'])
    session = json.loads(session_path.read_text()); pair = json.loads(args.pair_config.read_text())
    validate_scope(pair, session, args.role, args.gpu_uuid, os.environ.get('SLURM_JOB_ID'), socket.gethostname())
    shared = Path(pair['shared_directory']); assert shared.is_dir()
    reference_path = Path(pair['reference_run']); reference = json.loads(reference_path.read_text())
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == pair['reference_run_sha256']
    from h200_priority_campaign import valid_tokens, strict_reference_record
    assert valid_tokens(reference)
    hf_path = Path(pair['strict_HF_reference']); hf = json.loads(hf_path.read_text())
    assert hashlib.sha256(hf_path.read_bytes()).hexdigest() == pair['strict_HF_reference_sha256']
    assert strict_reference_record(hf, pair['reference_run_sha256'])
    component_path = Path(pair['native_component_CPU_review'])
    assert hashlib.sha256(component_path.read_bytes()).hexdigest() == pair['native_component_CPU_review_sha256']
    component = json.loads(component_path.read_text())
    assert component['status'] == 'known_data_transport_functional_review_passed'
    assert component['job_id'] == pair['job_id']
    assert set(component['gpu_uuids']) == {r['gpu_uuid'] for r in pair['roles'].values()}
    case = reference['cases'][0]
    assert (case['batch_size'], case['context_length'], case['generated_tokens']) == (1,128,64)
    prompt = case['prompt_ids'][0]; expected = case['trials'][0]['output_ids'][0]
    assert reference['config']['seed'] == 20261013 and reference['engine_settings']['dtype'] == 'bfloat16'
    assert reference['config']['model_id'] == 'Qwen/Qwen2.5-32B-Instruct'
    assert reference['config']['model_revision'] == '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
    assert reference['engine_settings']['tensor_parallel_size'] == 1
    assert not reference['engine_settings']['enable_prefix_caching']
    assert not reference['config'].get('decode_conditioning')
    import h200_session
    allocation = h200_session.verify_allocation()
    assert allocation is not None
    deadline = min(time.time()+660, pair['deadline_unix_s'],
                   allocation['effective_launch_cutoff_unix_s'],
                   dt.datetime.fromisoformat(session['launch_cutoff']).timestamp())
    finished = threading.Event()
    def watchdog():
        if not finished.wait(max(0, deadline-time.time())):
            dump(args.output/'deadline.json', {'status':'failed_deadline','deadline_unix_s':deadline})
            os._exit(124)
    threading.Thread(target=watchdog, daemon=True).start()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        VLLM_HOST_IP='127.0.0.1', VLLM_DISABLE_REQUEST_ID_RANDOMIZATION='1',
        VLLM_ENABLE_V1_MULTIPROCESSING='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
    ledger = {'status':'running','classification':'measured-real-GPU-functional-instrumented',
        'protocol':pair['protocol'],'role':args.role,'gpu_uuid':args.gpu_uuid,
        'model_KV_generated':args.role=='producer','model_KV_injected_verified':False,
        'source_identity':session['frozen_source_identity'],
        'session_sha256':hashlib.sha256(session_path.read_bytes()).hexdigest(),
        'allocation_verification':allocation,
        'reference_run_sha256':pair['reference_run_sha256'], 'trials':[],
        'pair_sha256':hashlib.sha256(args.pair_config.read_bytes()).hexdigest(),
        'quantitative_performance_qualified':False, 'serving_qualified':False,
        'functional_synchronization':'Producer waits all native send completions before publishing first token; consumer starts afterward. This serializes the handoff and is not default deployed serving timing.',
        'scope':'Twelve sequential paired functional requests:3warm/3diagnostic metadata-only controls, then3warm/3diagnostic intrusive payload proofs. Not independent acquisitions, a producer-penalty timing comparison, loaded serving, clean latency, physical bytes or calibrated BW.'}
    dump(args.output/'run.json', ledger)
    nv = None; llm = None; hook = None
    try:
        import pynvml as nv
        nv.nvmlInit()
        from gpu_target import select, verify_cuda, describe
        handle, _ = select(nv, args.gpu_uuid)
        import torch
        verify_cuda(torch, args.gpu_uuid)
        import vllm
        from vllm.config import KVTransferConfig
        from vllm.sampling_params import RequestOutputKind
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p import p2p_nccl_connector, p2p_nccl_engine
        for module, expected_sha in ((p2p_nccl_connector,CONNECTOR_SHA),(p2p_nccl_engine,ENGINE_SHA)):
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected_sha
        ledger.update(hardware=describe(nv,handle,torch),versions={'torch':torch.__version__,
            'vllm':vllm.__version__,'cuda_runtime':torch.version.cuda},
            connector_sha256=CONNECTOR_SHA,native_engine_sha256=ENGINE_SHA)
        assert ledger['versions'] == reference['versions']
        cfg = KVTransferConfig(kv_connector='P2pNcclConnector',kv_role='kv_'+args.role,
            kv_rank=0 if args.role=='producer' else 1,kv_parallel_size=2,
            kv_port=pair['roles'][args.role]['port'])
        settings = dict(reference['engine_settings'], kv_transfer_config=cfg)
        ledger['engine_settings_without_KV_object'] = reference['engine_settings']
        ledger['KV_transfer_config'] = dataclasses.asdict(cfg)
        dump(args.output/'run.json', ledger)
        llm = vllm.LLM(**settings); engine = llm.llm_engine
        assert type(engine.engine_core).__name__ == 'InprocClient'
        connector = get_kv_transfer_group(); native = connector.p2p_nccl_engine
        assert type(connector).__name__ == 'P2pNcclConnector'
        assert native.zmq_address == f"127.0.0.1:{pair['roles'][args.role]['port']}"
        assert native.send_type == 'PUT_ASYNC'
        hook = NativeLedger(connector)
        dump(shared/(args.role+'_model_ready.json'), {'gpu_uuid':args.gpu_uuid,'job_id':pair['job_id']})
        peer_role = 'consumer' if args.role=='producer' else 'producer'
        ready = wait_json(shared/(peer_role+'_model_ready.json'), deadline)
        assert ready['gpu_uuid'] == pair['roles'][peer_role]['gpu_uuid'] and ready['job_id'] == pair['job_id']
        from h200_scheduler_observer import Observer
        for payload in (False,True):
            mode = 'payload_proof' if payload else 'metadata_control'
            for trial in range(6):
                rid = request_id(pair['request_prefix']+'_'+mode+'_'+str(trial),
                    pair['host'],pair['roles']['producer']['port'],pair['host'],pair['roles']['consumer']['port'])
                row = {'request_id':rid,'mode':mode,'payload_proof':payload,'trial':trial,'warmup':trial<3,
                       'status':'running','sent':[],'received':[],'injected':[],'native_receive_completions':[]}
                ledger['trials'].append(row); hook.active = row; dump(args.output/'run.json',ledger)
                marker = mode+'_'+str(trial)
                if args.role == 'consumer':
                    produced = wait_json(shared/(marker+'_first.json'), deadline)
                    assert produced['request_id'] == rid and produced['first_token'] == expected[0]
                    tokens = prompt+[produced['first_token']]; count = 63
                else:
                    tokens = prompt; count = 1
                sampling = vllm.SamplingParams(temperature=0,repetition_penalty=1.0,ignore_eos=True,
                    detokenize=False,max_tokens=count,seed=20261013,output_kind=RequestOutputKind.CUMULATIVE)
                observer = Observer(engine)
                try:
                    row['submit_unix_ns'] = time.time_ns()
                    internal = engine.add_request(rid,{'prompt_token_ids':tokens},sampling)
                    assert internal == rid, 'native producer/consumer request key differs'
                    outputs = []
                    while engine.has_unfinished_requests():
                        assert time.time() < deadline
                        batch = engine.step()
                        assert len(batch) == 1 and batch[0].request_id == rid
                        result = batch[0]; ids = list(result.outputs[0].token_ids)
                        assert len(ids) == len(outputs)+1
                        assert ids == (expected[:1] if args.role=='producer' else expected[1:len(ids)+1])
                        outputs.append({'output_ids':ids,'host_observed_unix_ns':time.time_ns()})
                    assert result.finished and result.outputs[0].finish_reason == 'length' and len(outputs) == count
                    row.update(outputs=outputs,scheduler_rows=observer.rows,observer_identity=observer.identity)
                    validate_queries(observer.rows,args.role,rid)
                finally:
                    observer.close()
                if args.role == 'producer':
                    assert len(row['sent']) == 64
                    identifiers = [r['tensor_id'] for r in row['sent']]
                    assert len(set(identifiers)) == 64
                    while not all_native_sends_completed(native,rid,identifiers):
                        assert time.time() < deadline; time.sleep(.001)
                    for sent in row['sent']: sent['completed'] = True
                    dump(shared/(marker+'_first.json'), {'request_id':rid,'first_token':expected[0],
                         'send_completion_unix_ns':time.time_ns(),'sent':row['sent']})
                    acknowledged = wait_json(shared/(marker+'_ack.json'), deadline)
                    assert acknowledged['request_id'] == rid and acknowledged['canonical_output_ids'] == expected
                else:
                    assert len(row['received']) == 64
                    assert set(row['native_receive_completions']) == {r['tensor_id'] for r in row['received']}
                    assert [produced['first_token']]+outputs[-1]['output_ids'] == expected
                    if payload:
                        from h200_native_pd_contract import validate_layers
                        total_bytes = phase_contract(prompt,expected[0])['analytical_requested_KV_tensor_bytes']
                        row['payload_gate'] = validate_layers(produced['sent'],row['injected'],rid,total_bytes)
                    dump(shared/(marker+'_ack.json'), {'request_id':rid,'canonical_output_ids':expected,
                         'completion_unix_ns':time.time_ns()})
                row.update(status='functional_phase_passed',completed_unix_ns=time.time_ns())
                hook.active = None; dump(args.output/'run.json',ledger)
        ledger.update(status='native_model_functional_captured_pending_pair_review',
            model_KV_injected_verified=args.role=='consumer')
        dump(args.output/'run.json',ledger)
        dump(shared/(args.role+'_model_complete.json'), {'status':ledger['status']})
        assert wait_json(shared/(peer_role+'_model_complete.json'),deadline)['status'] == ledger['status']
    except BaseException as error:
        ledger.update(status='failed',error=str(error),traceback=traceback.format_exc())
        dump(args.output/'run.json',ledger)
        raise
    finally:
        if hook is not None: hook.close()
        if llm is not None:
            from controlled_vllm_experiments import cleanup_owned_engine
            cleanup_owned_engine(llm.llm_engine,args.output)
        finished.set()
        if nv is not None: nv.nvmlShutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-config',type=Path,required=True)
    parser.add_argument('--role',choices=('producer','consumer'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    try:
        main(args)
    except BaseException as error:
        from software_decode_experiments import dump
        dump(args.output/'failure.json', {'error':str(error),'traceback':traceback.format_exc()})
        raise
