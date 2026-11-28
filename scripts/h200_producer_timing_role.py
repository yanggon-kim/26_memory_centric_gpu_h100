#!/usr/bin/env python3
"""Prospective matched stock/real-transfer producer host-timing role.

Strict references and a same-allocation, same-context functional certificate
are mandatory. Timed requests have no scheduler/connector observer, payload
hashes, NVTX, CUDA events or added GPU synchronization. Warmups alone retain
scheduler query proof. File IPC conveys the *actual* first producer token
before waiting for the final native send; it is not HTTP or deployed TTFT.
"""
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import traceback
from h200_native_pd_contract import request_id
from h200_native_model_pilot import validate_scope,validate_queries,CONNECTOR_SHA,ENGINE_SHA

ACCOUNTING_SOURCES={
    'outputs.py':'991c0960770a619066ccbf469cc08ce1e9523e7fe19e3a116c4cd13c1e1750a0',
    'v1/request.py':'462c17cfde4308f0eb35eb0b2afa6b4ba3e628caa481dee9f476297bec373f1e',
    'v1/core/sched/scheduler.py':'41ff2e524c90d9aa72b72cd77492eb62ee2a729a773bd8233e970f39abbb5983',
    'v1/engine/output_processor.py':'464c3cda347512cf1090f8e6471d0d3f0f65e286eb9a7b97be52efbc1ffd572a',
    'v1/metrics/stats.py':'08eb6bb969ad47e4207ba5568beaec35bdb09192b6db046f046dab4892ac1bf3'}


def check_step_accounting(row,role,context):
    assert row['engine_request_preemptions']==0
    outputs=row['outputs'];assert len(outputs)==(63 if role=='consumer' else 1)
    previous_cached=0
    for index,output in enumerate(outputs):
        cached=output['num_cached_tokens']
        assert type(cached) is int and cached==(context if role=='consumer' else 0)
        before=output['computed_tokens_before_step'];after=output['computed_tokens_after_step']
        assert type(before) is int and type(after) is int
        assert before==(0 if index==0 else context+index)
        actual_queries=after-before-(cached-previous_cached)
        assert actual_queries==(1 if role=='consumer' else context)
        assert after==(context+index+1 if role=='consumer' else context)
        previous_cached=cached
    return {'scheduled_input_queries':63 if role=='consumer' else context,
        'original_prompt_queries':0 if role=='consumer' else context,
        'first_augmented_prompt_query':1 if role=='consumer' else None,
        'following_post_prompt_queries':62 if role=='consumer' else 0,
        'external_prefix_tokens':context if role=='consumer' else 0,
        'scope':'Pinned synchronous non-speculative request counters and output cache hits; local prefix caching disabled, no preemption. Scheduled logical queries, not physical padded GPU instruction counts.'}


def descriptor_scope(pair,session,role,uuid,job,hostname):
    assert pair['protocol']=='matched_native_producer_host_timing_v1'
    assert role in ('stock','producer','consumer')
    proxy=dict(pair,protocol='native_model_KV_long_context_functional_v2')
    validate_scope(proxy,session,'producer' if role=='stock' else role,uuid,job,hostname)
    assert type(pair['acquisition']) is int and pair['acquisition'] in (0,1,2)
    assert pair['mode'] in ('pilot','comparison')
    assert pair['warmup_trials']==3
    assert pair['timed_trials']==(3 if pair['mode']=='pilot' else 10)


def checked_outputs(actual,expected,role):
    wanted=expected[:1] if role in ('stock','producer') else expected[1:len(actual)+1]
    assert actual==wanted,'exact reference output differs'


def energy_readout(nv,handle):
    """Raw NVML board counter, with its host query bracket; no estimate."""
    result={'query_start_unix_ns':time.time_ns(),'query_end_unix_ns':None,
        'energy_mj':None,'status':'unavailable','error':None}
    try:
        value=nv.nvmlDeviceGetTotalEnergyConsumption(handle)
        assert type(value) is int and value>=0,'invalid cumulative energy counter'
        result.update(energy_mj=value,status='available')
    except (AttributeError,nv.NVMLError,AssertionError) as error:
        result['error']=type(error).__name__+': '+str(error)
    finally:result['query_end_unix_ns']=time.time_ns()
    return result


def main(args):
    from software_decode_experiments import dump
    from h200_priority_campaign import load,sha,valid_tokens,strict_reference_record,write,limitation
    from h200_native_transport_probe import wait_json,all_native_sends_completed
    session_path=Path(os.environ['H200_SESSION_CONFIG']);session=load(session_path);pair=load(args.pair_config)
    descriptor_scope(pair,session,args.role,args.gpu_uuid,os.environ.get('SLURM_JOB_ID'),socket.gethostname())
    source_path=Path(pair['reference_run']);assert sha(source_path)==pair['reference_run_sha256']
    reference=load(source_path);assert valid_tokens(reference)
    hf_path=Path(pair['strict_HF_reference']);assert sha(hf_path)==pair['strict_HF_reference_sha256']
    assert strict_reference_record(load(hf_path),sha(source_path))
    case=reference['cases'][0]
    assert (case['batch_size'],case['context_length'],case['generated_tokens'])==(1,pair['context'],64)
    prompt=case['prompt_ids'][0];expected=case['trials'][0]['output_ids'][0]
    certificate_path=Path(pair['functional_CPU_review']);assert sha(certificate_path)==pair['functional_CPU_review_sha256']
    certificate=load(certificate_path)
    assert certificate['status']=='native_model_functional_review_passed'
    # A certificate is a prerequisite, not a substitute for this role's owned
    # runtime/source, exact outputs and warmup query proof.
    functional_pair=load(certificate_path.parents[1]/'pair.json')
    assert functional_pair['job_id']==pair['job_id'] and functional_pair['context']==pair['context']
    assert functional_pair['reference_run_sha256']==pair['reference_run_sha256']
    assert {r:v['gpu_uuid'] for r,v in functional_pair['roles'].items()}=={r:v['gpu_uuid'] for r,v in pair['roles'].items()}
    dependencies=None
    for certified_role in certificate['roles']:
        for key in ('session','run','process','monitor'):
            assert sha(Path(certified_role[key]))==certified_role[key+'_sha256']
        certified_session=load(Path(certified_role['session']))
        assert certified_session['frozen_source_identity']==session['frozen_source_identity']
        certified_run=load(Path(certified_role['run']))
        if dependencies is None:dependencies=certified_run['optional_dependencies']
        assert certified_run['optional_dependencies']==dependencies
    assert reference['config']['seed']==20261013 and reference['config']['model_id']=='Qwen/Qwen2.5-32B-Instruct'
    assert reference['config']['model_revision']=='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
    assert not reference['config'].get('decode_conditioning')
    settings=reference['engine_settings']
    assert settings['tensor_parallel_size']==1 and settings['dtype']=='bfloat16'
    assert not settings['enable_prefix_caching'] and settings['enforce_eager'] is False and settings['async_scheduling'] is False
    import h200_session
    allocation=h200_session.verify_allocation();assert allocation is not None
    deadline=min(time.time()+660,pair['deadline_unix_s'],allocation['effective_launch_cutoff_unix_s'],
        dt.datetime.fromisoformat(session['launch_cutoff']).timestamp())
    assert deadline>time.time()
    args.output.mkdir(parents=True,exist_ok=False)
    ledger={'status':'running','classification':'real GPU host-timing pilot' if pair['mode']=='pilot' else 'real GPU host-timing acquisition pending independent review',
        'protocol':pair['protocol'],'role':args.role,'gpu_uuid':args.gpu_uuid,'acquisition':pair['acquisition'],
        'mode':pair['mode'],'source_identity':session['frozen_source_identity'],'pair_sha256':sha(args.pair_config),
        'reference_run_sha256':sha(source_path),'functional_CPU_review_sha256':sha(certificate_path),
        'engine_settings_without_KV_object':settings,'allocation_verification':allocation,'trials':[],
        'clean_timing_qualified':False,'serving_qualified':False,'primary_J_per_token':None,
        'scope':'Producer submit-to-first-step-return host time is separate from all-native-send readiness, file IPC, consumer host step times and end-to-end serving goodput. No HTTP/SLO or clean GPU-event latency is measured. Warmup query proof is instrumented; timed rows have no scheduler/connector observer, payload proof or added GPU synchronization. Consumer per-step CPU recording/output checks occur between engine calls, so its whole-request elapsed includes that overhead.'}
    dump(args.output/'run.json',ledger)
    finished=threading.Event()
    def watchdog():
        if not finished.wait(max(0,deadline-time.time())):
            try:dump(args.output/'run.json',dict(ledger,status='failed_deadline'))
            finally:os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        VLLM_HOST_IP='127.0.0.1',VLLM_DISABLE_REQUEST_ID_RANDOMIZATION='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
    llm=None;nv=None
    try:
        import pynvml as nv
        nv.nvmlInit()
        from gpu_target import select,verify_cuda,describe
        handle,_=select(nv,args.gpu_uuid)
        import torch
        verify_cuda(torch,args.gpu_uuid)
        import vllm
        package=Path(vllm.__file__).parent
        accounting={name:sha(package/name) for name in ACCOUNTING_SOURCES}
        assert accounting==ACCOUNTING_SOURCES
        ledger['request_accounting_source_sha256']=accounting
        from vllm.config import KVTransferConfig
        from vllm.sampling_params import RequestOutputKind
        from h200_scheduler_observer import Observer
        import msgpack
        from importlib.metadata import version
        optional={'msgpack':version('msgpack'),'module_path':msgpack.__file__,
            'package_file_sha256':{p.name:sha(p) for p in sorted(Path(msgpack.__file__).parent.iterdir()) if p.is_file()}}
        assert optional==dependencies and optional['msgpack']=='1.1.1'
        ledger['optional_dependencies']=optional
        ledger.update(hardware=describe(nv,handle,torch),versions={'torch':torch.__version__,'vllm':vllm.__version__,'cuda_runtime':torch.version.cuda})
        assert ledger['versions']==reference['versions']
        engine_settings=dict(settings);native=None
        if args.role!='stock':
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            from vllm.distributed.kv_transfer.kv_connector.v1.p2p import p2p_nccl_connector,p2p_nccl_engine
            assert sha(Path(p2p_nccl_connector.__file__))==CONNECTOR_SHA
            assert sha(Path(p2p_nccl_engine.__file__))==ENGINE_SHA
            cfg=KVTransferConfig(kv_connector='P2pNcclConnector',kv_role='kv_'+args.role,
                kv_rank=0 if args.role=='producer' else 1,kv_parallel_size=2,kv_port=pair['roles'][args.role]['port'])
            engine_settings['kv_transfer_config']=cfg;ledger['KV_transfer_config']=dataclasses.asdict(cfg)
        dump(args.output/'run.json',ledger)
        llm=vllm.LLM(**engine_settings);engine=llm.llm_engine
        assert type(engine.engine_core).__name__=='InprocClient'
        core=engine.engine_core.engine_core
        assert core.batch_queue is None and not core.async_scheduling and not core.use_spec_decode
        scheduler=core.scheduler
        shared=Path(pair['shared_directory']);assert shared.is_dir()
        if args.role!='stock':
            connector=get_kv_transfer_group();native=connector.p2p_nccl_engine
            assert connector.is_producer==(args.role=='producer') and native.send_type=='PUT_ASYNC'
            assert native.zmq_address==f"127.0.0.1:{pair['roles'][args.role]['port']}"
            ledger['effective_native_transport']={'send_type':native.send_type,'GPU_staging_threshold_bytes':native.buffer_size_threshold,
                'pinned_host_pool_bytes':native.pool.max_block_size,'configured_nccl_channels':native.nccl_num_channels,
                'native_engine_sha256':ENGINE_SHA,'connector_sha256':CONNECTOR_SHA}
            dump(shared/(args.role+'_timing_ready.json'),{'job_id':pair['job_id'],'gpu_uuid':args.gpu_uuid})
            peer='consumer' if args.role=='producer' else 'producer'
            ready=wait_json(shared/(peer+'_timing_ready.json'),deadline)
            assert ready['job_id']==pair['job_id'] and ready['gpu_uuid']==pair['roles'][peer]['gpu_uuid']
        total=pair['warmup_trials']+pair['timed_trials']
        energy_notice_written=False
        def energy_notice(raw):
            nonlocal energy_notice_written
            reads=[raw[k] for k in ('before','after') if raw[k] is not None]
            bad=any(r['status']!='available' for r in reads)
            reset=len(reads)==2 and not bad and reads[1]['energy_mj']<reads[0]['energy_mj']
            if (bad or reset) and not energy_notice_written:
                write(session_path.parent/'results/energy_counter_limitation.json',raw)
                limitation(session_path.parent,'producer_raw_energy_counter','Cumulative energy counter unavailable or reset; primary J/token remains NA',
                    'results/energy_counter_limitation.json; runs/timing/run.json',
                    'Record raw counter values/errors with host query brackets outside the primary producer window; sampled power is independently retained',
                    'Qualify declared aligned energy windows, cadence and counter/power agreement before any primary energy claim; latency gates remain independent')
                energy_notice_written=True
        for trial in range(total):
            marker='a'+str(pair['acquisition'])+'_t'+str(trial)
            rid=request_id(pair['request_prefix']+'_'+marker,pair['host'],pair['roles']['producer']['port'],pair['host'],pair['roles']['consumer']['port'])
            row={'trial':trial,'warmup':trial<3,'request_id':rid,'status':'running','outputs':[],
                'host_step_intervals':[],'warmup_scheduler_rows':None,'observer_present':trial<3,
                'producer_first_step_return_unix_ns':None,'all_native_sends_observed_unix_ns':None,'engine_request_preemptions':None,
                'raw_energy_counter':{'before':energy_readout(nv,handle),'after':None,
                    'counter_unit':'mJ','primary_J_per_token':None,
                    'scope':'Per-role request lifecycle including handoff/peer waiting, not the primary producer timing window or an aligned all-board deployment window. Power samples are retained by the independent owned monitor; counter cadence/window/agreement gates are not established by these two readouts.'}}
            ledger['trials'].append(row);dump(args.output/'run.json',ledger)
            energy_notice(row['raw_energy_counter'])
            if args.role=='consumer':
                first=wait_json(shared/(marker+'_first.json'),deadline)
                assert first['request_id']==rid and first['first_token']==expected[0]
                row['first_token_file_read_unix_ns']=time.time_ns()
                tokens=prompt+[first['first_token']];count=63
            else:tokens=prompt;count=1
            sampling=vllm.SamplingParams(temperature=0,repetition_penalty=1.0,ignore_eos=True,
                detokenize=False,max_tokens=count,seed=20261013,output_kind=RequestOutputKind.CUMULATIVE)
            observer=Observer(engine) if trial<3 else None
            try:
                row['hardware_before']=describe(nv,handle,torch)
                row['submit_unix_ns']=time.time_ns();began=time.perf_counter_ns()
                assert engine.add_request(rid,{'prompt_token_ids':tokens},sampling)==rid
                request_ref=scheduler.requests[rid]
                while engine.has_unfinished_requests():
                    assert time.time()<deadline
                    computed_before=request_ref.num_computed_tokens
                    before_unix=time.time_ns();before=time.perf_counter_ns()
                    batch=engine.step()
                    after=time.perf_counter_ns();after_unix=time.time_ns()
                    row['host_step_intervals'].append({'start_unix_ns':before_unix,'end_unix_ns':after_unix,'elapsed_ns':after-before})
                    assert len(batch)==1 and batch[0].request_id==rid
                    result=batch[0];actual=list(result.outputs[0].token_ids)
                    # Preserve actual output even when the exact gate rejects it.
                    row['outputs'].append({'observed_unix_ns':after_unix,'output_ids':actual,
                        'num_cached_tokens':result.num_cached_tokens,'computed_tokens_before_step':computed_before,
                        'computed_tokens_after_step':request_ref.num_computed_tokens})
                    checked_outputs(actual,expected,args.role)
                row['submit_to_final_step_return_ns']=after-began
                row['final_step_return_unix_ns']=after_unix
                row['hardware_after']=describe(nv,handle,torch)
                row['engine_request_preemptions']=request_ref.num_preemptions
                row['query_accounting']=check_step_accounting(row,args.role,pair['context'])
                assert result.finished and result.outputs[0].finish_reason=='length' and len(actual)==count
                assert len(row['host_step_intervals'])==(63 if args.role=='consumer' else 1)
                if observer is not None:
                    row['warmup_scheduler_rows']=observer.rows
                    validate_queries(observer.rows,'producer' if args.role=='stock' else args.role,rid,pair['context'])
            finally:
                if observer is not None:observer.close()
            if args.role=='producer':
                row['producer_first_step_return_unix_ns']=after_unix
                # This is outside the producer primary timing window. Unlike
                # the functional proof, publish p0 before the final send wait.
                dump(shared/(marker+'_first.json'),{'request_id':rid,'first_token':actual[0],
                    'producer_step_return_unix_ns':after_unix})
                layer_names=engine.vllm_config.compilation_config.static_forward_context
                identifiers=[rid+'#'+name for name in layer_names]
                assert len(identifiers)==64
                while not all_native_sends_completed(native,rid,identifiers):
                    assert time.time()<deadline;time.sleep(.001)
                row['all_native_sends_observed_unix_ns']=time.time_ns()
                row['native_sent_tensor_count']=len(identifiers)
                ack=wait_json(shared/(marker+'_ack.json'),deadline)
                assert ack['request_id']==rid and ack['canonical_output_ids']==expected
                row['consumer_completion_unix_ns']=ack['completion_unix_ns']
            elif args.role=='consumer':
                assert len(row['outputs'])==63 and expected[:1]+actual==expected
                dump(shared/(marker+'_ack.json'),{'request_id':rid,'canonical_output_ids':expected,
                    'completion_unix_ns':after_unix})
            row['raw_energy_counter']['after']=energy_readout(nv,handle)
            energy_notice(row['raw_energy_counter'])
            row['status']='exact_output_passed';row['completed_unix_ns']=time.time_ns()
            dump(args.output/'run.json',ledger)
        ledger['status']='complete';dump(args.output/'run.json',ledger)
        if args.role!='stock':
            dump(shared/(args.role+'_timing_complete.json'),{'status':'complete'})
            assert wait_json(shared/(peer+'_timing_complete.json'),deadline)['status']=='complete'
    except BaseException as error:
        ledger.update(status='failed',error=str(error),traceback=traceback.format_exc())
        dump(args.output/'run.json',ledger);raise
    finally:
        if llm is not None:
            from controlled_vllm_experiments import cleanup_owned_engine
            cleanup_owned_engine(llm.llm_engine,args.output)
        finished.set()
        if nv is not None:nv.nvmlShutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-config',type=Path,required=True)
    parser.add_argument('--role',choices=('stock','producer','consumer'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
