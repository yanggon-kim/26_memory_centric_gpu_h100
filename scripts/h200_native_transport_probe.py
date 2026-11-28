#!/usr/bin/env python3
"""Known-data native P2pNccl probe: not model KV, serving or physical traffic.

Run both roles through owned H200Session supervisors. A CPU watchdog bounds
native blocking calls; the original transport engine's unbounded close method
is deliberately not invoked. No installed vLLM source is changed.
"""
import argparse
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

ENGINE_SHA='bcbb73c586bc82d51c84e89e6b0dca9dc2354fa7a48856c1ae3a9f7dc45fd17a'
SHAPE=(2,8,16,8,128)
LAYER_BYTES=524288


def validate_pair(pair,session,role,uuid,job,hostname):
    assert role in ('producer','consumer') and pair['job_id']==session['slurm_job_id']==job
    assert pair['hostname']==session['expected_hostname']==hostname
    assert pair['roles'][role]['gpu_uuid']==session['gpu_uuid']==uuid
    assert len({v['gpu_uuid'] for v in pair['roles'].values()})==2
    assert set(pair['roles'])=={'producer','consumer'}
    ports=[v['port'] for v in pair['roles'].values()]
    assert len(set(ports))==2 and all(isinstance(p,int) and 0<p<65536 for p in ports)
    assert pair['host']=='127.0.0.1' and pair['synthetic_payload'] is True
    request_id(pair['request_prefix'],pair['host'],ports[0],pair['host'],ports[1])


def wait_json(path,deadline):
    while not path.exists():
        assert time.time()<deadline,'bounded peer-file wait expired: '+str(path)
        time.sleep(.01)
    return json.loads(path.read_text())


def all_native_sends_completed(engine,rid,identifiers):
    return set(identifiers).issubset(engine.send_request_id_to_tensor_ids.get(rid,set()))


def main(args):
    from software_decode_experiments import dump
    pair=json.loads(args.pair_config.read_text())
    session_path=Path(os.environ['H200_SESSION_CONFIG']);session=json.loads(session_path.read_text())
    validate_pair(pair,session,args.role,args.gpu_uuid,os.environ.get('SLURM_JOB_ID'),socket.gethostname())
    args.output.mkdir(parents=True,exist_ok=False)
    shared=Path(pair['shared_directory']);assert shared.is_dir()
    deadline=min(time.time()+480,float(pair['deadline_unix_s']),
                 dt.datetime.fromisoformat(session['launch_cutoff']).timestamp())
    assert deadline>time.time(),'no native component role budget remains'
    finished=threading.Event()
    ledger=None
    def watchdog():
        if not finished.wait(max(0,deadline-time.time())):
            snapshot_error=None
            try:
                if ledger is not None:
                    dump(args.output/'run.json',dict(ledger,status='failed_deadline'))
            except BaseException as error:
                snapshot_error=type(error).__name__+': '+str(error)
            try:
                dump(args.output/'deadline.json',{'status':'failed_deadline','deadline_unix_s':deadline,
                    'live_ledger_snapshot_error':snapshot_error,
                    'scope':'owned probe worker exit124; native call did not return within its bound'})
            finally:
                os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      VLLM_HOST_IP='127.0.0.1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
                      VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
    ledger={'status':'running','classification':'measured-real-GPU-known-data-transport',
        'role':args.role,'gpu_uuid':args.gpu_uuid,'pair_config_sha256':hashlib.sha256(args.pair_config.read_bytes()).hexdigest(),
        'session_sha256':hashlib.sha256(session_path.read_bytes()).hexdigest(),
        'numerical_source_identity':session['frozen_source_identity'],'shape':list(SHAPE),
        'dtype':'torch.bfloat16','packed_layer_tensors':64,'requested_tensor_bytes_per_trial':32*1024**2,
        'seed':20261013,'source_fill':'deterministic full blocks: float((trial*64+layer)%128); no RNG-dependent tensor values',
        'synthetic_payload':True,'model_KV_generated':False,'model_KV_injected':False,
        'physical_HBM_or_NVLink_bytes':None,'qualified_performance':False,'trials':[],
        'scope':'Synthetic full-block BF16 fills through the installed native transport only. Not a real model KV handoff, instruction trace, serving result, physical link/HBM byte count or calibrated bandwidth.'}
    dump(args.output/'run.json',ledger)
    nv=None
    try:
        import pynvml as nv
        from gpu_target import select,verify_cuda,describe
        nv.nvmlInit();handle,_=select(nv,args.gpu_uuid)
        import torch
        verify_cuda(torch,args.gpu_uuid)
        torch.manual_seed(20261013)
        ledger['hardware']=describe(nv,handle,torch)
        from vllm.config import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p import p2p_nccl_engine
        actual=hashlib.sha256(Path(p2p_nccl_engine.__file__).read_bytes()).hexdigest()
        assert actual==ENGINE_SHA,'native transport requires a new source review'
        role=pair['roles'][args.role]
        opposite='consumer' if args.role=='producer' else 'producer'
        peer=pair['roles'][opposite]
        cfg=KVTransferConfig(kv_connector='P2pNcclConnector',kv_role='kv_'+args.role,
            kv_rank=0 if args.role=='producer' else 1,kv_parallel_size=2,kv_port=role['port'],
            kv_buffer_device='cuda',kv_buffer_size=128*1024**2,
            kv_connector_extra_config={'send_type':'PUT_ASYNC','mem_pool_size_gb':.5})
        engine=p2p_nccl_engine.P2pNcclEngine(local_rank=0,config=cfg,hostname=pair['host'])
        assert engine.zmq_address==f"{pair['host']}:{role['port']}"
        remote=f"{pair['host']}:{peer['port']}"
        ledger.update(native_engine_sha256=actual,transport={'send_type':engine.send_type,
            'GPU_staging_threshold_bytes':engine.buffer_size_threshold,
            'host_pool_bytes':512*1024**2,'own_address':engine.zmq_address,'peer_address':remote,
            'host_pool_scope':'explicit small component-test pool; not the default deployment pool'},
            versions={'torch':torch.__version__,'cuda':torch.version.cuda})
        dump(shared/f'{args.role}_ready.json',{'gpu_uuid':args.gpu_uuid,'job_id':pair['job_id'],
            'pid':os.getpid(),'native_engine_sha256':actual})
        ready=wait_json(shared/f'{opposite}_ready.json',deadline)
        assert ready['gpu_uuid']==peer['gpu_uuid'] and ready['job_id']==pair['job_id']
        assert ready['native_engine_sha256']==actual
        layer_names=[f'model.layers.{i}.self_attn.attn' for i in range(64)]
        for trial in range(6):
            prefix=f"{pair['request_prefix']}_trial{trial}"
            rid=request_id(prefix,pair['host'],pair['roles']['producer']['port'],
                           pair['host'],pair['roles']['consumer']['port'])
            values=[float((trial*64+i)%128) for i in range(64)]
            identifiers=[rid+'#'+name for name in layer_names]
            row={'trial':trial,'warmup':trial<3,'request_id':rid,'status':'running','layers':[]}
            ledger['trials'].append(row);dump(args.output/'run.json',ledger)
            start=time.time_ns()
            if args.role=='producer':
                tensors=[torch.full(SHAPE,value,dtype=torch.bfloat16,device='cuda') for value in values]
                torch.cuda.synchronize()
                for i,tensor in enumerate(tensors):
                    assert tensor.is_contiguous() and tensor.numel()*tensor.element_size()==LAYER_BYTES
                    digest=hashlib.sha256(tensor.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                    row['layers'].append({'layer_index':i,'tensor_id':identifiers[i],'shape':list(tensor.shape),
                        'dtype':str(tensor.dtype),'nbytes':LAYER_BYTES,'sha256':digest})
                # Source buffers are complete before the transport-only probe.
                # This does not validate native model-hook stream ordering.
                dump(shared/f'trial{trial}_source.json',{'request_id':rid,'layers':row['layers']})
                for identifier,tensor in zip(identifiers,tensors):
                    assert engine.send_tensor(identifier,tensor,remote) is True
                # A drained queue alone does not prove its popped last item
                # has completed. Require the engine's post-send completion IDs.
                while not all_native_sends_completed(engine,rid,identifiers):
                    assert time.time()<deadline,'native send completion deadline reached'
                    time.sleep(.001)
                row['all_native_send_completions_observed']=True
                peer_result=wait_json(shared/f'trial{trial}_received.json',deadline)
                assert peer_result['status']=='all_known_tensors_received_and_verified'
                assert peer_result['request_id']==rid
            else:
                source=wait_json(shared/f'trial{trial}_source.json',deadline)
                assert source['request_id']==rid and len(source['layers'])==64
                for i,identifier in enumerate(identifiers):
                    tensor=engine.recv_tensor(identifier,remote)
                    assert tensor is not None and tuple(tensor.shape)==SHAPE and tensor.dtype==torch.bfloat16
                    assert tensor.device==engine.device and tensor.is_cuda
                    assert tensor.is_contiguous()
                    host=tensor.cpu()
                    digest=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
                    assert torch.equal(host,torch.full(SHAPE,values[i],dtype=torch.bfloat16))
                    assert digest==source['layers'][i]['sha256']
                    row['layers'].append({**source['layers'][i],'received_sha256':digest,'known_values_verified':True})
                while not set(identifiers).issubset(engine.recv_request_id_to_tensor_ids.get(rid,set())):
                    assert time.time()<deadline,'native receive completion deadline reached'
                    time.sleep(.001)
                dump(shared/f'trial{trial}_received.json',{'status':'all_known_tensors_received_and_verified',
                    'request_id':rid,'layers':row['layers']})
            row.update(status='known_data_transport_passed',wall_start_unix_ns=start,wall_end_unix_ns=time.time_ns(),
                timing_scope='includes source fill/hash, peer waits and receiver checks; not clean link latency')
            engine.get_finished({rid},{name:None for name in layer_names})
            dump(args.output/'run.json',ledger)
        ledger['status']='known_data_transport_probe_passed'
        dump(args.output/'run.json',ledger)
        dump(shared/f'{args.role}_complete.json',{'status':ledger['status'],'gpu_uuid':args.gpu_uuid})
        wait_json(shared/f'{opposite}_complete.json',deadline)
        # All six sends/receives and checks are complete. Native daemon threads
        # remain process-lifetime resources; no unbounded transport close call.
    except BaseException as error:
        ledger.update(status='failed',error=str(error),traceback=traceback.format_exc())
        dump(args.output/'run.json',ledger)
        raise
    finally:
        finished.set()
        if nv is not None:nv.nvmlShutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-config',type=Path,required=True)
    parser.add_argument('--role',choices=('producer','consumer'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
