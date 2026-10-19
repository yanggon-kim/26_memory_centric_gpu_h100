#!/usr/bin/env python3
"""Owned disjoint green-context controls; never substitute unrelated GPU traffic."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback


def main(a):
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    import torch
    import triton
    from cuda.core import Device,ContextOptions,SMResourceOptions
    from gpu_target import verify_cuda
    from shoreline_green import stream_proof,retain_for_process
    from shoreline_kernels import arithmetic,transfer,condition
    from software_decode_experiments import dump
    verify_cuda(torch,a.gpu_uuid)
    device=Device(0);device.set_current();resource=device.resources.sm
    import math
    count=max(resource.min_partition_size,math.ceil(resource.sm_count*.75/resource.coscheduled_alignment)*resource.coscheduled_alignment)
    groups,remainder=resource.split(SMResourceOptions(count=count))
    assert len(groups)==1 and remainder.sm_count>=resource.min_partition_size,'usable disjoint remainder unavailable'
    dc=device.create_context(ContextOptions(resources=[groups[0]]))
    cc=device.create_context(ContextOptions(resources=[remainder]))
    ds,cs=dc.create_stream(),cc.create_stream()
    # Framework allocator/event caches outlive this function. Retain both CUDA
    # contexts and their streams until process exit, as in the reviewed J5 fix.
    retain_for_process(dc,ds)
    retain_for_process(cc,cs)
    decoder=torch.cuda.ExternalStream(int(ds.handle),device=0)
    competitor=torch.cuda.ExternalStream(int(cs.handle),device=0)
    proof={'decoder':stream_proof(decoder),'competitor':stream_proof(competitor),
           'partition_proof':'groups and remainder returned from one SMResource.split call',
           'parent_sm_count':resource.sm_count,'requested_decoder_sm_count':count,
           'returned_group_sm_count':groups[0].sm_count,'returned_remainder_sm_count':remainder.sm_count,
           'disjoint_api_resources':groups[0].handle!=remainder.handle}
    proof['lifetime_protocol']='disjoint_lifetime_v2; process-lifetime retained contexts and streams'
    assert proof['decoder']['green_context_id']!=proof['competitor']['green_context_id']
    assert proof['decoder']['actual_sm_count']+proof['competitor']['actual_sm_count']<=resource.sm_count
    # Probe actual SM IDs on both contexts, in addition to resource API evidence.
    observed=[]
    for stream,allocation in ((decoder,proof['decoder']),(competitor,proof['competitor'])):
        blocks=allocation['actual_sm_count']*4;n=blocks*256
        with torch.cuda.stream(stream):
            x=torch.ones(n,device='cuda');y=torch.empty_like(x);ids=torch.empty(blocks,device='cuda',dtype=torch.int32)
            arithmetic[(blocks,)](x,y,ids,n,1024,False,256);stream.synchronize()
            assert torch.isfinite(y).all().item()
            observed.append(sorted(set(ids.cpu().tolist())))
    proof['observed_smids']=observed
    assert not set(observed[0])&set(observed[1]),'observed SM sets overlap'
    dump(a.output/'disjoint_allocation.json',proof)
    if a.probe:
        print(json.dumps({'status':'probe passed','allocation':proof}));return
    assert a.config
    # Identical private allocations are made in every treatment before model/KV allocation.
    n=256*1024**2//4
    private=torch.ones(n,device='cuda');private_output=torch.empty_like(private)
    ids=torch.empty(triton.cdiv(n,4096),device='cuda',dtype=torch.int32)
    cn=remainder.sm_count*256
    state=torch.ones(cn,device='cuda');state_out=torch.empty_like(state)
    treatment=a.treatment;errors=[];all_samples=[]
    import controlled_vllm_experiments as baseline
    original=baseline.run_trial

    def trial(engine,prompts,sampling,prefix,torch,*args,**kwargs):
        stop=threading.Event();ready=threading.Event();samples=[]
        duty=0 if treatment=='idle' else .5 if treatment.endswith('50') else 1.0
        def competing():
            try:
                with torch.cuda.stream(competitor):
                    ready.set()
                    while not stop.is_set():
                        if duty==0:
                            stop.wait(.001);continue
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        t0=time.time_ns();begin.record()
                        if treatment.startswith('memory'):
                            transfer[(triton.cdiv(n,4096),)](private,private_output,ids,n,0,4096)
                        else:condition[(remainder.sm_count,)](state,state_out,cn,200000,256)
                        end.record();competitor.synchronize();elapsed=begin.elapsed_time(end)
                        samples.append({'trial':prefix,'start_unix_ns':t0,'end_unix_ns':time.time_ns(),'gpu_active_ms':elapsed})
                        if duty<1:stop.wait(elapsed/1000*(1/duty-1))
            except BaseException as error:
                errors.append(str(error));ready.set();stop.set()
        thread=threading.Thread(target=competing,daemon=True)
        step=engine.step;started=False
        def observed():
            nonlocal started
            # Start at the first decode step, after the verified prefill barrier/global synchronization.
            if not started and hasattr(observed,'prefill_done'):
                started=True;thread.start();assert ready.wait(10)
            result=step();observed.prefill_done=True
            if result and all(o.finished for o in result):
                stop.set()
                if started:thread.join(timeout=10)
                assert not thread.is_alive(),'owned competitor did not stop'
                assert not errors,errors
            return result
        engine.step=observed
        try:
            result=original(engine,prompts,sampling,prefix,torch,*args,**kwargs)
            result['competition']={'treatment':treatment,'requested_duty':duty,'samples':samples,
                'scope':'includes owned thread start/stop costs; fixed split and private buffers across treatments',
                'active_gpu_ms':sum(s['gpu_active_ms'] for s in samples),
                'observed_interval_ms':(samples[-1]['end_unix_ns']-samples[0]['start_unix_ns'])/1e6 if samples else None}
            all_samples.extend(samples)
            return result
        finally:
            stop.set()
            if started:thread.join(timeout=10)
            engine.step=step
    baseline.run_trial=trial
    with torch.cuda.stream(decoder):
        if a.external_profiler:
            import shoreline_decode
            shoreline_decode.main(a)
        else:baseline.main(a)
    dump(a.output/'competition_ledger.json',{'treatment':treatment,'allocation':proof,'samples':all_samples,
        'physical_traffic':None,'reason':'concurrent counter replay requires independent validation',
        'actual_concurrency':'requires NSYS evidence of both context IDs during selected decode'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',action='store_true');p.add_argument('--config')
    p.add_argument('--treatment',choices=['idle','memory50','memory100','arithmetic50','arithmetic100'],default='idle')
    p.add_argument('--mode',default='timing');p.add_argument('--external-profiler',action='store_true')
    p.add_argument('--collect-worker-metadata',action='store_true');p.add_argument('--batch',type=int)
    p.add_argument('--context',type=int);p.add_argument('--generated-tokens',type=int,default=64)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except BaseException as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
