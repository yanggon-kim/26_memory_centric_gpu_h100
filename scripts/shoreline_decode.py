#!/usr/bin/env python3
"""Full 64-token decode with a bounded ticks-4/5/6 profiler interval."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main(a):
    if getattr(a,'ordinary_control',False):a.external_profiler=False
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import controlled_vllm_experiments as baseline
    import h200_worker_metadata
    import shoreline_worker
    h200_worker_metadata.install_layer_ranges=shoreline_worker.install_ranges
    from software_decode_experiments import dump
    original=baseline.run_trial
    original_cleanup=baseline.cleanup_owned_engine
    def cleanup(engine,output):
        if shoreline_worker.ACTIVE_WORKERS:
            dump(output/'shoreline_worker_metadata.json',
                 [shoreline_worker.read_metadata(worker) for worker in shoreline_worker.ACTIVE_WORKERS])
        original_cleanup(engine,output)
    baseline.cleanup_owned_engine=cleanup
    if getattr(a,'pressure',None) or getattr(a,'layer_events',False) or getattr(a,'empty_layer_events',False):
        if getattr(a,'pressure',None):os.environ['SHORELINE_PRESSURE']=a.pressure
        import vllm
        original_llm=vllm.LLM
        class ObservedLLM(original_llm):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                assert kwargs['enforce_eager'],'layer/treatment callbacks require eager execution'
                self.collective_rpc(shoreline_worker.install_ranges)
                if getattr(a,'layer_events',False) or getattr(a,'empty_layer_events',False):
                    self.collective_rpc(shoreline_worker.enable_events,args=('empty' if getattr(a,'empty_layer_events',False) else 'complete',))
        vllm.LLM=ObservedLLM
        # Our factory installs once even when the ordinary runner also requests ranges.
        h200_worker_metadata.install_layer_ranges=lambda worker:{'ranges':'installed by observed factory'}

    def trial(engine,prompts,sampling,prefix,torch,output_dir,sensor=None,
              profile_capture=False,observer=None,hardware_sampler=None):
        if not profile_capture and not getattr(a,'pressure',None) and not getattr(a,'layer_events',False):
            return original(engine,prompts,sampling,prefix,torch,output_dir,sensor,
                            False,observer,hardware_sampler)
        original_step=engine.step
        calls=0;ledger=[];capturing=False
        range_start=torch.cuda.Event(enable_timing=True) if not profile_capture else None
        range_end=torch.cuda.Event(enable_timing=True) if not profile_capture else None
        selected_result=None

        def step():
            nonlocal calls,capturing,selected_result
            tick=calls
            calls+=1
            for worker in shoreline_worker.ACTIVE_WORKERS:
                worker._shoreline_tick=tick;worker._shoreline_prefix=prefix
            if tick==4:
                if profile_capture:
                    dump(output_dir/'selected_interval.json',{'status':'running','pid':os.getpid(),
                        'ticks':[4,5,6],'scope':'three complete engine.step calls; prefill/first selected token excluded'})
                    torch.cuda.synchronize();torch.cuda.profiler.start()
                else:range_start.record()
                torch.cuda.nvtx.range_push('shoreline_ticks_4_6')
                capturing=True
            if capturing:torch.cuda.nvtx.range_push('tick_'+str(tick))
            outputs=original_step()
            if capturing:
                torch.cuda.nvtx.range_pop()
                ledger.append({'tick':tick,'query_tokens_per_request':1,
                    'output_token_counts':[len(o.outputs[0].token_ids) for o in outputs],
                    'request_ids':[o.request_id for o in outputs],
                    'output_ids':[list(o.outputs[0].token_ids) for o in outputs],
                    'attention_length_source':'one-token boundary plus exact cumulative decode progress; backend metadata collected separately',
                    'expected_sequence_lengths':[len(p)+tick for p in prompts]})
            if tick==6:
                if profile_capture:torch.cuda.synchronize()
                else:range_end.record()
                torch.cuda.nvtx.range_pop()
                if profile_capture:torch.cuda.profiler.stop()
                capturing=False
                assert [r['tick'] for r in ledger]==[4,5,6]
                assert all(r['output_token_counts']==[r['tick']+1]*len(prompts) for r in ledger)
                selected_result={'status':'complete','pid':os.getpid(),
                    'ticks':[4,5,6],'ledger':ledger,'timed_tokens':3*len(prompts),
                    'scope':'full engine.step includes graph/model, LM head, sampling and cache updates',
                    'actual_backend_attention_lengths':None,
                    'backend_length_reason':'not inferred from stale CPU input-batch counters'}
                if profile_capture:dump(output_dir/'selected_interval.json',selected_result)
            return outputs
        engine.step=step
        try:
            result=original(engine,prompts,sampling,prefix,torch,output_dir,sensor,
                            False,observer,hardware_sampler)
            assert calls==sampling.max_tokens
            assert not capturing and len(ledger)==3
            if not profile_capture:
                selected_result['gpu_event_ms']=range_start.elapsed_time(range_end)
                selected_result['scope_note']='event-marked three-step interval; no synchronous file publication inside timed decode'
                dump(output_dir/'selected_interval.json',selected_result)
                result['selected_three_step_event_interval']=selected_result
            return result
        finally:
            engine.step=original_step
            for worker in shoreline_worker.ACTIVE_WORKERS:worker._shoreline_tick=None
            if capturing and profile_capture:
                torch.cuda.profiler.stop()

    baseline.run_trial=trial
    baseline.main(a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['timing','energy'],default='timing')
    p.add_argument('--external-profiler',action='store_true',default=True)
    p.add_argument('--collect-worker-metadata',action='store_true')
    p.add_argument('--batch',type=int);p.add_argument('--context',type=int)
    p.add_argument('--generated-tokens',type=int,default=64)
    p.add_argument('--pressure',choices=['normal','pressure'])
    p.add_argument('--layer-events',action='store_true')
    p.add_argument('--empty-layer-events',action='store_true')
    p.add_argument('--ordinary-control',action='store_true',help='no external profiler; separately labeled matched treatment timing')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
