#!/usr/bin/env python3
"""Full 64-token decode with a bounded ticks-4/5/6 profiler interval."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main(a):
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import controlled_vllm_experiments as baseline
    from software_decode_experiments import dump
    original=baseline.run_trial

    def trial(engine,prompts,sampling,prefix,torch,output_dir,sensor=None,
              profile_capture=False,observer=None,hardware_sampler=None):
        if not profile_capture:
            return original(engine,prompts,sampling,prefix,torch,output_dir,sensor,
                            False,observer,hardware_sampler)
        original_step=engine.step
        calls=0;ledger=[];capturing=False

        def step():
            nonlocal calls,capturing
            tick=calls
            calls+=1
            if tick==4:
                dump(output_dir/'selected_interval.json',{'status':'running','pid':os.getpid(),
                    'ticks':[4,5,6],'scope':'three complete engine.step calls; prefill/first selected token excluded'})
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
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
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
                capturing=False
                assert [r['tick'] for r in ledger]==[4,5,6]
                assert all(r['output_token_counts']==[r['tick']+1]*len(prompts) for r in ledger)
                dump(output_dir/'selected_interval.json',{'status':'complete','pid':os.getpid(),
                    'ticks':[4,5,6],'ledger':ledger,'timed_tokens':3*len(prompts),
                    'scope':'full engine.step includes graph/model, LM head, sampling and cache updates',
                    'actual_backend_attention_lengths':None,
                    'backend_length_reason':'not inferred from stale CPU input-batch counters'})
            return outputs
        engine.step=step
        try:
            result=original(engine,prompts,sampling,prefix,torch,output_dir,sensor,
                            False,observer,hardware_sampler)
            assert calls==sampling.max_tokens
            assert not capturing and len(ledger)==3
            return result
        finally:
            engine.step=original_step
            if capturing:
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
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
