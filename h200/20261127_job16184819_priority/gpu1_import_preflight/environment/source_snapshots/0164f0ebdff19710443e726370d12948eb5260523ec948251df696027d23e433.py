#!/usr/bin/env python3
"""Versioned post-prefill workload conditioning; no clock or power settings change."""
import argparse
from pathlib import Path
import time


def main(a):
    import controlled_vllm_experiments as baseline
    original=baseline.run_trial
    def trial(engine,prompts,sampling,prefix,torch,*args,**kwargs):
        step=engine.step;first=True;conditioning={}
        if a.conditioning_mode=='scalar_fma':
            from shoreline_kernels import condition
            n=torch.cuda.get_device_properties(0).multi_processor_count*256
            state=torch.ones(n,device='cuda');out=torch.empty_like(state)
        def conditioned_step():
            nonlocal first
            output=step()
            if first:
                first=False
                assert len(output)==len(prompts) and all(len(o.outputs[0].token_ids)==1 for o in output)
                # A bounded register-only delay kernel conditions ordinary DVFS after compute-heavy prefill.
                # It does not advance requests, read model/KV data, or alter any device setting.
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();start=time.perf_counter()
                if a.conditioning_mode=='register_sleep':torch.cuda._sleep(a.conditioning_cycles)
                else:condition[(n//256,)](state,out,n,10000000,256)
                end.record();torch.cuda.synchronize()
                conditioning.update(mode=a.conditioning_mode,cycles=a.conditioning_cycles if a.conditioning_mode=='register_sleep' else None,
                    scalar_fma_steps=10000000 if a.conditioning_mode=='scalar_fma' else None,gpu_event_ms=begin.elapsed_time(end),
                    wall_ms=(time.perf_counter()-start)*1000,
                    scope='post-prefill GPU-workload conditioning outside timed decode; not ordinary unconditioned TTFT')
            return output
        engine.step=conditioned_step
        try:
            result=original(engine,prompts,sampling,prefix,torch,*args,**kwargs)
            result['post_prefill_conditioning']=conditioning
            return result
        finally:engine.step=step
    baseline.run_trial=trial
    baseline.main(a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',default='timing')
    p.add_argument('--external-profiler',action='store_true');p.add_argument('--collect-worker-metadata',action='store_true')
    p.add_argument('--batch',type=int);p.add_argument('--context',type=int);p.add_argument('--generated-tokens',type=int,default=64)
    p.add_argument('--conditioning-cycles',type=int,default=100000000)
    p.add_argument('--conditioning-mode',choices=['register_sleep','scalar_fma'],default='register_sleep')
    a=p.parse_args();assert 0<a.conditioning_cycles<=200000000
    a.output.mkdir(parents=True,exist_ok=False)
    main(a)
