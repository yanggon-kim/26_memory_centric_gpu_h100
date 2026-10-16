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
                torch.cuda._sleep(a.conditioning_cycles)
                end.record();torch.cuda.synchronize()
                conditioning.update(cycles=a.conditioning_cycles,gpu_event_ms=begin.elapsed_time(end),
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
    a=p.parse_args();assert 0<a.conditioning_cycles<=200000000
    a.output.mkdir(parents=True,exist_ok=False)
    main(a)
