#!/usr/bin/env python3
"""Separate first-step/prefill event and wall measurements from pure decode."""
import argparse
from pathlib import Path
import time


def main(a):
    import controlled_vllm_experiments as baseline
    original=baseline.run_trial
    def trial(engine,prompts,sampling,prefix,torch,*args,**kwargs):
        step=engine.step;first=True;prefill={}
        def observed():
            nonlocal first
            if not first:return step()
            first=False
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();begin.record();start=time.perf_counter()
            result=step();end.record();torch.cuda.synchronize()
            prefill.update(wall_ms=(time.perf_counter()-start)*1000,gpu_event_ms=begin.elapsed_time(end),
                input_prompt_tokens=sum(len(p) for p in prompts),first_selected_tokens=len(prompts),
                scope='engine first step including prefill and first selected token; excluded from decode')
            return result
        engine.step=observed
        try:
            result=original(engine,prompts,sampling,prefix,torch,*args,**kwargs)
            result['prefill_and_first_token']=prefill
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
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    main(a)
