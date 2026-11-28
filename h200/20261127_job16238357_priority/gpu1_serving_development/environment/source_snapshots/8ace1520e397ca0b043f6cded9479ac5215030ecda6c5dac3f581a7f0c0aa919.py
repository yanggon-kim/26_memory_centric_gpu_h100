#!/usr/bin/env python3
"""Matched clean/observed B1 decode pilot; not a loaded-serving experiment."""
import argparse
import json
from pathlib import Path
import traceback


def run_observed(original,Observer,engine,prompts,sampling,prefix,torch,output_dir,*args,**kwargs):
    observer=Observer(engine)
    try:
        result=original(engine,prompts,sampling,prefix,torch,output_dir,*args,**kwargs)
    finally:
        observer.close()
        from software_decode_experiments import dump
        dump(output_dir/('scheduler_'+prefix+'.json'),{'identity':observer.identity,'rows':observer.rows,
             'scope':'instrumented scheduler decisions including prefill; not clean latency or generated-token throughput'})
    rows=observer.rows
    assert len(rows)==sampling.max_tokens
    assert sum(r['prompt_input_queries'] for r in rows)==sum(map(len,prompts))
    assert sum(r['post_prompt_input_queries'] for r in rows)==len(prompts)*(sampling.max_tokens-1)
    assert sum(r['composition']=='prefill_only' for r in rows)==1
    assert sum(r['composition']=='decode_only' for r in rows)==sampling.max_tokens-1
    assert not any(r['composition']=='mixed' or r['any_preempted_request'] for r in rows)
    return result


def main(a):
    import controlled_vllm_experiments as baseline
    from software_decode_experiments import dump
    from h200_scheduler_observer import Observer
    original=baseline.run_trial
    if a.observe:
        def trial(*args,**kwargs):return run_observed(original,Observer,*args,**kwargs)
        baseline.run_trial=trial
    try:baseline.main(a)
    finally:baseline.run_trial=original
    path=a.output/'run.json';data=json.loads(path.read_text())
    assert data['status']=='complete'
    data['scheduler_observation']={'enabled':a.observe,'clean_timing_eligible':not a.observe,
          'scope':'B1 decode-only observer/intrusion pilot; no mixed policies, open-loop arrivals or loaded goodput claim'}
    if a.observe:data['classification']='measured-real-GPU-instrumented'
    dump(path,data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--observe',action='store_true')
    p.add_argument('--mode',default='timing');p.add_argument('--generated-tokens',type=int,default=64)
    p.add_argument('--batch',type=int);p.add_argument('--context',type=int)
    p.add_argument('--external-profiler',action='store_true');p.add_argument('--collect-worker-metadata',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
