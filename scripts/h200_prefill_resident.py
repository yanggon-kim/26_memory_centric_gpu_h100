#!/usr/bin/env python3
"""Prepared full-model whole/chunked prefill pilot; GPU validation is pending.

One full model per independent acquisition, no prefix-cache reuse. The first
selected token is measured with prefill; the unchanged controlled decoder
then times exactly63 decode steps after its original one-token barrier.
All policy/observer/event costs remain in the reported instrumented scopes.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback


def drain_prefill(step,limit=1024):
    for calls in range(1,limit+1):
        outputs=step()
        if outputs:
            assert len(outputs)==1 and len(outputs[0].outputs[0].token_ids)==1
            assert not outputs[0].finished,'prefill must not finish a64-token request'
            return outputs,calls
    raise RuntimeError('bounded prefill did not produce the one-token barrier')


def prefill_trial(original,engine,prompts,sampling,prefix,torch,output_dir,budget,whole,capture=False):
    from h200_serving_policy import Policy
    from h200_scheduler_observer import Observer
    from gpu_target import describe
    from software_decode_experiments import dump
    import pynvml
    assert len(prompts)==1 and sampling.max_tokens==64
    policy=Policy(engine,'whole_separate' if whole else 'chunked_separate',budget)
    try:observer=Observer(engine)
    except BaseException:
        policy.close();raise
    step=engine.step;first=True;prefill={};capturing=False
    handle=pynvml.nvmlDeviceGetHandleByUUID(__import__('os').environ['H200_PREFILL_UUID'])
    def observed_step():
        nonlocal first,capturing
        if not first:return step()
        first=False
        before=describe(pynvml,handle,torch)
        torch.cuda.synchronize()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        if capture:
            torch.cuda.profiler.start();capturing=True
            torch.cuda.nvtx.range_push('prefill_selected')
        begin.record();start_ns=time.time_ns();start=time.perf_counter()
        outputs,calls=drain_prefill(step)
        end.record();torch.cuda.synchronize()
        finish=time.perf_counter();end_ns=time.time_ns()
        if capture:torch.cuda.nvtx.range_pop()
        after=describe(pynvml,handle,torch)
        rows=copy.deepcopy(observer.rows)
        assert len(rows)==calls
        assert all(r['composition']=='prefill_only' and not r['any_preempted_request'] for r in rows)
        assert sum(r['prompt_input_queries'] for r in rows)==len(prompts[0])
        assert not any(r['post_prompt_input_queries'] for r in rows)
        prefill.update(wall_ms=(finish-start)*1000,gpu_event_ms=begin.elapsed_time(end),
            wall_start_unix_ns=start_ns,wall_end_unix_ns=end_ns,hardware_before=before,hardware_after=after,
            actual_prefill_steps=calls,input_prompt_tokens=len(prompts[0]),first_selected_tokens=1,
            token_budget=budget,whole=whole,scheduler_rows=rows,
            scope='instrumented complete prefill plus first selected token; includes policy/observer/event and chunk launch costs; excludes later63decode steps')
        if capture:
            dump(output_dir/'prefill_interval_output.json',{'prefix':prefix,'prefill':prefill,
                'first_selected_ids':[list(o.outputs[0].token_ids) for o in outputs],
                'scope':'first-token functional publication before profiler stop; full64token correctness remains a separate required gate'})
            torch.cuda.profiler.stop();capturing=False
        return outputs
    engine.step=observed_step
    try:
        result=original(engine,prompts,sampling,prefix,torch,output_dir,
                        hardware_sampler=lambda:describe(pynvml,handle,torch))
        assert len(observer.rows)==prefill['actual_prefill_steps']+63
        result['prefill_and_first_token']=prefill
        result['scheduler_decisions']=copy.deepcopy(observer.rows)
        return result
    finally:
        engine.step=step
        observer.close();policy.close()
        dump(output_dir/('scheduler_'+prefix+'.json'),{'policy':policy.identity,'observer':observer.identity,
            'rows':observer.rows,'prefill':prefill,'scope':'instrumented request trajectory; not clean baseline timing'})
        if capturing:
            torch.cuda.profiler.stop()


def main(a):
    import os
    os.environ['H200_PREFILL_UUID']=a.gpu_uuid
    import controlled_vllm_experiments as baseline
    from software_decode_experiments import dump
    original=baseline.run_trial;cfg=json.loads(Path(a.config).read_text())
    assert a.profile_condition in (('whole','chunk256') if a.pilot else ('whole','chunk256','chunk1024','chunk4096','whole_end'))
    assert cfg['batch_sizes']==[1] and len(cfg['context_lengths'])==1, 'prefill producer requires one batch-one workload shape'
    assert cfg['warmup_trials']==3 and cfg['trials'] in (3,10)
    assert not cfg['engine']['enforce_eager'] and not cfg.get('decode_conditioning')
    records={};initialized=False;expected=None
    manifest={'status':'running','protocol':'resident_prefill_v1','acquisition':a.acquisition,'conditions':records,
        'profile_condition':a.profile_condition if a.external_profiler else None,
        'scope':'one resident full BF16 model per acquisition; cold means no prefix/KV reuse, not cold weights or forced cache flushing; instrumented policy/event/observer scopes'}
    def save():dump(a.output/'prefill_resident.json',manifest)
    save()
    def trial(engine,prompts,sampling,prefix,torch,output_dir,*args,**kwargs):
        nonlocal initialized,expected
        if not initialized:
            initialized=True;assert 'warmup0' in prefix and len(prompts)==1
            choices=[('whole',len(prompts[0]),True),('chunk256',256,False)]
            if not a.pilot:choices.extend([('chunk1024',1024,False),('chunk4096',4096,False),('whole_end',len(prompts[0]),True)])
            manifest['condition_order']=[c[0] for c in choices]
            for name,budget,whole in choices:
                sub=output_dir/name;sub.mkdir(exist_ok=False)
                record={'status':'running','budget':budget,'whole':whole,'warmups':[],'trials':[]}
                records[name]=record;save()
                for phase,n in (('warmup',3),('measured',cfg['trials'])):
                    for index in range(n):
                        result=prefill_trial(original,engine,prompts,sampling,f'{name}_{phase}{index}',
                            torch,sub,budget,whole,capture=a.external_profiler and name==a.profile_condition and phase=='measured' and index==0)
                        record['warmups' if phase=='warmup' else 'trials'].append(result);save()
                        if expected is None:expected=result['output_ids']
                        assert result['output_ids']==expected,'chunked/full greedy output disagreement retained'
                record['status']='complete';save()
        phase,index=('warmups',int(prefix.rsplit('warmup',1)[1])) if 'warmup' in prefix else ('trials',int(prefix.rsplit('measured',1)[1]))
        return copy.deepcopy(records['whole'][phase][index])
    baseline.run_trial=trial
    try:baseline.main(a)
    finally:baseline.run_trial=original
    path=a.output/'run.json';template=json.loads(path.read_text())
    for name,record in records.items():
        data=copy.deepcopy(template);case=data['cases'][0]
        case.update(trials=record['trials'],warmup_output_ids=[t['output_ids'] for t in record['warmups']],
            warmup_output_sha256=[t['output_sha256'] for t in record['warmups']],output_digests_identical=True)
        case['hardware_before_after_scope']='outer snapshots span the whole resident acquisition; use per-trial prefill/decode endpoints and sampled monitor for each condition'
        data['prefill_protocol']={'version':'resident_prefill_v1','acquisition':a.acquisition,
            'condition':name,'budget':record['budget'],'whole':record['whole'],'full_model_loads_in_producer':1}
        dump(a.output/name/'run.json',data);baseline.validate(a.output/name)
        data.update(classification='measured-real-GPU-instrumented',baseline_comparison_eligible=False)
        dump(a.output/name/'run.json',data)
    template.update(classification='measured-real-GPU-instrumented',baseline_comparison_eligible=False)
    dump(path,template);manifest['status']='complete';save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--acquisition',type=int,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--external-profiler',action='store_true')
    p.add_argument('--profile-condition',choices=('whole','chunk256','chunk1024','chunk4096','whole_end'),default='whole')
    p.add_argument('--mode',choices=['timing'],default='timing');p.add_argument('--generated-tokens',type=int,default=64)
    p.add_argument('--batch',type=int);p.add_argument('--context',type=int);p.add_argument('--collect-worker-metadata',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except BaseException as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
