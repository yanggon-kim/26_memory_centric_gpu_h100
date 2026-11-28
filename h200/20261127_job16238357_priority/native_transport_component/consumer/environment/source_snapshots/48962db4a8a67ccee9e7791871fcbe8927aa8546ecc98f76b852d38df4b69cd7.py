#!/usr/bin/env python3
"""Eager SM sweep with one resident full model per independent acquisition.

This is a new protocol, never pooled with the earlier fresh-model-per-point
curve. Conditions execute sequentially; no two green streams run concurrently.
The unchanged controlled decoder checks every one-token prefill barrier.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import traceback
SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def conditions(pilot=False, proof=False):
    if pilot:
        return [('stock',1.,False),('green132',1.,True),('green40',.3,True)]
    rows=[('stock',1.,False),('green132',1.,True),('green112',.85,True),
          ('green96',.75,True),('green80',.6,True),('green56',.45,True),('green40',.3,True)]
    if not proof:
        rows.insert(4,('green132_interleaved',1.,True))
        rows.append(('green132_end',1.,True))
    return rows


def bound_trial(original,engine,prompts,sampling,prefix,torch,output,green,stream,proof,
                capture=False,range_name=None,**kwargs):
    """Match the established eager per-step binding, including stock control."""
    from shoreline_green import stream_proof
    primary=torch.cuda.current_stream()
    step=engine.step
    pushed=False;capturing=False;calls=0
    if green is not None:
        green.set_context();pushed=True
    torch.cuda.set_stream(stream)
    def bound_step():
        nonlocal calls,capturing
        tick=calls;calls+=1
        if green is not None:green.set_context()
        torch.cuda.set_stream(stream)
        try:
            if capture and tick==4:
                torch.cuda.synchronize();torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(range_name);capturing=True
            result=step()
            if capture and tick==6:
                torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop();capturing=False
            return result
        finally:
            if green is not None:green.pop_context()
    engine.step=bound_step
    try:
        before=stream_proof(torch.cuda.current_stream())
        result=original(engine,prompts,sampling,prefix,torch,output,profile_capture=False,**kwargs)
        after=stream_proof(torch.cuda.current_stream())
        assert calls==sampling.max_tokens
        assert before['actual_sm_count']==after['actual_sm_count']==proof['actual_sm_count']
        assert before['green_context_id']==after['green_context_id']==proof['green_context_id']
        result.update(green_stream_before=before,green_stream_after=after)
        return result
    finally:
        engine.step=step
        if capturing:
            torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop()
        if pushed:green.pop_context()
        torch.cuda.set_stream(primary)


def main(a):
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
    import controlled_vllm_experiments as baseline
    from software_decode_experiments import dump,digest
    from shoreline_green import create
    original=baseline.run_trial
    cfg=json.loads(Path(a.config).read_text())
    assert len(cfg['batch_sizes'])==len(cfg['context_lengths'])==1, 'resident producer requires exactly one workload shape'
    assert cfg['engine']['enforce_eager'] and cfg['engine']['compilation_config']=={'mode':0}
    assert cfg['warmup_trials']==3 and cfg['trials']==(3 if a.external_profiler else 10)
    conditioning=cfg.get('decode_conditioning')
    if conditioning:
        from decode_precondition import validate
        validate(conditioning)
    protocol='resident_eager_memory_settle_v3' if conditioning else 'resident_eager_v1'
    rows=conditions(a.pilot,a.external_profiler)
    records={};expected=None;initialized=False
    manifest={'status':'running','protocol':protocol,'conditioning':conditioning,'acquisition':a.acquisition,
              'condition_order':[r[0] for r in rows],'conditions':records,
              'scope':'one resident full BF16 model per independent process/acquisition; sequential SM conditions; new cache/initialization history, never pooled with fresh-model-per-point protocol'}
    def save():dump(a.output/'resident_sweep.json',manifest)
    save()
    def trial(engine,prompts,sampling,prefix,torch,output_dir,sensor=None,
              profile_capture=False,observer=None,hardware_sampler=None):
        nonlocal initialized,expected
        assert sensor is None and observer is None and not engine.has_unfinished_requests()
        if not initialized:
            initialized=True
            assert 'warmup0' in prefix
            import pynvml
            from gpu_target import describe
            handle=pynvml.nvmlDeviceGetHandleByUUID(a.gpu_uuid)
            sample=lambda:describe(pynvml,handle,torch)
            # Initialize after the full primary-context model is loaded. Only
            # resources/streams vary; weights, KV precision and kernels do not.
            resources={}
            for name,fraction,green_path in rows:
                key=(fraction,green_path)
                if key not in resources:
                    resources[key]=create(torch,fraction,force_green=green_path,rounding='nearest')
                green,stream,proof=resources[key]
                actual=proof['actual_sm_count'];physical=torch.cuda.get_device_properties(0).multi_processor_count
                assert (actual==physical if not green_path or fraction==1. else 0<actual<physical)
                record={'name':name,'fraction':fraction,'stock':not green_path,'allocation':proof,
                        'physical_sm_count':physical,'status':'running','warmups':[],'trials':[],
                        'prompt_ids':prompts,'prompt_sha256':digest(prompts)}
                records[name]=record;save()
                sub=output_dir/name;sub.mkdir(exist_ok=False)
                for phase,n in (('warmup',3),('measured',cfg['trials'])):
                    for index in range(n):
                        result=bound_trial(original,engine,prompts,sampling,
                            f'{name}_{phase}{index}',torch,sub,green,stream,proof,
                            capture=a.external_profiler and phase=='measured' and index==0,
                            range_name=f'resident_{name}_ticks_4_6',
                            hardware_sampler=sample)
                        if expected is None:expected=result['output_ids']
                        assert result['output_ids']==expected,'condition output differs from stock; all raw evidence retained'
                        record['warmups' if phase=='warmup' else 'trials'].append(result);save()
                record['status']='complete';save()
            manifest['status']='conditions_captured';save()
        # Baseline owns model creation, canonical prompts, metadata, validation
        # and cleanup. Return already captured stock trials to its ordinary
        # fixed loop; never run or time a second hidden decode.
        phase,index=('warmups',int(prefix.rsplit('warmup',1)[1])) if 'warmup' in prefix else ('trials',int(prefix.rsplit('measured',1)[1]))
        return copy.deepcopy(records['stock'][phase][index])
    baseline.run_trial=trial
    try:baseline.main(a)
    finally:baseline.run_trial=original
    template=json.loads((a.output/'run.json').read_text())
    assert template['status']=='complete'
    for name,_,_ in rows:
        record=records[name];data=copy.deepcopy(template);case=data['cases'][0]
        if conditioning:
            case['warmup_conditioning_receipts']=[t.get('post_prefill_conditioning') for t in record['warmups']]
        case.update(trials=record['trials'],warmup_output_ids=[t['output_ids'] for t in record['warmups']],
                    warmup_output_sha256=[t['output_sha256'] for t in record['warmups']],
                    output_digests_identical=True)
        data['resident_protocol']={'version':protocol,'conditioning':conditioning,'acquisition':a.acquisition,
                                   'condition':name,'condition_order':manifest['condition_order'],
                                   'full_model_loads_in_this_producer':1,'allocation':record['allocation']}
        # Per-condition timing windows are the retained original trial windows;
        # the outer baseline's hardware_before/after cover the entire sweep.
        case['hardware_before_after_scope']='outer baseline snapshots cover whole resident acquisition; use per-trial endpoints and sampled monitor for each condition'
        dump(a.output/name/'run.json',data)
        baseline.validate(a.output/name)
    manifest['status']='complete';save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--acquisition',type=int,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--external-profiler',action='store_true')
    p.add_argument('--mode',default='timing');p.add_argument('--generated-tokens',type=int,default=64)
    p.add_argument('--batch',type=int);p.add_argument('--context',type=int)
    p.add_argument('--collect-worker-metadata',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except BaseException as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()});raise
