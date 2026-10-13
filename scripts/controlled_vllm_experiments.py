#!/usr/bin/env python3
"""Single-process vLLM decode measured after an experimentally checked prefill barrier."""
import argparse
import atexit
import json
import math
import os
from pathlib import Path
import time
import traceback
from software_decode_experiments import Sensor, digest, dump


def validate(path):
    data = json.loads((Path(path)/'run.json').read_text())
    assert data['status'] == 'complete' and data['classification'] == 'measured-real-GPU'
    assert data['engine_client_class'] == 'InprocClient'
    assert data['engine_settings']['async_scheduling'] is False
    assert data['engine_settings']['distributed_executor_backend'] == 'uni'
    assert data['environment']['VLLM_ENABLE_V1_MULTIPROCESSING'] == '0'
    assert data['config']['warmup_trials'] >= 3 and data['config']['trials'] >= 3
    selected={(c['batch_size'],c['context_length'],c['generated_tokens']) for c in data['selected_cases']}
    actual={(c['batch_size'],c['context_length'],c['generated_tokens']) for c in data['cases']}
    assert selected and actual==selected and len(data['cases'])==len(selected), 'selected case identities/count differ'
    for case in data['cases']:
        assert len(case['trials']) == data['config']['trials']
        assert len(case['prompt_ids']) == case['batch_size'] and all(len(row) == case['context_length'] for row in case['prompt_ids'])
        assert case['output_digests_identical'] == (len(set(case['warmup_output_sha256']) | {t['output_sha256'] for t in case['trials']}) == 1)
        assert case['prompt_sha256'] == digest(case['prompt_ids'])
        for trial in case['trials']:
            count, batch = case['generated_tokens'], case['batch_size']
            assert trial['prefill_barrier']['passed'] and trial['prefill_barrier']['request_count'] == batch
            assert all(v == 1 for v in trial['prefill_barrier']['output_token_counts'].values())
            assert all(v == 0 for v in trial['prefill_barrier']['cache_hits'].values())
            assert not any(trial['prefill_barrier']['finished'].values())
            assert trial['decode_iterations'] == count-1
            assert trial['timed_generated_tokens'] == batch*(count-1)
            assert len(trial['output_ids']) == batch and all(len(ids)==count for ids in trial['output_ids'])
            assert trial['output_sha256'] == digest(trial['output_ids'])
            for key in ['cuda_event_ms', 'wall_ms', 'tokens_per_s']:
                assert math.isfinite(trial[key]) and trial[key]>0
            assert math.isclose(trial['tokens_per_s'], batch*(count-1)/(trial['wall_ms']/1000))
            if data['mode'] == 'energy' and trial['telemetry']['joules_per_token'] != 'NA':
                sensor = trial['telemetry']
                assert sensor['window_s'] >= 5 and sensor['counter_j'] > 0
                assert sensor['counter_power_relative_difference'] <= sensor['counter_power_agreement_threshold']
                assert math.isclose(sensor['joules_per_token'], sensor['counter_j']/trial['timed_generated_tokens'])
    return len(data['cases'])


if __name__ == '__main__' and '--validate' in __import__('sys').argv:
    p=argparse.ArgumentParser()
    p.add_argument('--validate',required=True)
    a=p.parse_args()
    try:
        print(f'Validated {validate(a.validate)} controlled vLLM cases')
    except (OSError,ValueError,KeyError,AssertionError) as error:
        p.exit(1,f'INVALID CONTROLLED VLLM: {error}\n')
    raise SystemExit(0)


def run_trial(engine, prompts, sampling, prefix, torch, output_dir, sensor=None):
    assert not engine.has_unfinished_requests(), 'previous requests not drained'
    external = [f'{prefix}_{i}' for i in range(len(prompts))]
    internal = [engine.add_request(rid, {'prompt_token_ids': ids}, sampling)
                for rid,ids in zip(external,prompts)]
    torch.cuda.nvtx.range_push('prefill')
    first = engine.step()
    torch.cuda.nvtx.range_pop()
    barrier = {'request_count':len(first),
               'output_token_counts':{o.request_id:len(o.outputs[0].token_ids) for o in first},
               'cache_hits':{o.request_id:o.num_cached_tokens for o in first},
               'finished':{o.request_id:o.finished for o in first}}
    barrier['passed'] = (len(first)==len(prompts) and set(barrier['output_token_counts'])==set(external)
                         and all(c==1 for c in barrier['output_token_counts'].values())
                         and all(c==0 for c in barrier['cache_hits'].values())
                         and not any(barrier['finished'].values()))
    if not barrier['passed']:
        dump(output_dir/'prefill_barrier_failure.json', {'requested_ids':external,'internal_ids':internal,'barrier':barrier})
    assert barrier['passed'], 'all-request one-token prefill barrier failed; see prefill_barrier_failure.json'
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    if sensor:
        sensor.start()
        before=sensor.read()
    completed={}
    iterations=0
    torch.cuda.nvtx.range_push('decode')
    start=time.perf_counter()
    a.record()
    while engine.has_unfinished_requests():
        outputs=engine.step()
        iterations+=1
        assert outputs and iterations<=sampling.max_tokens-1, 'empty/extra decode scheduler tick'
        # Full cumulative outputs must contain exactly one additional token per tick.
        assert len(outputs)==len(prompts) and {o.request_id for o in outputs}==set(external)
        assert all(len(o.outputs[0].token_ids)==iterations+1 for o in outputs), 'decode did not progress one token per request'
        for output in outputs:
            if output.finished:
                completed[output.request_id]=output
    b.record()
    torch.cuda.synchronize()
    end=time.perf_counter()
    torch.cuda.nvtx.range_pop()
    telemetry=None
    if sensor:
        after=sensor.read()
        telemetry=sensor.finish(start,end,before,after,len(prompts)*(sampling.max_tokens-1),5)
    assert iterations==sampling.max_tokens-1 and set(completed)==set(external)
    ids=[list(completed[rid].outputs[0].token_ids) for rid in external]
    assert all(completed[rid].outputs[0].finish_reason=='length' for rid in external)
    assert all(completed[rid].prompt_token_ids==prompt for rid,prompt in zip(external,prompts))
    result={'prefill_barrier':barrier,'request_ids':external,'internal_request_ids':internal,
            'decode_iterations':iterations,'timed_generated_tokens':len(prompts)*iterations,
            'cuda_event_ms':a.elapsed_time(b),'wall_ms':(end-start)*1000,
            'tokens_per_s':len(prompts)*iterations/(end-start),'output_ids':ids,'output_sha256':digest(ids)}
    if telemetry:
        result['telemetry']=telemetry
    return result


def cleanup_owned_engine(engine, output_dir):
    """Release this standalone runner's engine/groups after measurement or failure."""
    import torch.distributed as distributed
    errors=[]
    try:
        # InprocClient -> EngineCore.shutdown also invokes official distributed cleanup.
        engine.engine_core.shutdown()
    except Exception:
        errors.append({'stage':'engine shutdown','traceback':traceback.format_exc()})
    if distributed.is_initialized():
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            errors.append({'stage':'remaining owned process groups','traceback':traceback.format_exc()})
    dump(output_dir/'cleanup.json',{'status':'failed' if errors else 'complete','errors':errors,
                                   'distributed_initialized_after':distributed.is_initialized(),
                                   'scope':'owned standalone in-process engine; outside all measured windows'})


def main(args):
    config=json.loads(Path(args.config).read_text())
    assert config['warmup_trials']>=3 and config['trials']>=3
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
    import pynvml as nvml
    nvml.nvmlInit()
    from gpu_target import select,verify_cuda
    handle,physical=select(nvml,args.gpu_uuid)
    hardware={'uuid':args.gpu_uuid,'physical_nvml_index':physical,'driver':nvml.nvmlSystemGetDriverVersion(),
              'name':nvml.nvmlDeviceGetName(handle),'graphics_processes':[p.pid for p in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)],
              'preflight_scope':'startup_exclusivity_snapshot; no runtime exclusivity guarantee'}
    import torch
    verify_cuda(torch,args.gpu_uuid)
    import vllm
    from vllm.sampling_params import RequestOutputKind
    assert vllm.__version__=='0.22.1'
    path=Path(config['model_path'])
    assert path.is_dir() and path.name==config['model_revision']
    settings={'model':str(path),'dtype':'bfloat16','seed':config['seed'],'generation_config':'vllm',
              'enable_prefix_caching':False,'enable_chunked_prefill':True,'disable_log_stats':False,
              'async_scheduling':False,'distributed_executor_backend':'uni','tensor_parallel_size':1,
              'max_model_len':8192,'max_num_seqs':8,'max_num_batched_tokens':32768,
              'gpu_memory_utilization':0.70,'enforce_eager':False}
    allowed={'max_model_len','max_num_seqs','max_num_batched_tokens','gpu_memory_utilization'}
    assert set(config.get('engine',{}))<=allowed
    settings.update(config.get('engine',{}))
    data={'status':'running','classification':'measured-real-GPU','mode':args.mode,'config':config,
          'hardware':hardware,'engine_settings':settings,'cases':[],
          'external_instrumentation':args.external_profiler,'baseline_comparison_eligible':not args.external_profiler,
          'versions':{'vllm':vllm.__version__,'torch':torch.__version__,'cuda_runtime':torch.version.cuda},
          'environment':{k:os.environ.get(k,'unset') for k in ['CUDA_HOME','CUDA_VISIBLE_DEVICES','VLLM_ENABLE_V1_MULTIPROCESSING','VLLM_WORKER_MULTIPROC_METHOD','TORCHINDUCTOR_COMPILE_THREADS','MAX_JOBS','OMP_NUM_THREADS']},
          'timing_scope':'single-process GPU decode after all-request one-token prefill barrier; first token excluded',
          'cuda_event_scope':'same-process CUDA-event elapsed decode interval including launch gaps; synchronized wall reported separately',
          'backend_policy':'default optimized vLLM graph/attention kernels; actual engine log preserved externally',
          'sampling_policy':{'temperature':0,'repetition_penalty':1.0,'ignore_eos':True,'detokenize':False,'output_kind':'CUMULATIVE'}}
    dump(args.output/'run.json',data)
    assert not torch.distributed.is_initialized(), 'runner requires ownership of newly initialized process groups'
    llm=vllm.LLM(**settings)
    engine=llm.llm_engine
    cleanup_callback=lambda: cleanup_owned_engine(engine,args.output)
    atexit.register(cleanup_callback)
    data['engine_client_class']=type(engine.engine_core).__name__
    data['executor_class']=type(engine.model_executor).__name__ if hasattr(engine,'model_executor') else 'unavailable'
    assert data['engine_client_class']=='InprocClient' and hasattr(engine,'model_executor'), 'engine not in process'
    props=torch.cuda.get_device_properties(0)
    data['hardware'].update(sm_count=props.multi_processor_count,compute_capability=list(torch.cuda.get_device_capability()))
    count=args.generated_tokens or (1024 if args.mode=='energy' else 64)
    assert 2<=count<=4096
    sampling=vllm.SamplingParams(temperature=0,repetition_penalty=1.0,ignore_eos=True,detokenize=False,
                                 max_tokens=count,seed=config['seed'],output_kind=RequestOutputKind.CUMULATIVE)
    pairs=config.get('energy_cases',[[1,1024]]) if args.mode=='energy' else [[b,c] for b in config['batch_sizes'] for c in config['context_lengths']]
    pairs=[[b,c] for b,c in pairs if (args.batch is None or b==args.batch) and (args.context is None or c==args.context)]
    assert pairs,'no selected cases'
    data['selected_cases']=[{'batch_size':batch,'context_length':context,'generated_tokens':count} for batch,context in pairs]
    data['selection_scope']='explicit materialized stage; selected cases are the completion contract'
    data['requested_case_filters']={'batch':args.batch,'context':args.context,'generated_tokens':args.generated_tokens}
    dump(args.output/'run.json',data)
    vocabulary=json.loads((path/'config.json').read_text())['vocab_size']
    for batch,context in pairs:
        assert context+count<=settings['max_model_len'] and batch<=settings['max_num_seqs']
        generator=torch.Generator().manual_seed(config['seed'])
        prompts=torch.randint(0,vocabulary,(batch,context),generator=generator).tolist()
        case={'batch_size':batch,'context_length':context,'generated_tokens':count,
              'prompt_ids':prompts,'prompt_sha256':digest(prompts),'trials':[]}
        warmup=[]
        for trial in range(config['warmup_trials']):
            warmup.append(run_trial(engine,prompts,sampling,f'B{batch}C{context}_warmup{trial}',torch,args.output)['output_sha256'])
        for trial in range(config['trials']):
            sensor=Sensor(nvml,handle,0.02) if args.mode=='energy' else None
            case['trials'].append(run_trial(engine,prompts,sampling,f'B{batch}C{context}_measured{trial}',torch,args.output,sensor))
        case['warmup_output_sha256']=warmup
        case['output_digests_identical']=len(set(warmup)|{t['output_sha256'] for t in case['trials']})==1
        case['determinism_note']='identical output tokens' if case['output_digests_identical'] else 'output token differences preserved; numerical equivalence requires follow-up'
        data['cases'].append(case)
        dump(args.output/'run.json',data)
        print(f'controlled vLLM B={batch} C={context}: {[round(t["wall_ms"],2) for t in case["trials"]]} ms',flush=True)
    data['status']='complete'
    dump(args.output/'run.json',data)
    print(f'Validated {validate(args.output)} controlled vLLM cases',flush=True)
    cleanup_callback()
    atexit.unregister(cleanup_callback)
    nvml.nvmlShutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--mode',choices=['timing','energy'],required=True)
    p.add_argument('--external-profiler',action='store_true',help='exclude all process timings from ordinary baseline comparison')
    p.add_argument('--batch',type=int)
    p.add_argument('--context',type=int)
    p.add_argument('--generated-tokens',type=int,help='default64 timing,1024 energy; explicit longer energy window if needed')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        main(args)
    except Exception as error:
        if (args.output/'run.json').exists():
            data=json.loads((args.output/'run.json').read_text())
            data['status']='failed'
            dump(args.output/'run.json',data)
        dump(args.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
