#!/usr/bin/env python3
"""Pending H100 eager-vLLM capture through public RPC and exact decode barrier."""
import argparse
from ctypes import CDLL,c_char_p
import json
import os
from pathlib import Path
import sys
from controlled_vllm_experiments import run_trial,cleanup_owned_engine
from software_decode_experiments import digest,dump
from tensor_manifest import parameters,vllm_kv
import hashlib


def ordinary_control_guard(env, warmups, trials):
    assert warmups>=3 and trials>=10, 'ordinary trace-harness control requires >=3 warmups and >=10 trials'
    assert not env.get('CUDA_INJECTION64_PATH') and not env.get('LD_PRELOAD'), 'ordinary calibration rejects injected/preloaded libraries'
    assert env.get('NVBIT_INSTRUMENTATION_ENABLED','0')=='0', 'ordinary calibration rejects active NVBit'


def describe_worker(worker):
    model=worker.model_runner.model # inspected upstream vllm_example.py worker API
    return {'modules':[n for n,_ in model.named_modules()], 'parameter_storage_manifest':parameters(model)}


def attach_worker_hooks(worker,layers):
    from torch_hook import TorchModelHookWrapper,hook_nvbit_to_layer
    import torch_hook.pytorch_profiler as hooks
    wrapper=TorchModelHookWrapper(worker.model_runner.model)
    for name in layers:hook_nvbit_to_layer(wrapper,name)
    assert hooks.g_found_nvbit_tracer,'hook did not find actual tracer callbacks'
    return True


def main(a):
    if a.ordinary_calibration:ordinary_control_guard(os.environ,a.warmups,a.trials)
    config=json.loads(a.config.read_text())
    assert Path(config['model_path']).name==config['model_revision'],'immutable model snapshot mismatch'
    assert 1<=a.steps<=4 and 1<=a.batch<=8 and a.context>0
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_ALLOW_INSECURE_SERIALIZATION='1')
    import pynvml as nvml
    from gpu_target import select,verify_cuda,describe
    nvml.nvmlInit();handle,_=select(nvml,a.gpu_uuid)
    import torch
    verify_cuda(torch,a.gpu_uuid)
    import vllm
    from vllm.sampling_params import RequestOutputKind
    from vllm_identity import verify as verify_vllm
    vllm_identity=verify_vllm(vllm)
    sys.path.insert(0,str(a.accel_root/'util/tracer_nvbit/others'))
    settings={'model':config['model_path'],'dtype':'bfloat16','seed':config['seed'],'generation_config':'vllm',
              'enable_prefix_caching':False,'enable_chunked_prefill':True,'disable_log_stats':False,
              'async_scheduling':False,'distributed_executor_backend':'uni','tensor_parallel_size':1,
              'enforce_eager':True,'compilation_config':{'mode':0},**config['engine']}
    assert settings['enforce_eager'] and settings['compilation_config']['mode']==0, 'trace/calibration requires the declared eager engine'
    assert a.batch<=settings['max_num_seqs'] and a.context+a.steps+1<=settings['max_model_len']
    settings['max_num_batched_tokens']=max(settings['max_num_batched_tokens'],a.batch*a.context)
    llm=vllm.LLM(**settings);engine=llm.llm_engine
    assert type(engine.engine_core).__name__=='InprocClient'
    a.output.mkdir(parents=True,exist_ok=True)
    try:
        description=llm.collective_rpc(describe_worker)[0]
        dump(a.output/'worker_model_manifest.json',description)
        vocabulary=json.loads((Path(config['model_path'])/'config.json').read_text())['vocab_size']
        generator=torch.Generator().manual_seed(config['seed'])
        prompts=torch.randint(0,vocabulary,(a.batch,a.context),generator=generator).tolist()
        params=vllm.SamplingParams(temperature=0,repetition_penalty=1.0,ignore_eos=True,detokenize=False,
                                  max_tokens=a.steps+1,seed=config['seed'],output_kind=RequestOutputKind.CUMULATIVE)
        if a.ordinary_calibration:
            maps=Path('/proc/self/maps').read_text()
            assert 'tracer_tool.so' not in maps and 'libnvbit.so' not in maps, 'tracer already loaded in ordinary control'
            case={'batch_size':a.batch,'context_length':a.context,'generated_tokens':a.steps+1,
                  'prompt_ids':prompts,'prompt_sha256':digest(prompts),'trials':[],
                  'warmup_output_ids':[],'warmup_output_sha256':[]}
            data={'status':'running','classification':'measured-real-GPU','mode':'ordinary trace-harness calibration',
                  'config':config,'engine_settings':settings,'cases':[case],'external_instrumentation':False,
                  'same_program':'trace_vllm_decode.py engine initialization, prompts, sampling and run_trial references',
                  'coverage':'complete-model eager decode; selected-layer trace remains a distinct scope',
                  'no_injection_proof':{'environment_guard_passed':True,'tracer_libraries_absent_from_maps':True},
                  'versions':{'vllm':vllm.__version__,'torch':torch.__version__,'cuda_runtime':torch.version.cuda},
                  'vllm_identity':vllm_identity,'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(__file__).parent.glob('*.py')},
                  'timing_scope':'two pure-decode ticks after all-request one-token prefill barrier; host launch gaps included',
                  'trace_hook_overhead':'not silently transferred; tracer-injected disabled references and instrumented execution remain separate'}
            dump(a.output/'run.json',data)
            for i in range(a.warmups):
                warm=run_trial(engine,prompts,params,f'ordinary_warmup_{i}',torch,a.output)
                case['warmup_output_ids'].append(warm['output_ids']);case['warmup_output_sha256'].append(warm['output_sha256'])
            for i in range(a.trials):
                trial=run_trial(engine,prompts,params,f'ordinary_measured_{i}',torch,a.output,
                                hardware_sampler=lambda:describe(nvml,handle,torch))
                case['trials'].append(trial);dump(a.output/'run.json',data)
            hashes=case['warmup_output_sha256']+[t['output_sha256'] for t in case['trials']]
            case['output_digests_identical']=len(set(hashes))==1
            data['status']='complete' if case['output_digests_identical'] else 'failed'
            dump(a.output/'run.json',data)
            assert case['output_digests_identical'], 'ordinary trace-harness outputs differ across warmups/trials'
            return
        reference=None;references=[]
        for i in range(3):
            reference=run_trial(engine,prompts,params,f'untraced_reference_{i}',torch,a.output)
            references.append(reference)
        dump(a.output/'injection_disabled_references.json',{'classification':'measured-real-GPU-in-injected-process',
            'scope':'instrumentation-disabled in injected process; distinct from standalone no-injection eager calibration','trials':references})
        assert len({row['output_sha256'] for row in references})==1, 'injected-process disabled references disagree; preserve failure before tracing'
        request_ids=[f'capture_{i}' for i in range(a.batch)]
        for rid,prompt in zip(request_ids,prompts):engine.add_request(rid,{'prompt_token_ids':prompt},params)
        first=engine.step()
        barrier={'ids':[o.request_id for o in first],'counts':[len(o.outputs[0].token_ids) for o in first],
                 'cache_hits':[o.num_cached_tokens for o in first]}
        assert set(barrier['ids'])==set(request_ids) and all(n==1 for n in barrier['counts']) and all(n==0 for n in barrier['cache_hits']),'prefill barrier failed'
        torch.cuda.synchronize()
        lib=CDLL(os.environ['CUDA_INJECTION64_PATH'],mode=os.RTLD_NOLOAD|os.RTLD_NOW)
        for name in ['enable_nvbit_instrumentation','disable_nvbit_instrumentation','set_nvbit_instrumentation_tag']:
            getattr(lib,name).restype=None
        lib.set_nvbit_instrumentation_tag.argtypes=[c_char_p]
        layers=[] if a.layers=='full' else a.layers.split(',')
        assert all(name in description['modules'] for name in layers),'unknown module; see worker_model_manifest'
        assert not any(x!=y and (x=='' or y.startswith(x+'.')) for x in layers for y in layers),'nested hooks corrupt instrumentation state'
        if layers:llm.collective_rpc(attach_worker_hooks,args=(layers,))
        metadata={'status':'running','classification':'measured-real-GPU-trace','engine':'vllm-eager-inprocess',
                  'engine_settings':settings,'prefill_barrier':barrier,'gpu_uuid':a.gpu_uuid,
                  'model_id':config['model_id'],'model_revision':config['model_revision'],
                  'hardware':describe(nvml,handle,torch),'seed':config['seed'],'batch_size':a.batch,'context_length':a.context,'timed_decode_steps':a.steps,
                  'timed_generated_tokens':a.batch*a.steps,'generated_tokens_per_sequence':a.steps+1,
                  'prompt_ids':prompts,'prompt_sha256':digest(prompts),'selected_modules':layers,
                  'coverage':'entire controlled decode engine.step including LM head/sampling kernels' if not layers else 'selected disjoint worker modules only',
                  'address_manifest':'worker_model_manifest.json parameters and kv_storage_lifecycle.json; activation/temporary/slot mappings explicitly unmapped',
                  'boundary':'all requests complete prefill before runtime callback/hooks enabled; ACTIVE_FROM_START=1',
                  'reference_uninstrumented_output_ids':reference['output_ids'],'reference_sha256':reference['output_sha256'],
                  'workload':'seeded synthetic uniform token IDs; not representative expert/input locality',
                  'execution':'vLLM eager compilation_mode0; distinct from graph-optimized baseline; actual Hopper opcode coverage still to verify',
                  'versions':{'vllm':vllm.__version__,'torch':torch.__version__,'cuda_runtime':torch.version.cuda},
                  'vllm_identity':vllm_identity}
        dump(a.output/'trace_metadata.json',metadata)
        completed={}
        kv_lifecycle=[]
        request_pages=[]
        from h200_worker_metadata import snapshot_worker
        for step in range(a.steps):
            kv_lifecycle.append(llm.collective_rpc(vllm_kv,args=(f'before_decode_step_{step}',))[0])
            request_pages.append({'step':step,'phase':'before','worker':llm.collective_rpc(snapshot_worker)[0],
                'scope':'actual last-prepared CPU page mappings; computed counters may lag execution'})
            torch.cuda.nvtx.range_push(f'decode_step_{step}')
            if not layers:
                # A model.forward hook alone omits vLLM's separate LM-head logits path.
                lib.set_nvbit_instrumentation_tag(c_char_p(f'decode_step_{step}'.encode()))
                lib.enable_nvbit_instrumentation()
            outputs=engine.step()
            torch.cuda.synchronize()
            lib.disable_nvbit_instrumentation()
            torch.cuda.nvtx.range_pop()
            kv_lifecycle.append(llm.collective_rpc(vllm_kv,args=(f'after_decode_step_{step}',))[0])
            request_pages.append({'step':step,'phase':'after','worker':llm.collective_rpc(snapshot_worker)[0],
                'scope':'actual CPU page mappings after this step; no GPU copies added inside instrumentation'})
            assert len(outputs)==a.batch and {o.request_id for o in outputs}==set(request_ids)
            assert all(len(o.outputs[0].token_ids)==step+2 for o in outputs),'non-single-token progression'
            for out in outputs:
                if out.finished:completed[out.request_id]=out
        dump(a.output/'kv_storage_lifecycle.json',kv_lifecycle)
        dump(a.output/'request_page_lifecycle.json',request_pages)
        assert not engine.has_unfinished_requests() and set(completed)==set(request_ids)
        ids=[list(completed[rid].outputs[0].token_ids) for rid in request_ids]
        matched=ids==reference['output_ids']
        metadata.update(status='complete' if matched else 'failed',output_ids=ids,output_sha256=digest(ids),correctness_passed=matched)
        dump(a.output/'trace_metadata.json',metadata)
        assert matched,'instrumented output differs from same-shape uninstrumented eager vLLM reference'
    finally:
        if 'lib' in locals():lib.disable_nvbit_instrumentation()
        cleanup_owned_engine(engine,a.output)
        nvml.nvmlShutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['accel-root','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--layers',default='full')
    p.add_argument('--ordinary-calibration',action='store_true')
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--trials',type=int,default=10)
    main(p.parse_args())
