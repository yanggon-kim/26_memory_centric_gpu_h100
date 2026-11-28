#!/usr/bin/env python3
"""Eager decode-only NVBit capture through the pinned upstream torch_hook API."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from software_decode_experiments import digest,dump
from tensor_manifest import parameters,hf_kv


def main(args):
    config=json.loads(args.config.read_text())
    assert 1<=args.steps<=4 and 1<=args.batch<=8,'bounded trace smoke only; explicitly extend after storage pilot'
    import pynvml as nvml
    from gpu_target import select,verify_cuda,describe
    nvml.nvmlInit()
    handle,_=select(nvml,args.gpu_uuid)
    import torch
    import transformers
    verify_cuda(torch,args.gpu_uuid)
    sys.path.insert(0,str(args.accel_root/'util/tracer_nvbit/others'))
    from torch_hook import TorchModelHookWrapper,hook_nvbit_to_layer
    path=Path(config['model_path'])
    assert path.name==config['model_revision']
    model=transformers.AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
                     dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'parameter_storage_manifest.json',parameters(model))
    generator=torch.Generator().manual_seed(config['seed'])
    prompt=torch.randint(0,model.config.vocab_size,(args.batch,args.context),generator=generator).to('cuda')
    with torch.inference_mode():
        # No hook exists during warmup/prefill. ACTIVE_FROM_START=1 is safe only
        # because NVBIT_INSTRUMENTATION_ENABLED starts at zero in run_trace.py.
        for _ in range(3):
            warm=model(input_ids=prompt,use_cache=True,logits_to_keep=1)
            warm_token=warm.logits[:,-1:].argmax(-1)
            model(input_ids=warm_token,past_key_values=warm.past_key_values,use_cache=True,logits_to_keep=1)
        del warm,warm_token
        reference=model.generate(input_ids=prompt,max_new_tokens=args.steps+1,do_sample=False,eos_token_id=None,repetition_penalty=1.0,logits_to_keep=1)
        reference_ids=reference[:,args.context:].cpu().tolist()
        del reference
        torch.cuda.nvtx.range_push('prefill_untraced')
        prefill=model(input_ids=prompt,use_cache=True,logits_to_keep=1)
        past=prefill.past_key_values
        token=prefill.logits[:,-1:].argmax(-1)
        generated=[token.clone()]
        del prefill
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        available=[name for name,_ in model.named_modules()]
        requested=[''] if args.layers=='full' else args.layers.split(',')
        args.output.mkdir(parents=True,exist_ok=True)
        dump(args.output/'available_modules.json',available)
        assert all(name in available for name in requested),'unknown layer; inspect available_modules.json'
        assert not any(a!=b and (a=='' or b.startswith(a+'.')) for a in requested for b in requested),'nested hooks would incorrectly disable parent capture'
        args.output.mkdir(parents=True,exist_ok=True)
        dump(args.output/'available_modules.json',available)
        wrapper=TorchModelHookWrapper(model)
        for name in requested:
            hook_nvbit_to_layer(wrapper,name)
        # Upstream hook silently ignores a missing tracer: prove it actually loaded.
        import torch_hook.pytorch_profiler as hook_module
        assert hook_module.g_found_nvbit_tracer,'NVBit callback symbols not loaded; refuse empty capture'
        metadata={'status':'running','classification':'measured-real-GPU-trace','model_id':config['model_id'],
                  'model_revision':config['model_revision'],'gpu_uuid':args.gpu_uuid,'batch_size':args.batch,
                  'context_length':args.context,'timed_decode_steps':args.steps,'hardware':describe(nvml,handle,torch),'seed':config['seed'],
                  'generated_tokens_per_sequence':args.steps+1,'timed_generated_tokens':args.batch*args.steps,
                  'prompt_ids':prompt.cpu().tolist(),'prompt_sha256':digest(prompt.cpu().tolist()),
                  'selected_modules':requested,'coverage':'whole model forward per step' if requested==[''] else 'selected disjoint modules only',
                  'workload':'seeded synthetic uniform token IDs; not representative expert/input locality',
                  'execution':'HF eager BF16 SDPA dynamic KV; distinct from optimized vLLM/CUDA-graph baseline',
                  'address_manifest':'parameter_storage_manifest.json and kv_storage_lifecycle.json; allocation ranges not measured traffic; temporary/activation addresses unmapped',
                  'boundary':'hooks attached after all prefill; profiler API not used because upstream ACTIVE_FROM_START=0 bug',
                  'omitted':'non-model argmax/token copies and modules outside selected coverage',
                  'reference_uninstrumented_output_ids':reference_ids,'reference_sha256':digest(reference_ids),
                  'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'nvtx':importlib.metadata.version('nvtx')}}
        dump(args.output/'trace_metadata.json',metadata)
        kv_lifecycle=[]
        for step in range(args.steps):
            kv_lifecycle.append(hf_kv(past,f'before_decode_step_{step}'))
            torch.cuda.nvtx.range_push(f'decode_step_{step}')
            out=model(input_ids=token,past_key_values=past,use_cache=True,logits_to_keep=1)
            past=out.past_key_values
            token=out.logits[:,-1:].argmax(-1)
            generated.append(token.clone())
            del out
            kv_lifecycle.append(hf_kv(past,f'after_decode_step_{step}'))
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        hook_module.g_nvbit_tracer.disable_nvbit_instrumentation()
        dump(args.output/'kv_storage_lifecycle.json',kv_lifecycle)
        ids=torch.cat(generated,dim=1).cpu().tolist()
        metadata.update(status='complete' if ids==reference_ids else 'failed',output_ids=ids,output_sha256=digest(ids),correctness_passed=ids==reference_ids,final_cache_length=past.get_seq_length())
        assert metadata['final_cache_length']==args.context+args.steps
        dump(args.output/'trace_metadata.json',metadata)
        assert metadata['correctness_passed'],'instrumented HF output differs from uninstrumented eager greedy reference'
    nvml.nvmlShutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accel-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--layers',default='full',help='full model, or exact comma-separated disjoint module names')
    main(p.parse_args())
