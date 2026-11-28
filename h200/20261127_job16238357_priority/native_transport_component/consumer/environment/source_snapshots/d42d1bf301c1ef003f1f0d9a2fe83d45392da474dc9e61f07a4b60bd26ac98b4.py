#!/usr/bin/env python3
"""Bounded, forced-common-prefix BF16 diagnosis. Never replaces strict smoke gates."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path


def discrepancy(expected, actual):
    assert len(expected) == len(actual) and expected
    assert all(len(a) == len(b) for a, b in zip(expected, actual))
    return next(({'request':i, 'token_index':k, 'reference':a[k], 'actual':b[k]}
                 for k in range(len(expected[0]))
                 for i, (a,b) in enumerate(zip(expected,actual)) if a[k] != b[k]), None)


def forced_logits(model, cache, decode, prompt, expected, index, torch):
    """Every request consumes its reference prefix, including at divergent routes."""
    if cache is not None:
        cache.reset()
    logits = model(input_ids=prompt, past_key_values=cache,
                   cache_position=torch.arange(prompt.shape[1],device='cuda'),
                   use_cache=True,logits_to_keep=1)
    active = logits.past_key_values
    scores = logits.logits[:, -1]
    for k in range(index):
        token = torch.tensor([[row[k]] for row in expected],device='cuda',dtype=torch.long)
        position = torch.tensor([prompt.shape[1]+k],device='cuda',dtype=torch.long)
        if cache is not None:
            _, values = decode(token,position)
            scores = values[:, -1]
        else:
            output = model(input_ids=token,past_key_values=active,cache_position=position,
                           use_cache=True,logits_to_keep=1)
            active = output.past_key_values
            scores = output.logits[:, -1]
    return scores.float().cpu()


def main(a):
    import torch
    import transformers
    import pynvml as nv
    from gpu_target import select,verify_cuda
    from software_decode_experiments import controlled_trial,make_decode,digest,dump
    source=json.loads(a.source_run.read_text());assert source['status']=='complete'
    cfg=source['config'];case=source['cases'][0]
    assert case['batch_size'] in (1,8) and case['context_length']==4096 and case['generated_tokens']==64
    expected=case['trials'][0]['output_ids']
    assert all(t['output_ids']==expected for t in case['trials']) and len(case['trials'])>=3
    assert all(x==expected for x in case.get('warmup_output_ids',[]))
    nv.nvmlInit();select(nv,a.gpu_uuid);verify_cuda(torch,a.gpu_uuid)
    torch.manual_seed(cfg['seed'])
    prompt=torch.tensor(case['prompt_ids'],device='cuda',dtype=torch.long)
    assert digest(case['prompt_ids'])==case['prompt_sha256']
    path=Path(cfg['model_path']);assert path.name==cfg['model_revision']
    maximum=case['context_length']+64
    a.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','protocol':'common_prefix_v4','classification':'correctness-only GPU diagnostic',
        'source':str(a.source_run),'source_sha256':hashlib.sha256(a.source_run.read_bytes()).hexdigest(),
        'prompt_sha256':case['prompt_sha256'],'model_revision':cfg['model_revision'],'batch':len(expected),
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda},
        'precision':'BF16 weights/KV','exact_gate_relaxed':False,'routes':[],
        'scope':'3 warmups + 3 diagnostic trials/route; no timing claims; no alternate route promotes the original strict gate'}
    routes=[('sdpa','static','boolean','auto'),('sdpa','static','builtin','auto'),
            ('sdpa','dynamic','builtin','auto'),('eager','static','additive','auto'),
            ('eager','dynamic','builtin','auto'),('sdpa','static','builtin','math')]
    with torch.inference_mode():
        for backend,kind,mask,dispatch in routes:
            label='_'.join((backend,kind,mask,dispatch))
            row={'id':label,'status':'running','backend':backend,'cache':kind,'mask':mask,'dispatch':dispatch}
            report['routes'].append(row);dump(a.output/'diagnosis.json',report)
            # A fresh model/cache per route avoids contamination across interventions.
            model=transformers.AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
                dtype=torch.bfloat16,attn_implementation=backend).eval().to('cuda')
            cache=transformers.StaticCache(config=model.config,max_cache_len=maximum) if kind=='static' else None
            decode=make_decode(model,cache,maximum,torch,False,return_logits=True,mask_kind=mask) if cache is not None else None
            plain=make_decode(model,cache,maximum,torch,False,mask_kind=mask) if cache is not None else None
            context=(lambda:torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)) if dispatch=='math' else nullcontext
            with context():
                outputs=[]
                for _ in range(6):
                    if cache is not None:
                        ids=controlled_trial(model,cache,plain,prompt,64,torch)['output_ids']
                    else:
                        ids=model.generate(input_ids=prompt,max_new_tokens=64,do_sample=False,
                            eos_token_id=None,repetition_penalty=1.0,use_cache=True,
                            cache_implementation='dynamic',logits_to_keep=1)[:,case['context_length']:].cpu().tolist()
                    outputs.append(ids)
                delta=discrepancy(expected,outputs[3])
                row.update(status='complete',warmup_output_ids=outputs[:3],trial_output_ids=outputs[3:],
                    deterministic=all(x==outputs[0] for x in outputs),exact_vllm_ids=all(x==expected for x in outputs),first_difference=delta)
                probe=delta or {'token_index':5,'request':0,'reference':expected[0][5],'actual':outputs[3][0][5]}
                # CPU operator events distinguish actual SDPA dispatch from a requested label.
                # Profiling is confined to an untimed diagnostic, after all free-running trials.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    scores=forced_logits(model,cache,decode,prompt,expected,probe['token_index'],torch)
                row['loaded_attention_backend']=model.config._attn_implementation
                row['observed_attention_operators']=sorted({event.key for event in trace.key_averages()
                    if any(name in event.key.lower() for name in ('scaled_dot_product','flash_attention','efficient_attention'))})
                target=a.output/(label+'_common_prefix_logits.pt');torch.save(scores,target)
                logp=scores.log_softmax(-1);top=logp[probe['request']].topk(20)
                vsteps=case['trials'][0].get('diagnostic_logprobs')
                row['common_prefix']={'token_index':probe['token_index'],'request':probe['request'],
                    'selection':'first differing token' if delta else 'predeclared token index5 for matching routes',
                    'forced_prefix_ids':[x[:probe['token_index']] for x in expected],
                    'hf_top20':[{'id':int(t),'logprob':float(v)} for t,v in zip(top.indices,top.values)],
                    'hf_compared_logprobs':{str(t):float(logp[probe['request'],t]) for t in (probe['reference'],probe['actual'])},
                    'vllm_top_logprobs':vsteps[probe['request']][probe['token_index']] if vsteps else None,
                    'raw_logits':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            dump(a.output/'diagnosis.json',report)
            del cache,decode,plain,model;torch.cuda.empty_cache()
    report['status']='complete';dump(a.output/'diagnosis.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
