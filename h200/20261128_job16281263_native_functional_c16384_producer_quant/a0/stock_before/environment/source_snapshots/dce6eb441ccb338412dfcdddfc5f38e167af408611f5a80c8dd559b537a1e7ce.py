#!/usr/bin/env python3
"""BF16 common-prefix cache/backend diagnosis; never redefines the exact-ID gate."""
import argparse
import hashlib
import json
from pathlib import Path


def first_difference(a,b):
    assert len(a)==len(b) and all(len(x)==len(y) for x,y in zip(a,b))
    return next(({'request':i,'token_index':j,'reference':x,'actual':y}
        for i,(left,right) in enumerate(zip(a,b)) for j,(x,y) in enumerate(zip(left,right)) if x!=y),None)


def main(a):
    import torch
    import transformers
    import pynvml as nv
    from gpu_target import select,verify_cuda
    from software_decode_experiments import controlled_trial,make_decode,diagnostic_logits,digest,dump
    nv.nvmlInit();select(nv,a.gpu_uuid);verify_cuda(torch,a.gpu_uuid)
    source=json.loads(a.source_run.read_text());assert source['status']=='complete'
    cfg=source['config'];case=source['cases'][0]
    assert case['batch_size']==1 and case['context_length']==4096 and case['generated_tokens']==64
    assert case['output_digests_identical']
    prompt=torch.tensor(case['prompt_ids'],device='cuda',dtype=torch.long)
    assert digest(case['prompt_ids'])==case['prompt_sha256']
    expected=case['trials'][0]['output_ids'];maximum=4096+64
    historical=json.loads(a.historical_hf_run.read_text())
    assert historical['status']=='complete' and historical['config']['model_revision']==cfg['model_revision']
    assert historical['cases'][0]['prompt_sha256']==case['prompt_sha256']
    original_hf=historical['cases'][0]['trials'][0]['output_ids']
    path=Path(cfg['model_path']);assert path.name==cfg['model_revision']
    report={'status':'running','classification':'correctness-only instrumented diagnostic',
        'source_run':str(a.source_run),'source_sha256':hashlib.sha256(a.source_run.read_bytes()).hexdigest(),
        'model_revision':cfg['model_revision'],'precision':'BF16 weights/KV','prompt_sha256':case['prompt_sha256'],
        'historical_hf_run':str(a.historical_hf_run),'historical_hf_sha256':hashlib.sha256(a.historical_hf_run.read_bytes()).hexdigest(),
        'seed':cfg['seed'],'routes':[],'exact_gate_relaxed':False,
        'scope':'three warmups/three diagnostic trials per HF route; no latency/energy result; backend/cache differences may change numerical execution'}
    torch.manual_seed(cfg['seed'])
    with torch.inference_mode():
        for backend in ('sdpa','eager'):
            model=transformers.AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
                dtype=torch.bfloat16,attn_implementation=backend).eval().to('cuda')
            for kind in ('static','dynamic'):
                warmups=[];trials=[]
                cache=transformers.StaticCache(config=model.config,max_cache_len=maximum) if kind=='static' else None
                decode=make_decode(model,cache,maximum,torch,False) if kind=='static' else None
                for index in range(6):
                    if kind=='static':ids=controlled_trial(model,cache,decode,prompt,64,torch)['output_ids']
                    else:
                        generated=model.generate(input_ids=prompt,max_new_tokens=64,do_sample=False,
                            eos_token_id=None,repetition_penalty=1.0,use_cache=True,
                            cache_implementation='dynamic',logits_to_keep=1)
                        ids=generated[:,4096:].cpu().tolist()
                    (warmups if index<3 else trials).append(ids)
                delta=first_difference(expected,trials[0])
                row={'backend':backend,'cache':kind,'warmup_output_ids':warmups,'trial_output_ids':trials,
                    'deterministic':all(x==trials[0] for x in warmups+trials),'exact_vllm_ids':all(x==expected for x in warmups+trials),
                    'exact_historical_static_ids':all(x==original_hf for x in warmups+trials),
                    'first_difference':delta}
                if delta:
                    k=delta['token_index'];prefix=expected[0][:k]
                    if kind=='static':logits=diagnostic_logits(model,cache,maximum,prompt,k,torch,False)
                    else:
                        dynamic=transformers.DynamicCache(config=model.config)
                        logits=model(input_ids=prompt,past_key_values=dynamic,use_cache=True,logits_to_keep=1).logits[:,-1]
                        for i,token in enumerate(prefix):
                            logits=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=dynamic,
                                cache_position=torch.tensor([4096+i],device='cuda'),use_cache=True,logits_to_keep=1).logits[:,-1]
                        logits=logits.float().cpu()
                    a.output.mkdir(parents=True,exist_ok=True)
                    tensor_file=a.output/f'{backend}_{kind}_first_difference_logits.pt'
                    torch.save(logits,tensor_file)
                    logp=logits.float().log_softmax(-1)[0];top=logp.topk(20)
                    vsteps=case['trials'][0].get('diagnostic_logprobs')
                    row['common_prefix_diagnostic']={'token_index':k,'prefix_ids':prefix,
                        'hf_top20':[{ 'id':int(token),'logprob':float(value)} for token,value in zip(top.indices,top.values)],
                        'hf_compared_logprobs':{str(t):float(logp[t]) for t in (delta['reference'],delta['actual'])},
                        'vllm_top_logprobs':vsteps[0][k] if vsteps else None,
                        'raw_hf_logits':tensor_file.name,'sha256':hashlib.sha256(tensor_file.read_bytes()).hexdigest()}
                report['routes'].append(row);dump(a.output/'diagnosis.json',report)
                del cache,decode
            del model;torch.cuda.empty_cache()
    report.update(status='complete',interpretation='Exact cache/backend comparisons and common-prefix margins retained. Agreement on a diagnostic route does not replace the original StaticCache/SDPA gate or validate a new timing engine.')
    dump(a.output/'diagnosis.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--historical-hf-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
