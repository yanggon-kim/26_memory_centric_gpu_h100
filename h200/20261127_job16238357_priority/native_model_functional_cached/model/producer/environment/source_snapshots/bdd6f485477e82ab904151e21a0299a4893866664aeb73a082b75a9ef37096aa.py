#!/usr/bin/env python3
"""Independent full-length HF dynamic and strict StaticCache output gates."""
import argparse
import hashlib
import json
from pathlib import Path
from h200_session import dump


def static_cache_gate(model,torch,transformers,prompt,case,reference,save):
    from software_decode_experiments import controlled_trial,make_decode
    maximum=case['context_length']+case['generated_tokens']
    cache=transformers.StaticCache(config=model.config,max_cache_len=maximum)
    decode=make_decode(model,cache,maximum,torch,False)
    result={'status':'running','adapter':'unchanged controlled_trial/make_decode; uncompiled default SDPA boolean mask',
            'adapter_sha256':hashlib.sha256((Path(__file__).parent/'software_decode_experiments.py').read_bytes()).hexdigest(),
            'maximum_cache_length':maximum,'warmups':[],'trials':[],
            'scope':'matching full shape/count correctness only; no timing, energy or compiled-HF claim'}
    save(result)
    with torch.inference_mode():
        for i in range(6):
            trial=controlled_trial(model,cache,decode,prompt,case['generated_tokens'],torch)
            row={'output_ids':trial['output_ids'],'output_sha256':trial['output_sha256'],
                 'cache_length':trial['cache_length'],'matches_dynamic_reference':trial['output_ids']==reference,
                 'matches_source':all(trial['output_ids']==t['output_ids'] for t in case['trials'])}
            result['warmups' if i<3 else 'trials'].append(row);save(result)
    result['passed']=all(t['matches_dynamic_reference'] and t['matches_source'] and
        t['cache_length']==case['context_length']+case['generated_tokens']-1 for t in result['warmups']+result['trials'])
    result['status']='passed' if result['passed'] else 'failed'
    result['first_differences']=[{'request':i,'token_index':j,'reference':ref,'actual':actual}
        for i,(a,b) in enumerate(zip(reference,result['trials'][0]['output_ids']))
        for j,(ref,actual) in enumerate(zip(a,b)) if ref!=actual][:20]
    save(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'reference output must be fresh'
    source=json.loads(args.source_run.read_text())
    assert source['status']=='complete'
    config=source['config'];case=source['cases'][0]
    import pynvml as nvml
    from gpu_target import select,verify_cuda
    nvml.nvmlInit();select(nvml,args.gpu_uuid)
    import torch
    import transformers
    from software_decode_experiments import digest
    verify_cuda(torch,args.gpu_uuid)
    path=Path(config['model_path'])
    assert path.name==config['model_revision']
    torch.manual_seed(config['seed'])
    model=transformers.AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
          dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
    prompt=torch.tensor(case['prompt_ids'],device='cuda')
    # Uniform synthetic IDs are active tokens, including an ID whose tokenizer
    # role is PAD. Inferring padding here would change the vLLM input contract.
    attention_mask=torch.ones_like(prompt)
    result={'classification':'cross-engine-token-correctness','status':'running',
            'hypothesis':'test whether full-length dynamic-cache model.generate agrees with the existing StaticCache and vLLM routes',
            'model_id':config['model_id'],'revision':config['model_revision'],'gpu_uuid':args.gpu_uuid,
            'config_sha256':config['config_sha256'],'source_run':str(args.source_run),
            'source_sha256':hashlib.sha256(args.source_run.read_bytes()).hexdigest(),
            'prompt_ids':case['prompt_ids'],'prompt_sha256':case['prompt_sha256'],
            'batch':case['batch_size'],'context':case['context_length'],'generated_tokens':case['generated_tokens'],
            'precision':'BF16','reference_program':'HF generate dynamic cache plus unchanged controlled StaticCache SDPA gate',
            'attention_mask_policy':'explicit all-ones; every supplied prompt ID is active',
            'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda},
            'scope':'correctness-only; no timing or energy claims','warmups':[],'trials':[]}
    with torch.inference_mode():
        for i in range(6):
            generated=model.generate(input_ids=prompt,attention_mask=attention_mask,max_new_tokens=case['generated_tokens'],
                  do_sample=False,eos_token_id=None,repetition_penalty=1.0,use_cache=True,
                  cache_implementation='dynamic',logits_to_keep=1)
            ids=generated[:,case['context_length']:].cpu().tolist()
            assert all(len(row)==case['generated_tokens'] for row in ids)
            target=result['warmups'] if i<3 else result['trials']
            target.append({'output_ids':ids,'output_sha256':digest(ids)})
            dump(args.output,result)
    hashes={trial['output_sha256'] for trial in result['warmups']+result['trials']}
    reference=result['trials'][0]['output_ids']
    comparisons=[]
    for index,trial in enumerate(case['trials']):
        differences=[{'request':i,'token_index':j,'reference':ref,'actual':actual}
                     for i,(a,b) in enumerate(zip(reference,trial['output_ids']))
                     for j,(ref,actual) in enumerate(zip(a,b)) if ref!=actual]
        comparisons.append({'trial':index,'identical':not differences,'difference_count':len(differences),
                            'first_differences':differences[:20]})
    dynamic_passed=len(hashes)==1 and all(x['identical'] for x in comparisons)
    def save_static(value):
        result['static_cache_gate']=value
        dump(args.output,result)
    static=static_cache_gate(model,torch,transformers,prompt,case,reference,save_static)
    result.update(status='passed' if dynamic_passed and static['passed'] else 'failed',dynamic_cache_gate_passed=dynamic_passed,
                  reference_outputs_identical=len(hashes)==1,source_comparisons=comparisons)
    if result['status']=='failed' and comparisons[0]['first_differences']:
        first=comparisons[0]['first_differences'][0]
        with torch.inference_mode():
            scored=model.generate(input_ids=prompt,attention_mask=attention_mask,max_new_tokens=case['generated_tokens'],do_sample=False,
                   eos_token_id=None,repetition_penalty=1.0,use_cache=True,cache_implementation='dynamic',
                   logits_to_keep=1,return_dict_in_generate=True,output_scores=True)
        score=scored.scores[first['token_index']][first['request']].float().cpu()
        top=score.topk(8)
        scored_ids=scored.sequences[:,case['context_length']:].cpu().tolist()
        result['first_difference_score_diagnostic']={
            'first_difference':first,'scored_call_ids_match_reference':scored_ids==reference,
            'top8':[{'token':int(i),'score':float(v)} for i,v in zip(top.indices,top.values)],
            'top2_margin':float(top.values[0]-top.values[1]),
            'reference_token_score':float(score[first['reference']]),
            'source_token_score':float(score[first['actual']]),
            'scope':'separate HF dynamic generate with returned scores; allocation can change; no tolerance or gate relaxation'}
    dump(args.output,result)
    nvml.nvmlShutdown()
    print(json.dumps({'status':result['status'],'reference_outputs_identical':result['reference_outputs_identical'],
                      'dynamic_cache_gate_passed':dynamic_passed,'static_cache_gate_passed':static['passed'],
                      'static_first_differences':static['first_differences'][:2],
                      'first_differences':comparisons[0]['first_differences'][:2]},indent=2))
    raise SystemExit(0 if result['status']=='passed' else 1)


if __name__=='__main__':main()
