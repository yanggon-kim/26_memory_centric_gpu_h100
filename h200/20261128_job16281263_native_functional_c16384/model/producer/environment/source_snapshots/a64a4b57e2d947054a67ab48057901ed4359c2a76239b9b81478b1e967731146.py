#!/usr/bin/env python3
"""Real all-layer HF BF16 KV handoff between allocated H200s, correctness pilot.

Uses the pinned transformers4.57.6 DynamicCache legacy round trip:
https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/cache_utils.py
This is not a vLLM connector, open-loop serving result, or physical-link bandwidth measurement.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback


def verify_cache_transfer(source, receiver, expected_layers, dtype, equal):
    assert len(source)==len(receiver)==expected_layers, 'incomplete layer coverage'
    rows=[]
    for layer,(before,after) in enumerate(zip(source,receiver)):
        assert len(before)==len(after)==2
        for role,s,d in zip(('key','value'),before,after):
            assert s.dtype==d.dtype==dtype, 'KV precision changed'
            assert s.device.type==d.device.type=='cuda' and s.device.index==0 and d.device.index==1
            assert tuple(s.shape)==tuple(d.shape) and s.ndim==4, 'KV layout/shape mismatch'
            assert equal(s.cpu(),d.cpu()), 'transferred cache differs bitwise'
            rows.append({'layer':layer,'role':role,'shape':list(s.shape),'source_stride':list(s.stride()),
                         'receiver_stride':list(d.stride()),'dtype':str(s.dtype),
                         'payload_bytes':s.numel()*s.element_size()})
    return rows


def decode(model, cache, first, context, count, device, torch):
    selected=[first]
    for step in range(1,count):
        assert selected[-1].shape[1]==1 and cache.get_seq_length()==context+step-1
        result=model(input_ids=selected[-1],past_key_values=cache,use_cache=True,logits_to_keep=1,
                     attention_mask=torch.ones((1,context+step),device=device,dtype=torch.long),
                     cache_position=torch.tensor([context+step-1],device=device))
        cache=result.past_key_values
        assert cache.get_seq_length()==context+step
        selected.append(result.logits[:,-1,:].argmax(dim=-1,keepdim=True))
    return torch.cat(selected,dim=1).cpu().tolist()


def main(a):
    from h200_session import dump
    from h200_p2p_smoke import paired_scope
    cfg=json.loads(a.config.read_text());path=Path(cfg['model_path'])
    assert path.name==cfg['model_revision']
    architecture=json.loads((path/'config.json').read_text())
    assert architecture['model_type']=='qwen2' and not architecture.get('use_sliding_window',False)
    assert a.context+a.generated_tokens<=architecture['max_position_embeddings']
    state=None
    data={'status':'running','classification':'measured-real-GPU-functional-pilot',
          'real_KV_handoff_verified':False,'engine':'HF dynamic SDPA, BF16, one full model per GPU',
          'vllm_connector_validated':False,'serving_comparison':False,'physical_link_bytes':None,
          'physical_link_bytes_reason':'logical transferred tensor payload is recorded; physical link counters are not qualified',
          'primary_transfer_latency_ms':None,'latency_reason':'functional pilot; raw transfer/rebuild observations are not qualified timing comparisons',
          'model_id':cfg['model_id'],'model_revision':cfg['model_revision'],'model_config_sha256':cfg['config_sha256'],
          'context':a.context,'generated_tokens':a.generated_tokens,'seed':20261013,'warmups':3,'trials':[],
          'receiver_prefill_executed':False,'scope':'actual whole-model KV generated on GPU0, copied and reconstructed on GPU1; receiver runs one-token decode only',
          'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                           for name in ('h200_real_kv_smoke.py','h200_p2p_smoke.py','h200_session.py')},
          'cache_API_source':'https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/cache_utils.py'}
    dump(a.output,data)
    try:
        with paired_scope(a.devices,a.output.with_suffix('.monitor.jsonl')) as state:
            torch=state['torch']
            torch.manual_seed(20261013)
            import transformers
            from transformers import DynamicCache
            assert transformers.__version__=='4.57.6'
            data['cache_implementation_sha256']=hashlib.sha256(Path(inspect.getsourcefile(DynamicCache)).read_bytes()).hexdigest()
            data.update(devices=state['uuids'],peer_access=state['peer_access'],
                        versions={'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda})
            models=[transformers.AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
                    dtype=torch.bfloat16,attn_implementation='sdpa').eval().to(f'cuda:{i}') for i in (0,1)]
            assert all(p.dtype==torch.bfloat16 for model in models for p in model.parameters()), 'weight precision mismatch'
            generator=torch.Generator().manual_seed(20261013)
            prompt_cpu=torch.randint(0,architecture['vocab_size'],(1,a.context),generator=generator)
            prompt=prompt_cpu.to('cuda:0')
            data['prompt_ids']=prompt_cpu.tolist()
            data['prompt_sha256']=hashlib.sha256(json.dumps(data['prompt_ids'],separators=(',',':')).encode()).hexdigest()
            with torch.inference_mode():
                gold=models[0].generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
                    max_new_tokens=a.generated_tokens,do_sample=False,eos_token_id=None,repetition_penalty=1.,
                    use_cache=True,cache_implementation='dynamic',logits_to_keep=1)[:,a.context:].cpu().tolist()
                data['independent_generate_output_ids']=gold
                for trial in range(6):
                    prefill=models[0](input_ids=prompt,attention_mask=torch.ones_like(prompt),use_cache=True,logits_to_keep=1)
                    cache=prefill.past_key_values
                    assert cache.get_seq_length()==a.context
                    first=prefill.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                    legacy=cache.to_legacy_cache()
                    assert len(legacy)==architecture['num_hidden_layers']
                    assert all(t.shape[-2]==a.context for pair in legacy for t in pair)
                    assert all(t.dtype==torch.bfloat16 and t.device.index==0 for pair in legacy for t in pair)
                    torch.cuda.synchronize(0);torch.cuda.synchronize(1)
                    capture=a.profile and trial==3
                    if capture:torch.cuda.profiler.start()
                    with torch.cuda.device(1):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        torch.cuda.nvtx.range_push('real_hf_kv_handoff')
                        begin.record();start=time.perf_counter()
                        moved=tuple(tuple(t.to('cuda:1',non_blocking=True) for t in pair) for pair in legacy)
                        incoming_first=first.to('cuda:1',non_blocking=True)
                        torch.cuda.synchronize(1)
                        copy_ms=(time.perf_counter()-start)*1000
                        rebuild_start=time.perf_counter()
                        receiver=DynamicCache.from_legacy_cache(moved)
                        torch.cuda.synchronize(1)
                        rebuild_ms=(time.perf_counter()-rebuild_start)*1000
                        end.record();end.synchronize()
                        event_ms=begin.elapsed_time(end)
                        torch.cuda.nvtx.range_pop()
                    if capture:torch.cuda.profiler.stop()
                    layouts=verify_cache_transfer(legacy,receiver.to_legacy_cache(),architecture['num_hidden_layers'],
                                                   torch.bfloat16,torch.equal)
                    assert torch.equal(first.cpu(),incoming_first.cpu()) and receiver.get_seq_length()==a.context
                    control_ids=decode(models[0],cache,first,a.context,a.generated_tokens,'cuda:0',torch)
                    receiver_ids=decode(models[1],receiver,incoming_first,a.context,a.generated_tokens,'cuda:1',torch)
                    row={'trial':trial,'warmup':trial<3,'correctness':'passed' if control_ids==receiver_ids==gold else 'failed',
                         'control_output_ids':control_ids,'receiver_output_ids':receiver_ids,
                         'copied_KV_payload_bytes':sum(r['payload_bytes'] for r in layouts),'first_token_payload_bytes':first.numel()*first.element_size(),
                         'raw_copy_wall_ms':copy_ms,'raw_rebuild_wall_ms':rebuild_ms,'raw_event_ms':event_ms,
                         'instrumented_copy':capture,'source_cache_initial_length':a.context,'receiver_cache_initial_length':a.context,
                         'receiver_decode_queries':[1]*(a.generated_tokens-1)}
                    data['cache_layouts']=layouts;data['trials'].append(row);dump(a.output,data)
                    assert row['correctness']=='passed', 'exact transferred/control/independent-generate outputs differ'
                data.update(status='complete',real_KV_handoff_verified=True)
    except BaseException as error:
        data.update(status='failed',real_KV_handoff_verified=False,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if state is not None:data['monitor_failures']=state['failures']
        dump(a.output,data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=int,default=128)
    p.add_argument('--generated-tokens',type=int,default=64);p.add_argument('--profile',action='store_true')
    a=p.parse_args();assert a.context>0 and 2<=a.generated_tokens<=64
    a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists()
    main(a)
