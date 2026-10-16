"""Read actual worker backend, storage and request/page state in profiled controls."""
import hashlib
import inspect
from pathlib import Path
from tensor_manifest import parameters,vllm_kv


def install_layer_ranges(worker):
    import re
    import torch
    names=[]
    for name,module in worker.model_runner.model.named_modules():
        match=re.fullmatch(r'model\.layers\.(\d+)',name)
        if not match:continue
        label='layer_'+match.group(1)
        def before(module,arguments,label=label):
            torch.cuda.nvtx.range_push(label)
        def after(module,arguments,output):
            torch.cuda.nvtx.range_pop()
        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)
        names.append(label)
    return names


def install_attention_observer(worker):
    candidates=[(name,module.impl) for name,module in worker.model_runner.model.named_modules()
                if getattr(module,'impl',None) is not None and hasattr(module.impl,'num_heads')]
    worker._h200_attention_samples=[]
    selected=sorted(set([0,len(candidates)//2,len(candidates)-1])) if candidates else []
    for index in selected:
        name,impl=candidates[index]
        original=impl.forward
        signature=inspect.signature(original)
        def observed(*args,_original=original,_signature=signature,_name=name,**kwargs):
            result=_original(*args,**kwargs)
            bound=_signature.bind_partial(*args,**kwargs)
            metadata=bound.arguments.get('attn_metadata')
            if metadata is None or metadata.max_query_len!=1:return result
            samples=worker._h200_attention_samples
            if sum(row['module']==_name for row in samples)>=3:return result
            row={'module':_name,'num_actual_tokens':metadata.num_actual_tokens,
                 'max_query_len':metadata.max_query_len,'max_seq_len':metadata.max_seq_len,
                 'causal':metadata.causal,'use_cascade':metadata.use_cascade,
                 'max_num_splits':metadata.max_num_splits,
                 'split_bound_meaning':'zero delegates to FA3 heuristics; it does not mean zero dynamic splits',
                 'per_request_dynamic_splits':None,
                 'dynamic_split_reason':'raw scheduler metadata retained; field decoder not independently validated',
                 'productive_empty_work':None,'productive_empty_work_reason':'not inferred from launch grid'}
            for key in ('query_start_loc','seq_lens','scheduler_metadata','block_table'):
                value=getattr(metadata,key,None)
                row[key]=value.detach().cpu().tolist() if value is not None else None
            for key in ('query','key','value','kv_cache'):
                value=bound.arguments.get(key)
                if value is not None:
                    row[key]={'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype)}
            samples.append(row)
            return result
        impl.forward=observed
    return [candidates[index][0] for index in selected]


def read_attention_observer(worker):
    return getattr(worker,'_h200_attention_samples',[])


def describe_worker(worker):
    model=worker.model_runner.model
    attention=[]
    for name,module in model.named_modules():
        impl=getattr(module,'impl',None)
        if impl is None or not hasattr(impl,'num_heads'):continue
        source=inspect.getsourcefile(type(impl))
        row={'module':name,'implementation':type(impl).__module__+'.'+type(impl).__name__,
             'source':source,'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None}
        for key in ('num_heads','num_kv_heads','head_size','kv_cache_dtype','vllm_flash_attn_version',
                    'sliding_window','num_queries_per_kv'):
            value=getattr(impl,key,None)
            row[key]=value if isinstance(value,(int,float,str,bool,list,tuple,type(None))) else str(value)
        attention.append(row)
    return {'attention':attention,'parameter_storage':parameters(model),
            'kv_storage':vllm_kv(worker,'model_loaded'),
            'tiles_and_dynamic_splits':None,
            'tiles_and_dynamic_splits_unavailable_reason':'backend identity and kernel specialization retained; dynamic scheduler fields require a separate observer',
            'scope':'allocated storage and implementation identity; not measured accessed or DRAM bytes'}


def snapshot_worker(worker):
    batch=worker.model_runner.input_batch
    n=batch.num_reqs
    rows=[]
    for index,rid in enumerate(batch.req_ids[:n]):
        computed=int(batch.num_computed_tokens_cpu[index])
        tokens=int(batch.num_tokens_no_spec[index])
        rows.append({'request_id':rid,'row':index,'num_computed_tokens':computed,
                     'num_tokens_without_speculation':tokens,
                     'cpu_counter_difference':tokens-computed,
                     'counter_scope':'last prepared input batch; CPU computed counters can lag execution; actual attention metadata supplies executed query/KV lengths'})
    groups=[]
    for index,table in enumerate(batch.block_table.block_tables):
        groups.append({'group':index,'kernel_block_size':table.block_size,
                       'blocks_per_kv_block':table.blocks_per_kv_block,
                       'rows':[{'request_id':row['request_id'],
                                'allocated_block_count':int(table.num_blocks_per_row[row['row']]),
                                'block_ids':table.block_table.np[row['row'],:int(table.num_blocks_per_row[row['row']])].tolist()}
                               for row in rows]})
    return {'requests':rows,'kv_page_groups':groups,
            'scope':'actual last-prepared CPU input counters/page rows before engine.step updates them; allocation is not traffic; executed lengths/page tables are in attention observations'}
