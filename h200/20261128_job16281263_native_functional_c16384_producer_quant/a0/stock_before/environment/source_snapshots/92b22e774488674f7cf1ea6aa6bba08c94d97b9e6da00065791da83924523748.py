"""Runtime allocated ranges and aliases; allocation bytes are never access counts."""

def record(name,tensor):
    storage=tensor.untyped_storage()
    base=storage.data_ptr()
    return {'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device),
            'data_ptr':tensor.data_ptr(),'storage_base':base,'storage_nbytes':storage.nbytes(),
            'storage_range_end_exclusive':base+storage.nbytes(),'storage_offset_elements':tensor.storage_offset(),
            'stride_elements':list(tensor.stride()),'contiguous':tensor.is_contiguous(),
            'logical_tensor_bytes':tensor.numel()*tensor.element_size(),'storage_identity':f'{tensor.device}:{base}:{storage.nbytes()}'}


def parameters(model):
    rows=[record(name,tensor) for name,tensor in model.named_parameters(remove_duplicate=False)]
    aliases={}
    for row in rows:aliases.setdefault(row['storage_identity'],[]).append(row['name'])
    for row in rows:row['storage_aliases']=aliases[row['storage_identity']]
    return {'tensors':rows,'scope':'parameter tensors including tied/shared aliases and actual allocated ranges; not accessed traffic; packed kernel weights may differ from checkpoint naming',
            'unmapped':'activation/temporary/allocator/workspace ranges and semantic attribution among aliases require additional runtime mapping'}


def walk(value,name):
    if hasattr(value,'untyped_storage'):
        return [record(name,value)]
    if isinstance(value,(list,tuple)):
        return [row for i,item in enumerate(value) for row in walk(item,f'{name}[{i}]')]
    if isinstance(value,dict):
        return [row for key,item in value.items() for row in walk(item,f'{name}.{key}')]
    return []


def hf_kv(cache,phase):
    rows=[]
    for index,layer in enumerate(getattr(cache,'layers',[])):
        for kind in ['keys','values']:
            tensor=getattr(layer,kind,None)
            if tensor is not None:rows+=walk(tensor,f'layer{index}.{kind}')
    return {'phase':phase,'tensors':rows,'scope':'allocated dynamic KV before/after forward; storage can move; exact per-step lifetime snapshots, not accessed bytes'}


def vllm_kv(worker,phase):
    runner=worker.model_runner
    rows=walk(getattr(runner,'kv_caches',None),'model_runner.kv_caches')
    context=getattr(getattr(runner,'compilation_config',None),'static_forward_context',{})
    for name,layer in context.items():
        if hasattr(layer,'kv_cache'):rows+=walk(layer.kv_cache,'attention_layer.'+name+'.kv_cache')
    config=getattr(runner,'kv_cache_config',None)
    groups=[]
    for index,group in enumerate(getattr(config,'kv_cache_groups',[])):
        spec=group.kv_cache_spec
        fields={key:str(getattr(spec,key)) if key=='dtype' else getattr(spec,key) for key in
                ['block_size','page_size_bytes','num_kv_heads','head_size','dtype'] if hasattr(spec,key)}
        groups.append({'group':index,'layer_names':list(group.layer_names),'spec_class':type(spec).__name__,'observed_spec':fields})
    return {'phase':phase,'tensors':rows,'num_blocks':getattr(config,'num_blocks','NA'),'groups':groups,
            'scope':'actual allocated paged-KV ranges/shapes/strides plus runtime group/page specs; axes not guessed',
            'unmapped':'request-to-page block tables/slot mappings and transient activation/workspace addresses not yet exported; H100 smoke must extend mapping before semantic request attribution'}
