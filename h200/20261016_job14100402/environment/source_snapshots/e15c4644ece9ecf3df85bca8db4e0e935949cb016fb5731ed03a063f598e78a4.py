"""Operation ranges and disjoint complete-layer events; no metadata GPU copies."""
import re

ACTIVE_WORKERS=[]


def install_ranges(worker):
    import inspect
    import os
    import torch
    worker._shoreline_tick=None
    worker._shoreline_events=[]
    worker._shoreline_inputs=[]
    worker._shoreline_pressure_ledger=[]
    pressure=os.environ.get('SHORELINE_PRESSURE')
    if pressure in ('normal','pressure'):
        generator=torch.Generator(device='cuda').manual_seed(20261013)
        worker._shoreline_pressure=torch.rand((256*1024**2//4,),generator=generator,device='cuda')
        worker._shoreline_pressure_output=torch.empty((),device='cuda')
    ACTIVE_WORKERS.append(worker)
    total=worker.model_runner.model.config.num_hidden_layers
    selected={0,total//2,total-1}
    names=[]
    for name,module in worker.model_runner.model.named_modules():
        match=re.fullmatch(r'model\.layers\.(\d+)(?:\.(self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj)))?',name)
        if not match:continue
        layer=int(match.group(1));operation=match.group(2)
        label=f'operation_layer_{layer}_'+operation.replace('.','_') if operation else f'layer_{layer}'
        is_complete_layer=operation is None and layer in selected
        def before(module,arguments,label=label,layer=layer,whole=is_complete_layer):
            if whole and layer==0 and worker._shoreline_tick in (4,5,6) and pressure=='pressure':
                import time
                torch.sum(worker._shoreline_pressure,dim=0,out=worker._shoreline_pressure_output)
                torch.cuda.synchronize()
                worker._shoreline_pressure_ledger.append({'tick':worker._shoreline_tick,
                    'trial':getattr(worker,'_shoreline_prefix',None),'completed_unix_ns':time.time_ns(),
                    'buffer_bytes':256*1024**2,'placement':'complete before layer_0 NVTX push; charged inside full decode'})
            torch.cuda.nvtx.range_push(label)
            if whole and worker._shoreline_tick in (4,5,6) and getattr(worker,'_shoreline_enable_events',False):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                worker._shoreline_events.append({'layer':layer,'tick':worker._shoreline_tick,
                    'trial':getattr(worker,'_shoreline_prefix',None),'start':start,'end':end,'closed':False})
        def after(module,arguments,output,label=label,layer=layer,whole=is_complete_layer):
            if whole and worker._shoreline_tick in (4,5,6) and getattr(worker,'_shoreline_enable_events',False):
                entry=worker._shoreline_events[-1]
                assert entry['layer']==layer and not entry['closed']
                entry['end'].record();entry['closed']=True
            torch.cuda.nvtx.range_pop()
        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)
        if operation is None and layer in selected:
            signature=inspect.signature(module.forward)
            def inputs(module,args,kwargs,layer=layer,signature=signature):
                if worker._shoreline_tick not in (4,5,6):return
                from tensor_manifest import record
                bound=signature.bind_partial(*args,**kwargs)
                tensors=[record(key,v) for key,v in bound.arguments.items() if hasattr(v,'untyped_storage')]
                worker._shoreline_inputs.append({'layer':layer,'tick':worker._shoreline_tick,'tensors':tensors,
                    'value_versions':'input values are distinct across ticks even when allocation addresses recur'})
            module.register_forward_pre_hook(inputs,with_kwargs=True)
        names.append(label)
    return {'ranges':names,'selected_complete_layers':sorted(selected),
            'scope':'original module forwards; graph replay bypasses Python hooks'}


def read_events(worker):
    import torch
    torch.cuda.synchronize()
    assert all(e['closed'] for e in worker._shoreline_events)
    return [{'layer':e['layer'],'tick':e['tick'],'trial':e['trial'],'elapsed_ms':e['start'].elapsed_time(e['end'])}
            for e in worker._shoreline_events]


def read_metadata(worker):
    from tensor_manifest import parameters,vllm_kv
    from h200_worker_metadata import snapshot_worker
    return {'layer_inputs':getattr(worker,'_shoreline_inputs',[]),
            'parameter_storage':parameters(worker.model_runner.model),
            'kv_storage':vllm_kv(worker,'after_full_output_validation'),
            'final_cpu_request_page_state':snapshot_worker(worker),
            'pressure_ledger':getattr(worker,'_shoreline_pressure_ledger',[]),
            'pressure_allocation':{'bytes':worker._shoreline_pressure.numel()*4,
                 'data_ptr':worker._shoreline_pressure.data_ptr()} if hasattr(worker,'_shoreline_pressure') else None,
            'scope':'CPU tensor layout/alias metadata; allocations do not establish accessed/HBM traffic',
            'layer_events':read_events(worker)}


def enable_events(worker):
    worker._shoreline_enable_events=True
    return {'enabled':True}
