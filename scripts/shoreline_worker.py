"""Operation ranges and disjoint complete-layer events; no metadata GPU copies."""
import re

ACTIVE_WORKERS=[]


def install_ranges(worker):
    import torch
    worker._shoreline_tick=None
    worker._shoreline_events=[]
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
            torch.cuda.nvtx.range_push(label)
            if whole and worker._shoreline_tick in (4,5,6) and getattr(worker,'_shoreline_enable_events',False):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                worker._shoreline_events.append({'layer':layer,'tick':worker._shoreline_tick,'start':start,'end':end,'closed':False})
        def after(module,arguments,output,label=label,layer=layer,whole=is_complete_layer):
            if whole and worker._shoreline_tick in (4,5,6) and getattr(worker,'_shoreline_enable_events',False):
                entry=worker._shoreline_events[-1]
                assert entry['layer']==layer and not entry['closed']
                entry['end'].record();entry['closed']=True
            torch.cuda.nvtx.range_pop()
        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)
        names.append(label)
    return {'ranges':names,'selected_complete_layers':sorted(selected),
            'scope':'original module forwards; graph replay bypasses Python hooks'}


def read_events(worker):
    import torch
    torch.cuda.synchronize()
    assert all(e['closed'] for e in worker._shoreline_events)
    return [{'layer':e['layer'],'tick':e['tick'],'elapsed_ms':e['start'].elapsed_time(e['end'])}
            for e in worker._shoreline_events]


def enable_events(worker):
    worker._shoreline_enable_events=True
    return {'enabled':True}
