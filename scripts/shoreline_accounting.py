"""Analytical useful inputs from immutable headers, separated from measured DRAM."""
import json
import math
from pathlib import Path
import struct
import hashlib


def archive_headers(root, model, config):
    """Keep immutable analytical inputs after node-local checkpoint eviction."""
    snapshot=Path(config['model_path'])
    target=Path(root)/'environment/model_headers'/(model+'_v2.json')
    if target.exists():
        data=json.loads(target.read_text())
        assert data['model_revision']==config['model_revision']
        assert data['model_id']==config['model_id']
        assert data['model_config_sha256']==hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest()
        return target
    data={'schema_version':2,'classification':'checkpoint headers; analytical inputs, not traffic',
          'model_id':config['model_id'],'model_revision':config['model_revision'],
          'model_config_sha256':hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest(),
          'model_config_json_text':(snapshot/'config.json').read_text(),
          'model_config':json.loads((snapshot/'config.json').read_text()),
          'safetensors_headers':headers(snapshot)}
    download=Path(root)/'configs'/(model+'_smoke.download_manifest.json')
    assert download.exists(), 'header provenance requires pinned checkpoint download manifest'
    proof=json.loads(download.read_text())
    assert proof['revision']==config['model_revision'] and proof['model_id']==config['model_id']
    data['download_manifest']={'path':str(download.relative_to(root)),
        'sha256':hashlib.sha256(download.read_bytes()).hexdigest()}
    data['checkpoint_files']=proof['files']
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,indent=2)+'\n')
    return target


def headers(snapshot):
    snapshot=Path(snapshot)
    names=['model.safetensors'] if (snapshot/'model.safetensors').exists() else sorted(set(json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map'].values()))
    rows=[]
    for name in names:
        with (snapshot/name).open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            assert length<128*1024**2,'invalid safetensors header length'
            metadata=json.loads(f.read(length))
        for tensor,v in metadata.items():
            if tensor=='__metadata__':continue
            a,b=v['data_offsets'];assert b>a
            rows.append({'name':tensor,'shape':v['shape'],'dtype':v['dtype'],'bytes':b-a,
                         'checkpoint_range':[name,a,b]})
    return rows


def useful(snapshot,batch,context,layer=None,input_bytes=None,input_tokens=None):
    snapshot=Path(snapshot)
    if snapshot.is_file():
        manifest=json.loads(snapshot.read_text())
        assert manifest['schema_version']==2 and manifest['model_revision']
        assert hashlib.sha256(manifest['model_config_json_text'].encode()).hexdigest()==manifest['model_config_sha256']
        assert json.loads(manifest['model_config_json_text'])==manifest['model_config']
        config=manifest['model_config'];tensors=manifest['safetensors_headers']
    else:
        config=json.loads((snapshot/'config.json').read_text());tensors=headers(snapshot)
    assert config['model_type'] in ('qwen2','mistral'),'dense accounting adapter only'
    active_sliding=(config.get('sliding_window') is not None and config.get('use_sliding_window',True)) or any('sliding' in x for x in config.get('layer_types',[]))
    assert not active_sliding,'active sliding window requires a separately validated adapter'
    assert all(t['dtype']=='BF16' for t in tensors),'BF16 checkpoint accounting required'
    hq=config['num_attention_heads'];hkv=config.get('num_key_value_heads',hq)
    d=config.get('head_dim') or config['hidden_size']//hq
    layers=config['num_hidden_layers'] if layer is None else 1
    selected=[t for t in tensors if t['name'].startswith(f'model.layers.{layer}.')] if layer is not None else [t for t in tensors if t['name']!='model.embed_tokens.weight']
    tied=config.get('tie_word_embeddings',False) and not any(t['name']=='lm_head.weight' for t in tensors)
    if layer is None and tied:selected += [t for t in tensors if t['name']=='model.embed_tokens.weight']
    w=sum(t['bytes'] for t in selected)
    linear_elems=sum(math.prod(t['shape']) for t in selected if len(t['shape'])==2 and 'embed_tokens' not in t['name'])
    if layer is None and tied:
        linear_elems+=sum(math.prod(t['shape']) for t in selected if 'embed_tokens' in t['name'])
    rows=[]
    for tick in (4,5,6):
        historical=layers*2*batch*(context+tick-1)*hkv*d*2
        entry=0 if layer is None else input_bytes.get(tick) if input_bytes else None
        lookup=0 if layer is not None or tied else (len(set(input_tokens[tick]))*config['hidden_size']*2 if input_tokens else None)
        payload=w+historical+(entry or 0)+(lookup or 0)
        rows.append({'tick':tick,'required_weight_bytes':w,'historical_kv_bytes':historical,
            'layer_entry_value_bytes':entry,'untied_embedding_lookup_bytes':lookup,
            'U_main_bytes':payload if (entry is not None and lookup is not None) else None,
            'known_main_input_subtotal_bytes':payload,
            'useful_linear_flops':2*batch*linear_elems,
            'useful_attention_flops':layers*4*batch*hq*d*(context+tick),
            'missing_reason':None if (entry is not None and lookup is not None) else
               'exact layer-entry input aliases or actual selected embedding rows unavailable; subtotal is not complete U'})
    initial_kv=layers*2*batch*(context+3)*hkv*d*2
    accessed_unique_kv=layers*2*batch*(context+5)*hkv*d*2
    unique_entry=sum(input_bytes.values()) if input_bytes else 0
    if layer is None:unique_entry=0
    lookup_unique=0 if layer is not None or tied else (len(set(t for values in input_tokens.values() for t in values))*config['hidden_size']*2 if input_tokens else None)
    complete=all(r['U_main_bytes'] is not None for r in rows)
    return {'classification':'analytical-useful-input accounting','layer':layer,'batch':batch,'context':context,
            'ticks':rows,'U_tick_sum':sum(r['U_main_bytes'] for r in rows) if complete else None,
            'U_window_unique':w+accessed_unique_kv+unique_entry+(lookup_unique or 0) if complete else None,
            'U_window_initial_external_main':w+initial_kv+unique_entry+(lookup_unique or 0) if complete else None,
            'window_KV_definition':'unique historical inputs across ticks4/5/6 include new values produced at ticks4/5 and consumed by later ticks; initial external KV is reported separately',
            'accounting_reader_version':2,
            'geometry':{'layers':layers,'query_heads':hq,'kv_heads':hkv,'head_dim':d},
            'tied_output_head_inferred_from_config_and_absent_independent_tensor':tied,
            'omitted':'RoPE/scheduler metadata, sector overfetch, scratch and actual issued/padded work; physical traffic is measured separately',
            'interpretation':'U is logical demand, not compulsory DRAM; near-one ratios can conceal rereads offset by hits'}
