#!/usr/bin/env python3
"""Hash resolved model files and compute explicitly analytical KV payloads."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
from h200_session import DEFAULT_ROOT,dump


def record(root):
    models=[]
    for source in sorted((root/'configs').glob('*_smoke.json')):
        # Only registry materializations, not individual run configurations.
        if source.stem not in {m+'_smoke' for m in ['bridge','qwen7b','mistral7b','qwen14b','qwen32b','conditional_moe']}:continue
        config=json.loads(source.read_text())
        snapshot=Path(config['model_path'])
        destination=root/'environment'/(source.stem.removesuffix('_smoke')+'_model_identity.json')
        if destination.exists():
            models.append(json.loads(destination.read_text()));continue
        tensors=[];files=[];types=collections.Counter()
        names=['model.safetensors'] if (snapshot/'model.safetensors').exists() else sorted(set(json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map'].values()))
        for name in names:
            path=snapshot/name
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                size=struct.unpack('<Q',stream.read(8))[0]
                header=json.loads(stream.read(size))
                stream.seek(0)
                for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
            files.append({'path':name,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
            for key,value in header.items():
                if key=='__metadata__':continue
                length=value['data_offsets'][1]-value['data_offsets'][0]
                types[value['dtype']]+=length
                tensors.append({'name':key,'shape':value['shape'],'checkpoint_dtype':value['dtype'],'payload_bytes':length})
        actual=json.loads((snapshot/'config.json').read_text())
        hq=actual['num_attention_heads'];hkv=actual.get('num_key_value_heads',hq)
        dim=actual.get('head_dim') or actual['hidden_size']//hq
        layers=actual['num_hidden_layers']
        accounting=[]
        for batch in [1,4,8]:
            for context in [128,1024,4096]:
                accounting.append({'batch':batch,'context':context,'classification':'analytical',
                    'bf16_kv_payload_all_layers_bytes':layers*2*2*batch*hkv*context*dim,
                    'attention_flops_per_layer_ordinary_decode':4*batch*hq*context*dim,
                    'kv_only_flops_per_requested_byte':2*hq/(2*hkv),
                    'scope':'logical initial KV payload; not allocation padding, accessed traffic, HBM bytes or measured FLOPs'})
        model={'model_id':config['model_id'],'revision':config['model_revision'],
               'snapshot_path':str(snapshot),'config_sha256':hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest(),
               'selected_checkpoint_files':files,'checkpoint_payload_bytes_by_dtype':dict(types),
               'checkpoint_tensors':tensors,'logical_kv_attention_accounting':accounting,
               'runtime_precision':'BF16 requested; actual allocated dtype separately verified by worker metadata'}
        dump(destination,model);models.append(model)
        print('HASHED',config['model_id'],sum(x['bytes'] for x in files),flush=True)
    dump(root/'results/model_memory_accounting.json',{'models':[{
         k:v for k,v in model.items() if k!='checkpoint_tensors'} for model in models]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    record(parser.parse_args().root.resolve())
