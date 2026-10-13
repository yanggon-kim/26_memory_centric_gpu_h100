#!/usr/bin/env python3
"""Resolve/pin model snapshots and bounded stage configs; no GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main(args):
    package=Path(__file__).resolve().parents[1]
    model=json.loads((package/'configs/models.json').read_text())['models'][args.model]
    assert not (model['stage']=='conditional' and args.reuse_output),'MoE timing/trace allowed within budget; dense logical reuse adapter unsupported'
    expected=model['revision']
    if args.snapshot:
        snapshot=args.snapshot.resolve()
        sha=snapshot.name
        assert re.fullmatch('[0-9a-f]{40}',sha),'snapshot basename must be immutable HF commit SHA'
        assert expected=='RESOLVE_AND_PIN' or sha==expected,'model revision mismatch'
        cache_name='models--'+model['model_id'].replace('/','--')
        if not (snapshot.parent.name=='snapshots' and snapshot.parent.parent.name==cache_name):
            assert args.provenance_json,'offline custom snapshot requires trusted model/revision/config-hash provenance JSON'
            proof=json.loads(args.provenance_json.read_text())
            assert proof['model_id']==model['model_id'] and (proof.get('revision') or proof.get('model_revision'))==sha
            assert proof['config_sha256']==hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest(),'offline config hash mismatch'
    else:
        assert args.download,'provide --snapshot for offline materialization, or explicitly --download'
        from huggingface_hub import HfApi,snapshot_download
        info=HfApi().model_info(model['model_id'],revision=None if expected=='RESOLVE_AND_PIN' else expected,files_metadata=True)
        sha=info.sha
        assert re.fullmatch('[0-9a-f]{40}',sha)
        siblings=[s for s in info.siblings if s.rfilename.endswith(('.safetensors','.json','.model'))]
        assert all(s.size is not None for s in siblings),'download sizes unavailable; manually verify budget before download'
        total=sum(s.size for s in siblings)
        assert total<=model['max_download_gib']*1024**3,'declared model download budget exceeded'
        snapshot=Path(snapshot_download(model['model_id'],revision=sha,allow_patterns=['*.safetensors','*.json','*.model'],max_workers=2))
    actual=json.loads((snapshot/'config.json').read_text())
    assert actual['model_type']==model['expected_model_type'],'snapshot model architecture differs from declared family'
    assert list(snapshot.glob('*.safetensors')),'no pretrained safetensors found'
    stage=model['stages'][args.stage]
    config={'model_id':model['model_id'],'model_revision':sha,'model_path':str(snapshot),
            'batch_sizes':stage['batch_sizes'],'context_lengths':stage['context_lengths'],'precision':'BF16',
            'generated_tokens':64,'energy_generated_tokens':2048,'energy_cases':[[1,1024]],
            'warmup_trials':3,'trials':3,'seed':20261013,'compile':True,'maximum_cache_length':8192,
            'sensor_interval_s':.02,'idle_calibration_s':5,'minimum_energy_window_s':5,
            'copy_buffer_mib':256,'copy_repetitions':200,'engine':dict(model['engine']),
            'stage':args.stage,'config_sha256':hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest(),
            'layout':{key:actual.get(key,'NA') for key in ['model_type','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','sliding_window']}}
    config['engine']['max_num_seqs']=max(stage['batch_sizes'])
    config['engine']['max_num_batched_tokens']=max(8192,max(stage['batch_sizes'])*max(stage['context_lengths']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(config,indent=2)+'\n')
    if args.reuse_output:
        assert args.l2_bytes is not None and args.l2_bytes>0 and args.l2_source,'explicit measured/SKU-verified L2 bytes and provenance required'
        reuse={'models':{args.model:str(snapshot)},'cases':[{'case_id':f'{args.model}_b{b}_c{c}_p{int(p*100)}','model':args.model,
               'batch_size':b,'context_length':c,'shared_prefix_fraction':p} for b in stage['batch_sizes'] for c in stage['context_lengths'] for p in [0,.75]],
               'additional_sram_mib':[0,8,32,128,512],'chunk_bytes':65536,'decode_steps':3,'warmup_steps':1,
               'l2_bytes':args.l2_bytes,'l2_capacity_provenance':args.l2_source,
               'scope':'logical synthetic full-block fills; not measured GPU traffic, area, energy or timing'}
        args.reuse_output.parent.mkdir(parents=True,exist_ok=True)
        args.reuse_output.write_text(json.dumps(reuse,indent=2)+'\n')
    print(json.dumps({'model':model['model_id'],'immutable_revision':sha,'stage':args.stage,'output':str(args.output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--stage',choices=['smoke','baseline'],default='smoke')
    p.add_argument('--snapshot',type=Path,help='local immutable snapshot, no downloads')
    p.add_argument('--provenance-json',type=Path,help='trusted model/revision/config hash for custom offline snapshot')
    p.add_argument('--download',action='store_true',help='explicit bounded download and immutable revision resolution')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-output',type=Path)
    p.add_argument('--l2-bytes',type=int)
    p.add_argument('--l2-source')
    main(p.parse_args())
