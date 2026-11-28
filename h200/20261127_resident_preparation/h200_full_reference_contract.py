"""Read-only full-output correctness/provenance checks for future serving.

No GPU execution, serving qualification or timing reuse. Input identity must
also be tied to a sealed offered trace by the future serving controller.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_priority_campaign import valid_tokens, strict_reference_record
from h200_review_native_model import verified_process


def check_tokens(source, hf, contract, source_sha):
    """Recompute complete token agreement; caller flags alone never suffice."""
    assert valid_tokens(source) and strict_reference_record(hf,source_sha)
    config=source['config'];case=source['cases'][0]
    assert config['precision']=='BF16' and config['seed']==contract['seed']==20261013
    assert (config['model_id'],config['model_revision'])==(contract['model']['id'],contract['model']['revision'])
    assert contract['greedy'] is True and contract['ignore_eos'] is True
    assert contract['repetition_penalty']==1.0 and contract['prefix_cache'] is False
    assert contract['model']['precision']=='BF16 weights and KV' and contract['model']['tensor_parallel_size']==1
    settings=source['engine_settings']
    assert settings['dtype']=='bfloat16' and settings['tensor_parallel_size']==1
    assert settings['seed']==20261013 and not settings['enable_prefix_caching']
    assert not config.get('decode_conditioning') and not hf.get('deterministic_reference_diagnostic')
    count=contract['max_tokens'];context=len(contract['prompt_ids'])
    assert type(count) is int and 2<=count<=4096
    assert context+count<=contract['model']['max_positions']
    def token_schema(output,length):
        assert type(output) is list and len(output)==1
        assert type(output[0]) is list and len(output[0])==length
        assert all(type(token) is int and 0<=token<contract['model']['vocab_size'] for token in output[0]), 'integer token IDs required'
    assert context>0
    token_schema([contract['prompt_ids']],context)
    token_schema(case['prompt_ids'],context)
    token_schema(hf['prompt_ids'],context)
    assert (case['batch_size'],case['context_length'],case['generated_tokens'])==(1,context,count)
    assert case['prompt_ids']==[contract['prompt_ids']]
    assert (hf['model_id'],hf['revision'],hf['precision'])==(config['model_id'],config['model_revision'],'BF16')
    assert (hf['batch'],hf['context'],hf['generated_tokens'])==(1,context,count)
    assert hf['prompt_ids']==case['prompt_ids'] and hf['prompt_sha256']==case['prompt_sha256']
    assert hf['gpu_uuid']==source['hardware']['uuid']
    expected=case['trials'][0]['output_ids']
    for output in case['warmup_output_ids']+[row['output_ids'] for row in case['trials']]:
        token_schema(output,count)
    for arm in (hf,hf['static_cache_gate']):
        for phase in ('warmups','trials'):
            assert len(arm[phase])==3, 'three full warmups/trials required for both HF routes'
            for row in arm[phase]:token_schema(row['output_ids'],count)
            assert all(row['output_ids']==expected for row in arm[phase]), 'full HF token sequence differs'
    for row in hf['static_cache_gate']['warmups']+hf['static_cache_gate']['trials']:
        assert row['cache_length']==context+count-1, 'strict cache progress differs'
    return list(expected[0])


def read_reference(contract, source_path, source_sha, hf_path, hf_sha):
    source_path=Path(source_path).resolve();hf_path=Path(hf_path).resolve()
    def pinned(path,digest):
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest, 'reference file pin differs'
        return json.loads(raw)
    source=pinned(source_path,source_sha);hf=pinned(hf_path,hf_sha)
    output=check_tokens(source,hf,contract,source_sha)
    from controlled_vllm_experiments import validate
    assert validate(source_path.parent)==1
    base=source_path.parents[2];label=source_path.parent.name
    assert hf_path==base/'results'/(label+'_hf.json')
    session_path=base/'session.json';session=json.loads(session_path.read_text())
    assert session['gpu_uuid']==source['hardware']['uuid']
    source_provenance=verified_process(base,label,'controlled_vllm_experiments.py',session)
    hf_provenance=verified_process(base,label+'_hf','hf_full_reference.py',session)
    source_process=json.loads(Path(source_provenance['process']).read_text())
    hf_process=json.loads(Path(hf_provenance['process']).read_text())
    for process,entry,output_path in ((source_process,'controlled_vllm_experiments.py',source_path.parent),
            (hf_process,'hf_full_reference.py',hf_path)):
        command=process['command']
        assert command[:2]==[session['python'],str(Path(session['frozen_source_root'])/'scripts'/entry)]
        assert command[command.index('--gpu-uuid')+1]==session['gpu_uuid']
        assert Path(command[command.index('--output')+1]).resolve()==output_path
    assert Path(hf_process['command'][hf_process['command'].index('--source-run')+1]).resolve()==source_path
    adapter=hf['static_cache_gate']['adapter_sha256']
    assert adapter==hf_process['script_sha256']['software_decode_experiments.py']
    assert hashlib.sha256((base/'environment/source_snapshots'/(adapter+'.py')).read_bytes()).hexdigest()==adapter
    for path,digest in ((source_path,source_sha),(hf_path,hf_sha)):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'reference changed during read'
    return {'classification':'Read-only full-output contract and original-process provenance review',
        'full_output_contract_passed':True,'expected_output_ids':output,
        'input_contract_sha256':hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest(),
        'source_run':str(source_path),'source_sha256':source_sha,
        'strict_HF_reference':str(hf_path),'strict_HF_sha256':hf_sha,
        'source_provenance':source_provenance,'HF_provenance':hf_provenance,
        'session':str(session_path),'session_sha256':hashlib.sha256(session_path.read_bytes()).hexdigest(),
        'gpu_uuid':session['gpu_uuid'],'job_id':session['slurm_job_id'],
        'source_identity':session['frozen_source_identity'],
        'source_versions':source['versions'],'HF_versions':hf['versions'],
        'source_engine_settings':source['engine_settings'],
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'GPU_execution':False,'performance_qualified':False,'serving_qualified':False,
        'scope':'Exact full outputs and original owned process records only. Future controller must join this input contract to a sealed offered request, verify matching model/runtime/source/settings, and acquire current allocation/isolation/clock/scheduler/latency gates. No reference-only acceptance of a serving result or timing/UUID pooling.'}
