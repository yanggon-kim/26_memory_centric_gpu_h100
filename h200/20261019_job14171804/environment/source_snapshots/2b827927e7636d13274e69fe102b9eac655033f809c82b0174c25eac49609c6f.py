#!/usr/bin/env python3
"""CPU-only negative contracts and optional actual fixture/config integration tests."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from export_trace import export,replay_evidence


def main(a):
    package=Path(__file__).resolve().parents[1]
    checked=[]
    bad_logs={'unsupported exit-zero':'unsupported binary version120\n',
              'missing instructions':'kernel_name = real_kernel\ngpu_tot_sim_cycle = 1\nGPGPU-Sim: *** exit detected ***\n',
              'two cycles one instruction block':'kernel_name = unit_test_only1\ngpu_tot_sim_cycle = 1\ngpu_tot_sim_insn = 1\nkernel_name = unit_test_only2\ngpu_tot_sim_cycle = 2\nGPGPU-Sim: *** exit detected ***\n',
              'incomplete kernel count':'kernel_name = unit_test_only\ngpu_tot_sim_cycle = 1\ngpu_tot_sim_insn = 1\nGPGPU-Sim: *** exit detected ***\n'}
    for name,log in bad_logs.items():
        try:replay_evidence(log,2 if name in ['incomplete kernel count','two cycles one instruction block'] else 1)
        except AssertionError:checked.append('rejected '+name)
        else:raise AssertionError('accepted invalid replay '+name)
    if a.tensor_storage:
        import torch
        from tensor_manifest import parameters
        model=torch.nn.Module()
        model.register_parameter('base',torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(3,4)))
        model.register_parameter('tied',model.base)
        model.register_parameter('strided',torch.nn.Parameter(model.base[1:,::2]))
        tensors={row['name']:row for row in parameters(model)['tensors']}
        assert tensors['base']['storage_identity']==tensors['tied']['storage_identity']==tensors['strided']['storage_identity']
        assert tensors['strided']['storage_offset_elements']==4 and tensors['strided']['stride_elements']==[4,2]
        assert not tensors['strided']['contiguous'] and tensors['strided']['storage_nbytes']==48 and tensors['strided']['logical_tensor_bytes']==16
        checked.append('CPU tensor aliases/noncontiguous storage ranges; no GPU execution')
    from reuse_analysis import ByteLRU,PolicyCache
    from types import SimpleNamespace
    from gpu_target import verify_cuda
    fixture_uuid='GPU-00000000-0000-0000-0000-000000000001'
    def fake_cuda(name='NVIDIA H200', capability=(9,0), uuid=fixture_uuid, count=1):
        return SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:count,
            get_device_properties=lambda index:SimpleNamespace(name=name,uuid=uuid),
            get_device_capability=lambda index=0:capability))
    for name in ('NVIDIA H100 80GB HBM3','NVIDIA H200'):
        verify_cuda(fake_cuda(name=name),fixture_uuid)
    checked.append('accepted explicitly scoped H100/H200 sm_90 device identities')
    from fa3_metadata import decode as decode_fa3
    # Active fields only: nonzero/uninitialized padding is not request data.
    decoded=decode_fa3([6,-91,999,42,8,-100,1234,90,2,77,88,99,0],1)
    assert decoded['packed_query_lengths']==[6]
    assert decoded['per_request_dynamic_splits']==[8]
    assert decoded['num_nheads_in_l2']==[2]
    assert decode_fa3([6,77,88,99,2,55,66,77,0],1)['per_request_dynamic_splits']==[1]
    for raw,causal in [([6]*12,True),([6,0,0,0,0,0,0,0,2,0,0,0,0],True),([6]*13,False)]:
        try:decode_fa3(raw,1,causal)
        except ValueError:pass
        else:raise AssertionError('unsupported FA3 metadata accepted')
    checked.append('FA3 active metadata fields ignore padding; unsupported layout and invalid active splits rejected')
    for label,device in [('wrong CUDA UUID',fake_cuda(uuid='GPU-00000000-0000-0000-0000-000000000002')),
                         ('unsupported GPU family',fake_cuda(name='NVIDIA A100')),
                         ('wrong H200 capability',fake_cuda(capability=(8,0))),
                         ('multiple visible GPUs',fake_cuda(count=2))]:
        try:verify_cuda(device,fixture_uuid)
        except AssertionError:checked.append('rejected '+label)
        else:raise AssertionError('accepted '+label)
    for label,constructor in [('negative cache',lambda:ByteLRU(-1)),('negative added storage',lambda:PolicyCache(1024,-1,'lru',set()))]:
        try:constructor()
        except ValueError:checked.append('rejected '+label)
        else:raise AssertionError('negative capacity accepted')
    with tempfile.TemporaryDirectory(prefix='cpu_contract_') as temporary:
        temporary=Path(temporary)
        # A repository may publish two complete encodings of the same model.
        # The native alternate must not double the HF download or bypass budget.
        from unittest.mock import patch
        import contextlib
        import io
        from materialize_config import main as materialize
        fixture=temporary/'mock_hf_checkpoint'/'1111111111111111111111111111111111111111'
        fixture.mkdir(parents=True)
        (fixture/'config.json').write_text(json.dumps({'model_type':'mistral'}))
        shards=['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']
        for shard in shards:(fixture/shard).write_bytes(b'unit-test-only-not-model-weights')
        (fixture/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':shards[0],'b':shards[1]}}))
        siblings=[SimpleNamespace(rfilename=name,size=size,lfs=None) for name,size in
                  [('config.json',32),('model.safetensors.index.json',128),
                   ('tokenizer.json',1000),('tokenizer.model',130),
                   (shards[0],7*1024**3),(shards[1],7*1024**3),('consolidated.safetensors',14*1024**3)]]
        def snapshot_download(model_id,revision,allow_patterns,max_workers):
            assert set(shards)<=set(allow_patterns) and 'consolidated.safetensors' not in allow_patterns
            assert 'tokenizer.json' in allow_patterns and 'tokenizer.model' not in allow_patterns
            return str(fixture)
        hub=SimpleNamespace(HfApi=lambda:SimpleNamespace(model_info=lambda *a,**k:
                            SimpleNamespace(sha=fixture.name,siblings=siblings)),
                            hf_hub_download=lambda *a,**k:str(fixture/'model.safetensors.index.json'),
                            snapshot_download=snapshot_download)
        with patch.dict(sys.modules,{'huggingface_hub':hub}),contextlib.redirect_stdout(io.StringIO()):
            materialize(SimpleNamespace(model='mistral7b',stage='smoke',snapshot=None,download=True,revision=None,
                        output=temporary/'selected_hf.json',reuse_output=None))
        proof=json.loads((temporary/'selected_hf.download_manifest.json').read_text())
        assert proof['selected_bytes']<20*1024**3 and proof['selected_weights']==shards
        checked.append('complete HF sharded layout excludes duplicate native weights within unchanged budget')
        from run_h200_phase01 import config_for
        selected=config_for(temporary,temporary/'selected_hf.json','unit_test_energy_b8',8,4096)
        assert json.loads(selected.read_text())['energy_cases']==[[8,4096]]
        checked.append('per-case energy selection retains requested batch/context')
        import hashlib
        from resolve_mistral_tokenizer import resolve
        native=temporary/'tokenizer.model.v3'
        native.write_bytes(b'unit-test-tokenizer-artifact-not-a-real-tokenizer')
        expected=hashlib.sha256(native.read_bytes()).hexdigest()
        pointer=f'version https://git-lfs.github.com/spec/v1\noid sha256:{expected}\nsize {native.stat().st_size}\n'.encode()
        config={'model_id':'mistralai/Mistral-7B-Instruct-v0.3','model_revision':fixture.name,'model_path':str(fixture)}
        with patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=lambda *a,**k:str(native))}),patch('urllib.request.urlopen',return_value=io.BytesIO(pointer)):
            resolution=resolve(config,probe_tokenizer=False)
        assert resolution['resolved_sha256']==expected and (fixture/'tokenizer.model').resolve()==native
        bad_pointer=pointer.replace(expected.encode(),b'0'*64)
        with patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=lambda *a,**k:str(native))}),patch('urllib.request.urlopen',return_value=io.BytesIO(bad_pointer)):
            try:resolve(config,probe_tokenizer=False)
            except AssertionError:checked.append('tokenizer LFS alias SHA256 verified; mismatched payload rejected')
            else:raise AssertionError('wrong tokenizer alias hash accepted')
        from controlled_vllm_experiments import validate as validate_vllm
        from software_decode_experiments import digest
        contract=temporary/'vllm_contract'
        contract.mkdir()
        output_ids=[[1,2,3]]
        output_hash=digest(output_ids)
        case={'batch_size':1,'context_length':1,'generated_tokens':3,'prompt_ids':[[9]],
              'prompt_sha256':digest([[9]]),'output_digests_identical':True,
              'warmup_output_sha256':[output_hash]*3,'trials':[{
              'prefill_barrier':{'passed':True,'request_count':1,'output_token_counts':{'unit_test_request':1},
                                 'cache_hits':{'unit_test_request':0},'finished':{'unit_test_request':False}},
              'decode_iterations':2,'timed_generated_tokens':2,'output_ids':output_ids,
              'output_sha256':output_hash,'cuda_event_ms':1,'wall_ms':1,'tokens_per_s':2000} for _ in range(3)]}
        data={'status':'complete','classification':'measured-real-GPU','engine_client_class':'InprocClient',
              'engine_settings':{'async_scheduling':False,'distributed_executor_backend':'uni'},
              'environment':{'VLLM_ENABLE_V1_MULTIPROCESSING':'0'},'config':{'warmup_trials':3,'trials':3},
              'selected_cases':[{'batch_size':1,'context_length':1,'generated_tokens':3}],
              'cases':[case],'mode':'timing'}
        (contract/'run.json').write_text(json.dumps(data))
        assert validate_vllm(contract)==1
        case['warmup_output_sha256'][0]='unit_test_different_output'
        case['output_digests_identical']=False
        (contract/'run.json').write_text(json.dumps(data))
        try:validate_vllm(contract)
        except AssertionError:checked.append('rejected nondeterministic controlled vLLM outputs')
        else:raise AssertionError('nondeterministic vLLM outputs accepted')
        bad_config=temporary/'negative_cache_config.json'
        bad_config.write_text(json.dumps({'l2_bytes':1024,'additional_sram_mib':[-0.5]}))
        bad_run=subprocess.run([sys.executable,str(package/'scripts/reuse_analysis.py'),'--config',str(bad_config),'--output-dir',str(temporary/'invalid_reuse')],capture_output=True,text=True)
        assert bad_run.returncode!=0 and 'capacities must be nonnegative' in bad_run.stderr
        checked.append('rejected negative capacity in reuse CLI config')
        if a.snapshot:
            config=temporary/'bridge.json'
            common=[sys.executable,str(package/'scripts/materialize_config.py'),'--snapshot',str(a.snapshot),'--output',str(config)]
            subprocess.run(common+['--model','bridge'],check=True,capture_output=True,text=True)
            negative=subprocess.run(common+['--model','bridge','--reuse-output',str(temporary/'reuse.json'),'--l2-bytes','-1','--l2-source','unit-test-negative'],capture_output=True,text=True)
            assert negative.returncode!=0,'negative materialized L2 capacity was accepted'
            checked.append('rejected negative materialized L2')
            materialized=json.loads(config.read_text())
            assert materialized['model_id']=='Qwen/Qwen2.5-1.5B-Instruct' and materialized['precision']=='BF16'
            wrong=subprocess.run(common+['--model','qwen7b'],capture_output=True,text=True)
            assert wrong.returncode!=0,'offline wrong-model snapshot was accepted'
            checked+=['offline immutable bridge materialization','rejected wrong-model offline snapshot']
            from reuse_analysis import manifest,regions
            original=manifest(a.snapshot)
            for label,changes in [('unsupported dense family',{'model_type':'not_supported'}),('active sliding window',{'sliding_window':128,'use_sliding_window':True})]:
                model={**original,'config':{**original['config'],**changes}}
                try:next(regions(model,{'batch_size':1,'context_length':128},1))
                except ValueError:checked.append('rejected '+label)
                else:raise AssertionError('unsupported analytical layout accepted')
        if a.fixture_root:
            assert a.accel_root,'--accel-root required for CPU postprocessor fixture test'
            target=temporary/'actual_capture_fixture'
            shutil.copytree(a.fixture_root,target)
            summary=export(target,a.accel_root.resolve(),False)
            assert summary['reported_warp_instructions']>0 and summary['grouped_files']>0
            if a.expected_binary is not None:assert summary['binary_versions']==[a.expected_binary]
            checked.append({'actual_capture_fixture':str(a.fixture_root),'reported_warp_instructions':summary['reported_warp_instructions'],
                            'binary_versions':summary['binary_versions'],'scope':'CPU export only; no new GPU capture or H100 calibration'})
    if a.actual_replay_log:
        checked.append({'actual_replay':replay_evidence(a.actual_replay_log.read_text(),a.expected_kernels),
                        'scope':'existing genuine fixture completion; no new simulation or H100 calibration'})
    print(json.dumps({'status':'passed','CPU_checks':checked,'new_GPU_execution':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor-storage',action='store_true',help='CPU-only real torch storage/view alias test')
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--fixture-root',type=Path)
    p.add_argument('--accel-root',type=Path)
    p.add_argument('--expected-binary',type=int)
    p.add_argument('--actual-replay-log',type=Path)
    p.add_argument('--expected-kernels',type=int,default=10)
    main(p.parse_args())
