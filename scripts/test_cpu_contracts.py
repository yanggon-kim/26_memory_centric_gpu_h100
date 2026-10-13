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
    for label,constructor in [('negative cache',lambda:ByteLRU(-1)),('negative added storage',lambda:PolicyCache(1024,-1,'lru',set()))]:
        try:constructor()
        except ValueError:checked.append('rejected '+label)
        else:raise AssertionError('negative capacity accepted')
    with tempfile.TemporaryDirectory(dir=package/'results',prefix='cpu_contract_') as temporary:
        temporary=Path(temporary)
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
