#!/usr/bin/env python3
"""CPU-only package integrity checks; never certify future GPU/simulation results."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

REQUIRED = ['README.md','AGENTS.md','STATUS.json','configs/models.json','configs/simulator_pin.json',
 'configs/runtime.lock.txt','docs/primary_specification.md','docs/phase02_playbook.md','docs/trace_collection.md',
 'scripts/setup.sh','scripts/inventory.py','scripts/materialize_config.py','scripts/gpu_target.py',
 'scripts/controlled_vllm_experiments.py','scripts/software_decode_experiments.py','scripts/reuse_analysis.py',
 'scripts/trace_decode.py','scripts/trace_vllm_decode.py','scripts/run_trace.py','scripts/export_trace.py','scripts/setup_accelsim.sh','scripts/tensor_manifest.py','scripts/test_cpu_contracts.py']


def validate(root):
    missing=[name for name in REQUIRED if not (root/name).is_file()]
    assert not missing, 'missing files: '+', '.join(missing)
    status=json.loads((root/'STATUS.json').read_text())
    assert status['h100_measurements'] in ['pending','partial','complete'] and status['h100_trace_replay'] in ['pending','partial','complete']
    if status['h100_measurements']=='complete' or status['h100_trace_replay']=='complete':
        assert (root/'results/h100_completion_evidence.json').is_file(),'complete status requires separately verified evidence manifest; integrity check is not data certification'
    models=json.loads((root/'configs/models.json').read_text())['models']
    assert {'bridge','qwen7b','qwen14b','qwen32b','mistral7b'} <= set(models)
    for model in models.values():
        assert model['precision']=='BF16' and model['model_id'] and model['revision']
        assert model['stage']=='conditional' or model['stages']['smoke']['batch_sizes']==[1]
    for path in (root/'scripts').glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
        assert ('/home/'+'yanggon/') not in path.read_text(), 'host path in portable source '+str(path)
    for path in (root/'scripts').glob('*.sh'):
        subprocess.run(['bash','-n',str(path)],check=True)
    assert 'single-layer' in (root/'docs/trace_collection.md').read_text()
    assert 'ALLOW_REG_VAL_TRACING' in (root/'scripts/run_trace.py').read_text()
    return len(REQUIRED)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args()
    try:
        print(f'Validated {validate(a.root)} required portable files; integrity only, GPU/replay evidence separately gated')
    except (OSError,ValueError,AssertionError,subprocess.CalledProcessError) as error:
        p.exit(1,f'INVALID PACKAGE: {error}\n')
