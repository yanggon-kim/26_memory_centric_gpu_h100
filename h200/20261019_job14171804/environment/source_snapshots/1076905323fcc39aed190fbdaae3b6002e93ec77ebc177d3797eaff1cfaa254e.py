#!/usr/bin/env python3
"""Interleave explicit correctness/startup recovery at an owned stage boundary."""
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time
from h200_session import DEFAULT_ROOT,UUID,LAUNCH_CUTOFF,python_command,run,dump
from run_h200_phase01 import model_smoke,baseline,energy
from analyze_h200 import analyze


def canonical_reference(root,model,source):
    output=root/'results'/(model+'_full_dynamic_reference.json')
    if not output.exists():
        run(root,model+'_full_dynamic_reference',python_command('hf_full_reference.py',
            '--source-run',source/'run.json','--gpu-uuid',UUID,'--output',output),600,gpu=True)
    if not output.exists():return False
    reference=json.loads(output.read_text())
    ids=reference['trials'][0]['output_ids']
    controls=[]
    for path in [source,root/'runs'/f'{model}_eager_b1_c128_smoke']:
        if not (path/'run.json').exists():continue
        data=json.loads((path/'run.json').read_text())
        actual=data['cases'][0]
        match=(data['status']=='complete' and actual['output_digests_identical']
               and actual['prompt_sha256']==reference['prompt_sha256']
               and all(t['output_ids']==ids for t in actual['trials'])
               and all(row==ids for row in actual.get('warmup_output_ids',[])))
        controls.append({'source':str(path.relative_to(root)),'exact_reference_ids':match})
    passed=reference['status']=='passed' and controls and all(c['exact_reference_ids'] for c in controls)
    proof={'status':'passed' if passed else 'failed','reference':str(output.relative_to(root)),
           'reference_program':'independent full-length HF dynamic-cache model.generate',
           'controls':controls,'policy':'exact IDs across three complete warmups/trials; no tolerance',
           'historical_static_cache_gate':'preserved separately; a new reference does not repair a failed StaticCache control'}
    dump(root/'results'/(model+'_canonical_hf_smoke_tokens.json'),proof)
    return passed


def main():
    root=DEFAULT_ROOT
    marker=root/'results/schedule_interleave.json'
    state=json.loads(marker.read_text())
    parent=state['scheduler_pid'];stage=state['inflight_stage_pid']
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    try:
        while True:
            status=Path('/proc')/str(stage)/'stat'
            if not status.exists() or status.read_text().split()[2]=='Z':break
            if time.time()>=LAUNCH_CUTOFF:raise SystemExit('allocation closeout cutoff')
            time.sleep(2)
        print('Stage boundary reached; starting explicit recovery',flush=True)
        source=root/'runs/qwen7b_graph_b1_c128_smoke'
        canonical_reference(root,'qwen7b',source)
        for model in ['qwen7b','mistral7b']:
            base=model_smoke(root,model)
            if base is None and model=='mistral7b':
                source=root/'runs/mistral7b_graph_b1_c128_smoke_tokenizer_resolved'
                if (source/'run.json').exists() and json.loads((source/'run.json').read_text())['status']=='complete':
                    if canonical_reference(root,model,source):base=model_smoke(root,model)
            if base:
                baseline(root,model,base)
                config=json.loads(base.read_text())
                for b,c in [(1,1024),(max(config['batch_sizes']),max(config['context_lengths']))]:
                    energy(root,model,base,b,c)
            print('RECOVERY',model,'baseline ready' if base else 'correctness gate remains incomplete',flush=True)
            print('ANALYSIS',analyze(root),flush=True)
        # Complete the additional graph attention sample and quantify the
        # uncontrolled-cache replay issue with a distinctly named cold control.
        for suffix,extra in [('attention_completion',[]),('cold_cache_control',['--cache-control','all'])]:
            run(root,'bridge_graph_b1_'+suffix,['python3','scripts/profile_h200_graph.py',
                '--model','bridge','--batch','1','--context','1024',*extra],1800)
        run(root,'model_hashes_complete',['python3','scripts/record_h200_models.py'],600)
        analyze(root)
    finally:
        proc=Path('/proc')/str(parent)
        if proc.exists():
            assert proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode().strip()=='python3 -u -'
            os.kill(parent,signal.SIGCONT)
        state.update(state='scheduler resumed',resumed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        dump(marker,state)
        print('Main measurement schedule resumed',flush=True)


if __name__=='__main__':main()
