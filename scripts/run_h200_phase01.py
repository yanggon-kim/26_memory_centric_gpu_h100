#!/usr/bin/env python3
"""Sequential fresh H200 baselines with per-case persistence and strict gates."""
import argparse
import json
from pathlib import Path
import sys
from h200_session import DEFAULT_ROOT, PACKAGE, PYTHON, UUID, dump, python_command, run
from summarize_h200 import summarize


def materialize(root, model, stage):
    target=root/'configs'/f'{model}_{stage}.json'
    if target.exists():
        return target
    arguments=['--model',model,'--stage',stage,'--output',target]
    smoke=root/'configs'/f'{model}_smoke.json'
    if stage=='baseline' and smoke.exists():
        arguments += ['--snapshot',json.loads(smoke.read_text())['model_path']]
    else:
        arguments += ['--download']
    outcome=run(root,f'materialize_{model}_{stage}',python_command('materialize_config.py',*arguments),900)
    return target if outcome['status']=='complete' else None


def config_for(root, base, label, batch, context, eager=False, compile=False, workload=False):
    config=json.loads(Path(base).read_text())
    config.update(batch_sizes=[batch],context_lengths=[context],compile=compile)
    config['engine']['enforce_eager']=eager
    if eager:
        config['engine']['compilation_config']={'mode':0}
    if workload:
        config['workload_file']='configs/representative_text_v1.json'
    target=root/'configs'/(label+'.json')
    if target.exists():
        assert json.loads(target.read_text())==config,'changed config requires a new label'
    else:
        dump(target,config)
    return target


def case(root, model, base, batch, context, engine='graph', mode='timing', count=None,
         smoke=False, workload=False, retry=None):
    label=f'{model}_{engine}_b{batch}_c{context}'
    if smoke:label+='_smoke'
    if workload:label+='_text_v1'
    if mode=='energy':label+=f'_energy_n{count}'
    if retry:label+='_'+retry
    output=root/'runs'/label
    if (output/'run.json').exists():
        old=json.loads((output/'run.json').read_text())
        return output if old['status']=='complete' else None
    config=config_for(root,base,label,batch,context,eager=engine=='eager',
                      compile=engine=='hf_compiled',workload=workload)
    script='software_decode_experiments.py' if engine.startswith('hf_') else 'controlled_vllm_experiments.py'
    command=python_command(script,'--config',config,'--gpu-uuid',UUID,'--mode',mode,'--output',output,
                           '--batch',str(batch),'--context',str(context))
    if count is not None and not engine.startswith('hf_'):
        command+=['--generated-tokens',str(count)]
    outcome=run(root,label,command,1200 if engine=='hf_compiled' else 900,gpu=True)
    summarize(root)
    return output if outcome['status']=='complete' else None


def compare(root, label, vllm, hf):
    target=root/'results'/(label+'.json')
    if target.exists():return json.loads(target.read_text())['status']=='passed'
    outcome=run(root,label,python_command('compare_vllm_hf_tokens.py','--vllm-run',vllm/'run.json',
                 '--hf-run',hf/'run.json','--output',target),60)
    return outcome['status']=='complete'


def model_smoke(root, model):
    smoke=materialize(root,model,'smoke')
    if smoke is None:return None
    hf=case(root,model,smoke,1,128,'hf_eager',smoke=True)
    if hf is None:return None
    graph=case(root,model,smoke,1,128,smoke=True)
    if graph is None:return None
    if not compare(root,model+'_graph_hf_smoke_tokens',graph,hf):
        # A distinct eager path is a diagnostic, never a retroactive graph pass.
        eager=case(root,model,smoke,1,128,'eager',smoke=True)
        if eager:compare(root,model+'_eager_hf_smoke_tokens',eager,hf)
        return None
    return materialize(root,model,'baseline')


def energy(root,model,base,batch,context):
    for count in (1024,2048,4096):
        if context+count>8192:break
        output=case(root,model,base,batch,context,mode='energy',count=count)
        if output is None:return
        data=json.loads((output/'run.json').read_text())
        telemetry=[t['telemetry'] for t in data['cases'][0]['trials']]
        if all(isinstance(t['joules_per_token'],(float,int)) for t in telemetry):return
        if all(t['window_s']>=5 for t in telemetry):return


def baseline(root,model,base):
    config=json.loads(base.read_text())
    # Deliver contrasting regimes first, retaining the complete declared matrix.
    pairs=[(b,c) for b in config['batch_sizes'] for c in config['context_lengths']]
    preferred=[(1,128),(1,1024),(max(config['batch_sizes']),max(config['context_lengths']))]
    ordered=list(dict.fromkeys([p for p in preferred if p in pairs]+pairs))
    for batch,context in ordered:
        case(root,model,base,batch,context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--stage',choices=['bridge','portfolio','controls','energy'],required=True)
    parser.add_argument('--model',choices=['bridge','qwen7b','mistral7b','qwen14b','qwen32b'])
    args=parser.parse_args()
    root=args.root.resolve()
    if args.stage=='bridge':
        base=model_smoke(root,'bridge')
        if base is None:
            summarize(root)
            raise SystemExit('Bridge correctness/runtime gate failed; preserved diagnostics require investigation')
        baseline(root,'bridge',base)
        for b,c in [(1,128),(1,1024),(8,4096)]:case(root,'bridge',base,b,c,'eager')
        for batch in (1,4):case(root,'bridge',base,batch,128,workload=True)
    elif args.stage=='portfolio':
        for model in ([args.model] if args.model else ['qwen7b','mistral7b','qwen14b','qwen32b']):
            base=model_smoke(root,model)
            if base:baseline(root,model,base)
    elif args.stage=='controls':
        base=root/'configs/bridge_baseline.json'
        assert base.exists()
        for b,c in [(1,128),(1,1024),(8,4096)]:case(root,'bridge',base,b,c,'hf_compiled')
    elif args.stage=='energy':
        for model in ([args.model] if args.model else ['bridge','qwen7b','mistral7b','qwen14b','qwen32b']):
            base=root/'configs'/f'{model}_baseline.json'
            gate=root/'results'/f'{model}_graph_hf_smoke_tokens.json'
            if not base.exists() or not gate.exists() or json.loads(gate.read_text())['status']!='passed':continue
            config=json.loads(base.read_text())
            for b,c in list(dict.fromkeys([(1,1024),(max(config['batch_sizes']),max(config['context_lengths']))])):
                energy(root,model,base,b,c)
    summarize(root)


if __name__=='__main__':
    main()
