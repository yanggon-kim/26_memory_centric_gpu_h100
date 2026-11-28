#!/usr/bin/env python3
"""Allocation-bound H200 acquisitions, qualification and Git checkpoints."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(root, message):
    subprocess.run([sys.executable, str(PACKAGE/'scripts/checkpoint_h200.py'),
                    '--root', str(root), '--message', message, '--push'], check=True)


def update_job(root, job, status, reason=None):
    manifest = load(root/'results/shoreline_case_manifest.json')
    row = next(x for x in manifest['jobs'] if x['id'] == job)
    row.update(status=status, reason=reason, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    row['scientific_gate'] = 'passed' if status == 'complete' else 'incomplete'
    dump(root/'results/shoreline_case_manifest.json', manifest)
    dump(root/'STATUS.json', {'assignment': 'shoreline J0-J11',
                             'session': load(root/'session.json'), 'jobs': manifest['jobs'],
                             'historical_data': 'previous session retained; no pooling across allocations'})


def inventory(root):
    update_job(root, 'J0', 'in_progress')
    commands = [
        ('inventory', python_command('inventory.py', '--gpu-uuid', UUID, '--check-cuda',
                                    '--output', root/'environment/inventory.json'), True),
        ('cpu_contracts', python_command('test_cpu_contracts.py'), False),
        ('pip_freeze', [PYTHON, '-m', 'pip', 'freeze'], False),
        ('ncu_catalog', [NCU, '--query-metrics', '--query-metrics-mode', 'suffix', '--metrics',
                         'dram__bytes_read,dram__bytes_write,gpu__time_duration,sm__warps_active'], False),
    ]
    outcomes = []
    for label, command, gpu in commands:
        if (root/'results/processes'/(label+'.json')).exists():
            outcome = load(root/'results/processes'/(label+'.json'))
            if outcome['status'] != 'complete':
                outcome = run(root, label+'_revision_api_retry1', command, 300, gpu=gpu)
        else:
            outcome = run(root, label, command, 300, gpu=gpu)
        outcomes.append(outcome)
    props = load(root/'environment/inventory.json')
    assert props.get('selected_cuda', {}).get('uuid', '').removeprefix('GPU-') == UUID.removeprefix('GPU-')
    assert all(x['status'] == 'complete' for x in outcomes), 'J0 feasibility/CPU checks failed; inspect logs'
    capabilities = [
        {'family': 'baseline graph/eager and HF StaticCache', 'status': 'implemented; current native smoke pending'},
        {'family': 'whole-range ticks 4/5/6 and per-pass proof', 'status': 'requires adapter; previous ranges failed'},
        {'family': 'independent native service and overlap controls', 'status': 'requires adapter and bounded native pilot'},
        {'family': 'green contexts and resource/concurrency proofs', 'status': 'requires capability query and pilot'},
        {'family': 'pressure/retention controls', 'status': 'requires matched allocation/tick adapter'},
        {'family': 'native trace', 'status': 'previous micro evidence retained; current tools/semantics require validation'},
    ]
    dump(root/'results/capability_ledger.json', {'capabilities': capabilities, 'new_quantitative_protocol': load(root/'session.json')})
    (root/'analysis').mkdir(exist_ok=True)
    (root/'analysis/experiment_plan.md').write_text(
        '# Current H200 execution\n\nJ0 → J1 → J2 → J3 → J4 → J5 → J8 → J9 → J7 → J10 → J6 → J11. '
        'Every job/acquisition is checkpointed. The previous 26/35 dense matrix is historical; its nine '
        'blocked Qwen7B cells are not required for the focused study. Three process acquisitions, '
        'three warmups and ten timed trials are required for new comparisons.\n\n'
        'Allocation, UUID, actual toolchain and cutoff are recorded in session.json. No new Slurm '
        'job, clock/power change, quantization, multi-GPU run or architecture modification is included.\n')
    update_job(root, 'J0', 'complete', 'current allocation/runtime inventory and CPU contracts qualified; native families retain separate gates')
    checkpoint(root, 'J0: verify fresh H200 allocation, runtime and capability ledger')


def materialize(root, model):
    target = root/'configs'/f'{model}_smoke.json'
    if target.exists():
        assert Path(load(target)['model_path']).is_dir(), 'recorded model cache unavailable; rematerialize with a new attempt'
        from shoreline_accounting import archive_headers
        archive_headers(root,model,load(target))
        return target
    prior = load(PACKAGE/'h200/20261016_job14073758/configs'/f'{model}_smoke.json')
    outcome = run(root, f'download_{model}', python_command('materialize_config.py', '--model', model,
                  '--stage', 'smoke', '--download', '--revision', prior['model_revision'], '--output', target), 1800)
    assert outcome['status'] == 'complete', f'{model} materialization failed'
    if model == 'mistral7b':
        outcome = run(root, 'mistral_tokenizer_proof', python_command('resolve_mistral_tokenizer.py'), 120)
        assert outcome['status'] == 'complete'
    from shoreline_accounting import archive_headers
    archive_headers(root,model,load(target))
    checkpoint(root, f'J1 checkpoint: preserve pinned {model} materialization')
    return target


def make_config(root, model, engine, batch, context, trials=10):
    config = load(materialize(root, model))
    config.update(batch_sizes=[batch], context_lengths=[context], warmup_trials=3, trials=trials,
                  compile=False, energy_cases=[[batch,context]])
    config['engine'].update(max_num_seqs=max(8,batch), max_num_batched_tokens=max(8192,batch*context),
                            enforce_eager=engine in ('eager','hf_eager'))
    if engine == 'eager':
        config['engine']['compilation_config'] = {'mode':0}
    config['shoreline_protocol'] = {'acquisitions':3,'trials_per_acquisition':trials,'seed':20261013}
    conditioning=load(root/'session.json').get('decode_conditioning')
    if conditioning:config['decode_conditioning']=conditioning
    name = f'{model}_{engine}_b{batch}_c{context}_t{trials}'
    path = root/'configs'/(name+'.json')
    if path.exists():
        assert load(path) == config, 'config changed; new identity required'
    else:
        dump(path, config)
    return path


def clock_qualification(run_data, monitor):
    values = {'sm_clock_mhz': [], 'memory_clock_mhz': []}
    trials = [t for c in run_data['cases'] for t in c['trials']]
    for t in trials:
        for endpoint in ('hardware_before','hardware_after'):
            for metric in values:
                v = (t.get(endpoint) or {}).get(metric)
                if not isinstance(v, (int,float)) or v <= 0:
                    return {'passed':False,'reason':'missing/zero per-trial clock endpoint'}
                values[metric].append(v)
    rows = [json.loads(line) for line in Path(monitor).read_text().splitlines()]
    for row in rows:
        if any(t['wall_start_unix_ns'] <= row['unix_ns'] <= t['wall_end_unix_ns'] for t in trials):
            for metric in values:
                v = row.get(metric)
                if not isinstance(v,(int,float)) or v <= 0:
                    return {'passed':False,'reason':'missing/zero interior clock'}
                values[metric].append(v)
    extrema = {k:{'minimum':min(v),'maximum':max(v),'max_min_ratio':max(v)/min(v),'samples':len(v)}
               for k,v in values.items()}
    return {'passed':all(x['max_min_ratio'] <= 1.05 for x in extrema.values()),
            'clocks':extrema,'reason':'predeclared <=5% clock envelope; all trials retained'}


def acquisition(root, model, engine, batch, context, index, smoke=False):
    config = make_config(root,model,engine,batch,context,3 if smoke else 10)
    identity = f'{model}_{engine}_b{batch}_c{context}'+('_smoke' if smoke else f'_acq{index:02d}')
    qualification = root/'results/qualifications'/(identity+'.json')
    first_attempt=0;replacement=None
    if qualification.exists():
        previous=load(qualification)
        prior_process=root/'results/processes'/(identity+f'_attempt{previous["attempt"]}.json')
        contaminated=(previous.get('command_status')=='rejected' and prior_process.exists()
                      and load(prior_process).get('monitor',{}).get('failure')=='foreign GPU process detected')
        if contaminated and previous['attempt']<2:
            first_attempt=previous['attempt']+1;replacement=previous
        else:return previous
    outcome = None
    retry_limit=load(root/'session.json').get('rejected_acquisition_retry_limit',2)
    for attempt in range(first_attempt,min(3,first_attempt+retry_limit+1)):
        label = identity+f'_attempt{attempt}'
        output = root/'runs'/label
        script = 'software_decode_experiments.py' if engine == 'hf_eager' else 'controlled_vllm_experiments.py'
        saved = root/'results/processes'/(label+'.json')
        if saved.exists():
            outcome = load(saved)
        else:
            outcome = run(root,label,python_command(script,'--config',config,'--gpu-uuid',UUID,
                          '--mode','timing','--output',output,'--batch',str(batch),'--context',str(context)),1200,gpu=True)
        result = {'id':identity,'attempt':attempt,'output':str(output.relative_to(root)),
                  'status':'failed','source_config_sha256':sha(config),'command_status':outcome['status']}
        if replacement:
            result['replaces_contaminated_attempt']=replacement
            result['selection_rule']='replacement for detected interference, never selected by latency; prior raw attempt retained'
        path = output/'run.json'
        if path.exists():
            data = load(path)
            result['run_sha256'] = sha(path)
            if outcome['status'] == 'complete' and data['status'] == 'complete':
                if engine == 'hf_eager':
                    result.update(status='passed',qualification_scope='independent strict StaticCache smoke; no clock-matched timing join')
                else:
                    cq = clock_qualification(data,outcome['monitor']['source'])
                    result.update(clock_qualification=cq,status='passed' if cq['passed'] else 'rejected')
                    if smoke:
                        result.update(status='passed',qualification_scope='correctness-only smoke; clock failure cannot qualify a timing join')
        dump(root/'results/qualifications'/(label+'.json'), result)
        checkpoint(root, f'J1 acquisition checkpoint: {label} {result["status"]}')
        if result['status'] == 'passed' or outcome['status'] == 'pending' or engine == 'hf_eager':
            break
    dump(qualification, result)
    return result


def ensure_model_smoke(root,model):
    hf = acquisition(root,model,'hf_eager',1,128,0,smoke=True)
    graph = acquisition(root,model,'graph',1,128,0,smoke=True)
    comparison = root/'results'/(model+'_smoke_exact_ids.json')
    if not comparison.exists() and hf['status'] == graph['status'] == 'passed':
        run(root,model+'_cross_engine_smoke',python_command('compare_vllm_hf_tokens.py',
            '--vllm-run',root/graph['output']/'run.json','--hf-run',root/hf['output']/'run.json',
            '--output',comparison),120)
    return comparison.exists() and load(comparison)['status']=='passed'


def baselines(root):
    update_job(root,'J1','in_progress')
    session=load(root/'session.json')
    contrasts = session.get('baseline_contrasts',
        [('bridge',1,1024),('bridge',8,4096),('mistral7b',1,4096),('mistral7b',8,4096),('qwen14b',1,4096)])
    for model in session.get('smoke_models',('bridge','mistral7b','qwen14b')):
        if not ensure_model_smoke(root,model):
            checkpoint(root,f'J1: preserve {model} blocked smoke; continue independent qualified models')
            continue
        for m,b,c in contrasts:
            if m != model:
                continue
            for engine in ('graph','eager'):
                for index in range(3):
                    acquisition(root,model,engine,b,c,index)
    analyze_baselines(root)
    manifest = load(root/'results/baseline_qualification.json')
    update_job(root,'J1','complete' if manifest['all_conditions_qualified'] else 'partial',
               'per-condition exact IDs, acquisition count, pooled clocks and mode joins in baseline_qualification.json; timelines are separate')
    checkpoint(root,'J1: publish focused H200 baseline qualification and remaining gates')


def analyze_baselines(root):
    conditions=[]
    for model,b,c in [('bridge',1,1024),('bridge',8,4096),('mistral7b',1,4096),('mistral7b',8,4096),('qwen14b',1,4096)]:
        modes={}
        for engine in ('graph','eager'):
            rows=[]; ids=[]; extrema={k:[] for k in ('sm_clock_mhz','memory_clock_mhz')}
            for index in range(3):
                p=root/'results/qualifications'/f'{model}_{engine}_b{b}_c{c}_acq{index:02d}.json'
                if not p.exists():continue
                q=load(p)
                if q['status']!='passed':continue
                data=load(root/q['output']/'run.json');case=data['cases'][0]
                ids.append(case['trials'][0]['output_sha256'])
                for k in extrema:
                    v=q['clock_qualification']['clocks'][k];extrema[k]+=[v['minimum'],v['maximum']]
                rows.append({'qualification':str(p.relative_to(root)), 'output':q['output'],
                             'step_ms':[t['wall_ms']/t['decode_iterations'] for t in case['trials']]})
            pooled={k:max(v)/min(v) for k,v in extrema.items() if v}
            valid=len(rows)==3 and len(set(ids))==1 and len(pooled)==2 and all(v<=1.05 for v in pooled.values())
            modes[engine]={'qualified':valid,'acquisitions':rows,'pooled_clock_ratio':pooled,'output_hashes':ids}
        conditions.append({'model':model,'batch':b,'context':c,'modes':modes,
                           'exact_graph_eager_match':bool(modes['graph']['output_hashes'] and modes['graph']['output_hashes']==modes['eager']['output_hashes'])})
    dump(root/'results/baseline_qualification.json',{'conditions':conditions,
         'all_conditions_qualified':all(all(m['qualified'] for m in c['modes'].values()) and c['exact_graph_eager_match'] for c in conditions),
         'confidence_intervals':'pending hierarchical bootstrap; fewer than three acquisitions cannot supply CI'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--job',choices=['J0','J1'],required=True)
    args=parser.parse_args()
    os.environ['H200_SESSION_CONFIG']=str(args.session.resolve())
    from h200_session import DEFAULT_ROOT,PACKAGE,PYTHON,UUID,dump,run,python_command,environment,verify_allocation
    from h200_session import CUDA
    NCU=str(CUDA/'bin/ncu')
    verify_allocation()
    (inventory if args.job=='J0' else baselines)(DEFAULT_ROOT)
