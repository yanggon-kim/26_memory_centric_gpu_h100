#!/usr/bin/env python3
"""Version3 bounded diagnostics and conditional new-program acquisitions."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace


def exact(run,expected):
    cases=run.get('cases',[])
    return (run.get('status')=='complete' and len(cases)==1 and bool(cases[0].get('trials')) and
        all(t['output_ids']==expected for t in cases[0]['trials']) and
        all(ids==expected for ids in cases[0].get('warmup_output_ids',[])))


def main(session):
    root=session.resolve().parent;package=Path(__file__).resolve().parents[1]
    os.environ['H200_SESSION_CONFIG']=str(session.resolve())
    import h200_session as env
    import shoreline_campaign as campaign
    import shoreline_jobs as jobs
    from shoreline_treatment_analysis import analyze as treatments
    env.verify_allocation()
    for key in ('PACKAGE','PYTHON','UUID','dump','run','python_command'):
        setattr(campaign,key,getattr(env,key))
        setattr(jobs,key,getattr(env,key))
    jobs.ROOT=root;jobs.CUDA=env.CUDA;jobs.NCU=str(env.CUDA/'bin/ncu')
    jobs.PROGRAMS=Path(env.SESSION['frozen_source_root'])/'scripts'
    jobs.campaign=campaign;jobs.args=SimpleNamespace(job='J1')
    jobs.CONTRASTS=[('bridge',8,4096)]
    state={'status':'running','protocol':'failure_recovery_v3','steps':[],
        'interpretation':'new conditioned-decode program; original rejections retained; no claim of unconditioned latency repair'}
    state_path=root/'results/campaign_queue.json'
    def save():env.dump(state_path,state)
    def publish(message):campaign.checkpoint(root,message)
    def stage(name,function):
        row={'id':name,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['steps'].append(row);save()
        try:row['result']=function();row.update(status='command_complete',exit_code=0)
        except Exception as error:
            row.update(status='command_failed',exit_code=1,error=str(error),traceback=traceback.format_exc())
        row['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        publish(f'Failure recovery checkpoint: {name} {row["status"]}')
        return row['status']=='command_complete' and row.get('result') is True
    save()

    def correctness():
        # Reproduce actual vLLM outputs with scores; this process has no conditioning.
        path=campaign.make_config(root,'bridge','eager',1,4096,3)
        cfg=jobs.load(path);cfg.pop('decode_conditioning',None);cfg['diagnostic_logprobs']=20
        target=root/'configs/D3_c4096_logprobs.json';env.dump(target,cfg)
        output=root/'runs/D3_c4096_logprobs'
        row=env.run(root,'D3_c4096_logprobs',env.python_command('controlled_vllm_experiments.py',
            '--config',target,'--gpu-uuid',env.UUID,'--output',output,'--mode','timing'),600)
        if row['status']!='complete':return False
        # Report whether requesting scores changed the original program's outputs.
        original=package/'h200/20261019_job14177052/runs'
        prior=next(p for p in original.glob('*') if p.name=='bridge_eager_b1_c4096_smoke_attempt0')/'run.json'
        native=jobs.load(output/'run.json');old=jobs.load(prior)
        identity=all(native['config'][k]==old['config'][k] for k in ('model_id','model_revision','seed','precision'))
        same_prompt=native['cases'][0]['prompt_sha256']==old['cases'][0]['prompt_sha256']
        env.dump(root/'results/D3_logprobs_reproduction.json',{'same_model_identity':identity,'same_prompt':same_prompt,
            'original_source':str(prior.relative_to(package)),'original_source_sha256':jobs.digest(prior),
            'exact_original_vllm_ids':identity and same_prompt and exact(native,old['cases'][0]['trials'][0]['output_ids']),
            'scope':'score-requesting execution is diagnostic and cannot replace the original sampling program'})
        outcome=env.run(root,'D3_cache_backend_diagnosis',env.python_command('diagnose_decode_correctness.py',
            '--source-run',output/'run.json','--gpu-uuid',env.UUID,
            '--historical-hf-run',original/'bridge_hf_eager_b1_c4096_smoke_attempt0/run.json',
            '--output',root/'runs/D3_cache_backend_diagnosis'),900)
        return outcome['status']=='complete'
    stage('C4096_correctness_diagnosis',correctness)

    def references():
        for mode in ('hf_eager','eager','graph'):
            q=campaign.acquisition(root,'bridge',mode,8,4096,0,smoke=True)
            if q.get('status')!='passed':return False
        return jobs.functional_smoke_reference(root,'bridge','eager',8,4096) is not None and jobs.functional_smoke_reference(root,'bridge','graph',8,4096) is not None
    ready=stage('B8_functional_references',references)

    def pilot(label,engine,script,extra=(),generated=64,energy=False):
        cfg=campaign.make_config(root,'bridge',engine,8,4096,3)
        output=root/'runs'/label
        row=env.run(root,label,env.python_command(script,'--config',cfg,'--gpu-uuid',env.UUID,'--output',output,
            '--mode','energy' if energy else 'timing','--generated-tokens',str(generated),*extra),600)
        result={'id':label,'status':'failed','new_program':'memory_settle_v3','timing_comparison_eligible':False}
        if row['status']=='complete' and (output/'run.json').exists():
            d=jobs.load(output/'run.json');trials=d['cases'][0]['trials']
            ref=jobs.load(jobs.functional_smoke_reference(root,'bridge',engine,8,4096))['cases'][0]['trials'][0]['output_ids']
            exact_ids=d['status']=='complete' and all([x[:64] for x in t['output_ids']]==ref for t in trials)
            clock=campaign.clock_qualification(d,row['monitor']['source'])
            conditioned=all(t.get('post_prefill_conditioning',{}).get('protocol')=='memory_settle_v3' for t in trials)
            sensors=not energy or all(isinstance(t.get('telemetry',{}).get('joules_per_token'),(float,int)) for t in trials)
            result.update(status='complete' if exact_ids and clock['passed'] and conditioned and sensors else 'rejected',
                clock_qualification=clock,exact_ids=exact_ids,conditioned=conditioned,sensor_gate=sensors,
                source_freeze_identity=row.get('source_freeze_identity'))
        env.dump(root/'results'/(label+'.json'),result)
        publish(f'Bounded failure-recovery pilot: {label} {result["status"]}')
        return result['status']=='complete'

    eager=ready and stage('eager_clock_pilot',lambda:pilot('D3_eager_clock','eager','controlled_vllm_experiments.py'))
    graph=ready and stage('graph_clock_pilot',lambda:pilot('D3_graph_clock','graph','controlled_vllm_experiments.py'))
    if eager:
        def competitor_pilots():
            for treatment in ('idle','memory100','arithmetic100'):
                if not pilot('D3_competition_'+treatment,'eager','shoreline_competition.py',('--treatment',treatment)):return False
            return True
        if stage('J6_conditioned_pilots',competitor_pilots):
            jobs.args.job='J6';stage('J6_conditioned_acquisitions',lambda:jobs.competition())
        def event_pilots():
            for mode,flag in [('none',()),('empty',('--empty-layer-events',)),('complete',('--layer-events',))]:
                if not pilot('D3_events_'+mode,'eager','shoreline_decode.py',('--ordinary-control','--pressure','normal',*flag)):return False
            return True
        if stage('J8_conditioned_pilots',event_pilots):
            jobs.args.job='J8';stage('J8_conditioned_acquisitions',lambda:jobs.layer_events())
    if graph and stage('J10_conditioned_pilot',lambda:pilot('D3_energy_clock','graph','controlled_vllm_experiments.py',generated=2048,energy=True)):
        jobs.args.job='J10';stage('J10_conditioned_acquisitions',lambda:jobs.energy())

    def analysis():
        from shoreline_analyze import main as analyze
        analyze(root);treatments(root)
        for script in ('consolidate_h200.py','plot_shoreline.py'):
            command=[sys.executable,str(package/'scripts'/script)]
            if script=='consolidate_h200.py':command+=['--output',str(root/'analysis/consolidated'),'--repair-root',str(root)]
            else:command+=['--root',str(root)]
            subprocess.run(command,check=True)
        env.dump(root/'results/failure_recovery_disposition.json',{
            'steps':state['steps'],'original_gates_unchanged':True,
            'scope':'memory_settle_v3 changes cache history/TTFT; new qualified conditioned results must not be pooled with original unconditioned runs',
            'unresolved':['unconditioned DVFS','C4096 exact outputs unless diagnosis proves original route correct',
                'J4 held-out model','graph SM restriction','Hopper trace semantics/calibration','private archive access']})
        return True
    stage('CPU_analysis_and_consolidation',analysis)
    state['status']='commands_finished_scientific_gates_separate' if all(s['status']=='command_complete' for s in state['steps']) else 'needs_diagnosis'
    save();publish('Failure recovery finished: publish diagnostics, conditioned-program gates and unresolved failures')
    if state['status']=='needs_diagnosis':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    main(p.parse_args().session)
