#!/usr/bin/env python3
"""Three-request functional serving-policy pilot, never a load/SLO result."""
import argparse
import json
from pathlib import Path
import time
import traceback


def update_request(record,output,elapsed,expected):
    """One nonspeculative token per observed output; offered time owns TTFT."""
    previous=record['output_ids']
    ids=list(output.outputs[0].token_ids)
    record['output_ids']=ids
    record['token_observed_s'].append(elapsed)
    assert ids[:len(previous)]==previous and len(ids)==len(previous)+1
    assert ids==expected[:len(ids)],'greedy output differs from serial reference'
    if len(ids)==1:
        record.update(first_token_s=elapsed,ttft_s=elapsed-record['offered_s'],
                      admission_to_first_token_s=elapsed-record['admitted_s'])
    if output.finished:
        record['finished_s']=elapsed
        assert len(ids)==len(expected) and output.outputs[0].finish_reason=='length'
    else:assert len(ids)<len(expected)


def scenario(engine,prompts,sampling,torch,output,policy_name,expected):
    from software_decode_experiments import dump
    from h200_serving_policy import Policy
    from h200_scheduler_observer import Observer
    assert len(prompts)==1 and not engine.has_unfinished_requests()
    policy=Policy(engine,policy_name,8192 if policy_name=='whole_separate' else 256)
    try:observer=Observer(engine)
    except BaseException:
        policy.close();raise
    requests={f'{policy_name}_{i}':{'offered_s':offset,'admitted_s':None,'output_ids':[],
                                 'token_observed_s':[]} for i,offset in enumerate((0.,.01,.02))}
    ledger={'status':'running','classification':'measured-real-GPU-instrumented',
            'policy_identity':policy.identity,'observer_identity':observer.identity,'requests':requests,
            'policy_rows':policy.rows,'scheduler_rows':observer.rows,'steps':[],
            'scope':'three fixed requests, one prospectively fixed arrival pattern; functional policy proof only, no throughput/SLO/energy or independent-trace comparison'}
    def save():dump(output/(policy_name+'.json'),ledger)
    save();start=time.perf_counter();steps=0
    try:
        while any(r['admitted_s'] is None or 'finished_s' not in r for r in requests.values()):
            elapsed=time.perf_counter()-start
            assert elapsed<180,'bounded policy scenario deadline reached'
            for rid,record in requests.items():
                if record['admitted_s'] is None and record['offered_s']<=elapsed:
                    record['internal_request_id']=engine.add_request(rid,{'prompt_token_ids':prompts[0]},sampling)
                    record['admitted_s']=time.perf_counter()-start
            if not engine.has_unfinished_requests():
                assert any(r['admitted_s'] is None for r in requests.values()),'engine lost an unfinished admitted request'
                time.sleep(.001);continue
            before=len(observer.rows)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();step_start=time.perf_counter()
            outputs=engine.step()
            end.record();torch.cuda.synchronize()
            step_end=time.perf_counter()
            assert len(observer.rows)==before+1,'one synchronous engine step must have one schedule decision'
            ledger['steps'].append({'start_s':step_start-start,'end_s':step_end-start,
                'wall_ms':(step_end-step_start)*1000,'gpu_event_ms':begin.elapsed_time(end),
                'composition':observer.rows[-1]['composition'],'output_requests':[o.request_id for o in outputs]})
            for result in outputs:
                assert result.request_id in requests
                update_request(requests[result.request_id],result,step_end-start,expected)
            steps+=1;assert steps<=1024,'policy failed bounded forward progress'
            save()
        assert not engine.has_unfinished_requests()
        assert all(r['output_ids']==expected for r in requests.values())
        assert not any(r['any_preempted_request'] for r in observer.rows),'capacity/preemption needs a separate declared pilot'
        compositions={r['composition'] for r in observer.rows}
        if policy_name=='chunked_mixed':assert 'mixed' in compositions,'mixed pilot never executed a mixed step'
        else:assert 'mixed' not in compositions
        for record in requests.values():
            timestamps=record['token_observed_s']
            record['inter_token_s']=[b-a for a,b in zip(timestamps,timestamps[1:])]
            assert len(timestamps)==64 and len(record['inter_token_s'])==63
        ledger['status']='functional_policy_and_exact_outputs_passed'
        return ledger
    finally:
        observer.close();policy.close();save()


def main(a):
    import controlled_vllm_experiments as baseline
    from software_decode_experiments import dump
    original=baseline.run_trial;captured=[]
    def trial(engine,prompts,sampling,prefix,torch,output_dir,*args,**kwargs):
        result=original(engine,prompts,sampling,prefix,torch,output_dir,*args,**kwargs)
        if 'measured0' in prefix:
            assert not captured and len(prompts)==1 and sampling.max_tokens==64
            for name in ('whole_separate','chunked_separate','chunked_mixed'):
                captured.append(scenario(engine,prompts,sampling,torch,output_dir,name,result['output_ids'][0]))
            dump(output_dir/'policy_pilot.json',{'status':'three_functional_policies_passed',
                 'conditions':[{'policy':r['policy_identity']['policy'],'status':r['status']} for r in captured],
                 'scope':'one resident optimized full-GPU model, three requests per policy; not loaded serving, tuning, steady state, SLOs, energy or three independent acquisitions'})
        return result
    baseline.run_trial=trial
    try:baseline.main(a)
    finally:baseline.run_trial=original
    assert len(captured)==3
    path=a.output/'run.json';data=json.loads(path.read_text())
    data.update(baseline_comparison_eligible=False,classification='measured-real-GPU-instrumented',
                experimental_scope='functional policy pilot; serial controls are correctness references, not a clean timing comparison')
    dump(path,data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',default='timing')
    p.add_argument('--generated-tokens',type=int,default=64);p.add_argument('--batch',type=int);p.add_argument('--context',type=int)
    p.add_argument('--external-profiler',action='store_true');p.add_argument('--collect-worker-metadata',action='store_true')
    a=p.parse_args();assert not a.external_profiler and not a.collect_worker_metadata
    a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except BaseException as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
