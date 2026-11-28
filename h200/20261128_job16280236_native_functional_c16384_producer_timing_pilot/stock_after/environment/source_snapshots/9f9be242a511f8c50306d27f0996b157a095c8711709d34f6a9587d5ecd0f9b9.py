#!/usr/bin/env python3
"""CPU qualification of finite functional captures, never clean serving timing."""
import argparse
import math
import traceback
from pathlib import Path
from h200_priority_campaign import load,write,sha,valid_tokens,strict_reference_record,limitation
from h200_serving_trace import make_trace,trial_seed,summarize
from h200_serving_policy import PINNED_SCHEDULER_SHA256

POLICIES=('whole_separate','chunked_separate','chunked_mixed')


def close(a,b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9


def check_requests(records,expected,offered):
    assert len(records)==len(offered)
    assert len(expected)==64
    for record,arrival in zip(records.values(),offered):
        close(record['offered_s'],arrival)
        assert record['output_ids']==expected
        assert isinstance(record['internal_request_id'],str)
        stamps=record['token_observed_s'];gaps=record['inter_token_s']
        assert len(stamps)==64 and len(gaps)==63
        assert all(math.isfinite(v) for v in stamps+gaps)
        assert record['admitted_s']>=arrival and stamps[0]>=record['admitted_s']
        assert all(a<=b for a,b in zip(stamps,stamps[1:]))
        close(record['first_token_s'],stamps[0]);close(record['finished_s'],stamps[-1])
        close(record['ttft_s'],stamps[0]-arrival)
        for a,b,gap in zip(stamps,stamps[1:],gaps):close(b-a,gap)


def check_schedule(data,context,policy,sources=None):
    scheduler=data['scheduler_rows'];policies=data['policy_rows'];steps=data['steps']
    assert scheduler and len(scheduler)==len(policies)==len(steps)
    assert data['policy_identity']['policy']==policy
    assert data['policy_identity']['scheduler_source_sha256']==PINNED_SCHEDULER_SHA256
    assert data['observer_identity']['source_sha256']==PINNED_SCHEDULER_SHA256
    if sources is not None:
        assert data['policy_identity']['adapter_source_sha256']==sources['h200_serving_policy.py']
        assert data['observer_identity']['observer_source_sha256']==sources['h200_scheduler_observer.py']
    assert data['classification']=='measured-real-GPU-instrumented'
    internal={r['internal_request_id'] for r in data['requests'].values()}
    observed=set()
    for row,policy_row,step in zip(scheduler,policies,steps):
        assert not row['any_preempted_request']
        assert row['composition']==policy_row['composition']==step['composition']
        assert math.isfinite(step['wall_ms']) and step['wall_ms']>0
        assert math.isfinite(step['gpu_event_ms']) and step['gpu_event_ms']>=0
        if policy!='chunked_mixed':assert row['composition']!='mixed'
        assert sum(r['prompt_input_queries'] for r in row['requests'])==row['prompt_input_queries']
        assert sum(r['post_prompt_input_queries'] for r in row['requests'])==row['post_prompt_input_queries']
        for request in row['requests']:
            observed.add(request['request_id'])
            assert request['request_id'] in internal and request['original_prompt_tokens']==context
            assert request['num_preemptions']==0
            assert request['post_prompt_input_queries'] in (0,1)
            assert request['scheduled_input_queries']==request['prompt_input_queries']+request['post_prompt_input_queries']
            if request['prompt_input_queries']:
                assert request['post_prompt_input_queries']==0
                if policy=='whole_separate':
                    assert request['computed_tokens_before_step']==0 and request['prompt_input_queries']==context
    assert observed==internal
    assert sum(r['prompt_input_queries'] for r in scheduler)==context*len(internal)
    assert sum(r['post_prompt_input_queries'] for r in scheduler)==63*len(internal)
    return {'steps':len(steps),'compositions':{name:sum(r['composition']==name for r in scheduler)
            for name in ('prefill_only','decode_only','mixed','empty')}}


def review(root):
    root=Path(root).resolve();errors=[];checked=[]
    session=load(root/'session.json');disposition=load(root/'results/serving_pilot_disposition.json')
    try:
        assert disposition['status']=='functional_captured_pending_CPU_review'
        assert disposition['source_identity']==session['frozen_source_identity']
        assert disposition['gpu_uuid']==session['gpu_uuid']
        source_path=Path(disposition['run']);source=load(source_path)
        assert sha(source_path)==disposition['run_sha256'] and valid_tokens(source)
        assert source['baseline_comparison_eligible'] is False
        assert source['classification']=='measured-real-GPU-instrumented'
        assert source['engine_settings']['enable_prefix_caching'] is False
        assert source['config']['precision']=='BF16' and source['config']['seed']==20261013
        reference=load(disposition['strict_HF_reference'])
        assert sha(disposition['strict_HF_reference'])==disposition['strict_HF_reference_sha256']
        assert strict_reference_record(reference,sha(source_path))
        for label in ('P3a_openloop_development_pilot','P3a_openloop_development_pilot_hf'):
            process=load(root/'results/processes'/(label+'.json'))
            assert process['status']=='complete' and not process['monitor']['failure']
            assert process['source_freeze_identity']==session['frozen_source_identity']
        manifest_path=Path(disposition['manifest']);assert sha(manifest_path)==disposition['manifest_sha256']
        manifest=load(manifest_path);assert manifest['protocol']=='finite_open_loop_policy_v2'
        assert manifest['pilot'] and manifest['status']=='captured_pending_CPU_and_intrusion_review'
        assert manifest['acquisition']==0 and manifest['quantitative_qualified'] is False
        assert manifest['fresh_functional_policy_gate']['status']=='three_functional_policies_passed'
        expected=source['cases'][0]['trials'][0]['output_ids'][0]
        context=source['cases'][0]['context_length']
        assert context==4096 and manifest['tokens']==64 and manifest['request_count']==12
        assert len(manifest['captures'])==18
        assert len(manifest['fresh_functional_policy_gate']['paths'])==3
        combinations={(r['phase'],r['trial'],r['policy']) for r in manifest['captures']}
        assert combinations=={(phase,trial,policy) for phase in ('warmup','functional') for trial in range(3) for policy in POLICIES}
        for path,policy in zip(manifest['fresh_functional_policy_gate']['paths'],POLICIES):
            data=load(path);assert data['status']=='functional_policy_and_exact_outputs_passed'
            check_requests(data['requests'],expected,[0.,.01,.02]);check_schedule(data,context,policy,source['script_sha256'])
            if policy=='chunked_mixed':assert any(r['composition']=='mixed' for r in data['scheduler_rows'])
        for row in manifest['captures']:
            path=Path(row['path']);assert sha(path)==row['sha256']
            data=load(path);assert data['status']=='finite_trace_exact_outputs_and_composition_passed'
            seed=trial_seed(0,row['trial'],'warmup' if row['phase']=='warmup' else 'development')
            trace=make_trace(seed,manifest['rate'],12,'warmup' if row['phase']=='warmup' else 'functional')
            assert trace==data['trace'] and trace['sha256']==row['trace_sha256']
            check_requests(data['requests'],expected,trace['offered_s'])
            accounting=check_schedule(data,context,row['policy'],source['script_sha256'])
            assert data['slo']==manifest['slo']
            assert data['summary']==summarize(data['requests'],data['summary']['observation_wall_s'],data['slo'])
            assert data['summary']['all_requests_completed'] and not data['summary']['steady_state_demonstrated']
            assert data['clean_serving_qualified'] is False and data['primary_J_per_token'] is None
            checked.append({'phase':row['phase'],'trial':row['trial'],'policy':row['policy'],
                'requests':len(data['requests']),'path':str(path),'sha256':sha(path),**accounting})
    except (OSError,ValueError,KeyError,AssertionError,TypeError,IndexError,AttributeError) as error:
        errors.append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    result={'classification':'CPU review of finite functional policy evidence','functional_evidence_valid':not errors,
        'quantitative_qualified':False,'ordinary_serving_or_SLO_qualified':False,
        'independent_acquisitions':1,'completed_development_traces':len(checked),
        'checked_traces':checked,'errors':errors,'gpu_uuid':session['gpu_uuid'],
        'source_identity':session['frozen_source_identity'],'session_sha256':sha(root/'session.json'),
        'disposition_sha256':sha(root/'results/serving_pilot_disposition.json'),
        'scope':'Fresh targeted3policy proof plus finite development trace accounting only. Not3independent acquisitions, held-out judgments, tuning, steady state, overhead-neutral timing, clean SLOs or energy.'}
    write(root/'results/serving_pilot_CPU_review.json',result)
    with (root/'analysis/report.md').open('a') as stream:
        stream.write('\n## CPU review\n\nFunctional evidence valid: '+str(not errors)+'. '+result['scope']+' Evidence: `../results/serving_pilot_CPU_review.json`.\n')
    if errors:
        limitation(root,'serving_pilot_CPU_review','Finite functional capture failed CPU provenance/accounting checks; first error type: '+errors[0]['type'],
            'results/serving_pilot_CPU_review.json','Strict full-output/source/phase/arrival/scheduler checks; all failed traces retained',
            'Inspect first error and preserve raw capture before a new bounded cohort; no quantitative result inferred')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();result=review(args.root)
    print({k:result[k] for k in ('functional_evidence_valid','completed_development_traces','quantitative_qualified')})
