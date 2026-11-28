#!/usr/bin/env python3
"""Independent CPU reader for matched producer host-timing pilots.

A pilot never qualifies a3acquisition performance comparison. Raw host times
are retained even if clock, drift, source or completion gates reject a join.
"""
import argparse
from pathlib import Path
import statistics
from h200_priority_campaign import load,write,sha,limitation,valid_tokens,strict_reference_record
from h200_producer_timing_role import descriptor_scope,checked_outputs
from h200_native_model_pilot import validate_queries
from h200_review_native_model import verified_process

LABEL='matched_producer_timing'
ENTRY='h200_producer_timing_role.py'


def raw_energy(row):
    raw=row['raw_energy_counter'];assert raw['counter_unit']=='mJ' and raw['primary_J_per_token'] is None
    before=raw['before'];after=raw['after']
    for readout in (before,after):
        assert readout['query_start_unix_ns']<=readout['query_end_unix_ns']
        if readout['status']=='available':assert type(readout['energy_mj']) is int and readout['energy_mj']>=0
        else:assert readout['status']=='unavailable' and readout['energy_mj'] is None and readout['error']
    assert before['query_end_unix_ns']<=row['submit_unix_ns']
    assert after['query_start_unix_ns']>=row['final_step_return_unix_ns']
    assert after['query_end_unix_ns']<=row['completed_unix_ns']
    present=before['status']==after['status']=='available'
    delta=after['energy_mj']-before['energy_mj'] if present else None
    return {'status':'available_unqualified' if present and delta>=0 else 'counter_reset' if present else 'counter_unavailable',
        'raw_counter_delta_mj':delta,'primary_J_per_token':None,
        'scope':'Per-role lifecycle counter difference only, including waiting. No aligned primary/all-board window, update cadence or counter/power agreement qualification; no rescaling.'}


def checked_certificate(path,pair,session):
    cert=load(path)
    assert cert['status']=='native_model_functional_review_passed' and cert['job_id']==session['slurm_job_id']
    original=load(path.parents[1]/'pair.json')
    assert original['context']==pair['context'] and original['reference_run_sha256']==pair['reference_run_sha256']
    assert len(cert['roles'])==2 and len(cert['reference_process_provenance'])==2
    dependencies=None
    for role in cert['roles']:
        for field in ('session','run','process','monitor'):assert sha(Path(role[field]))==role[field+'_sha256']
        owned=load(Path(role['session']))
        assert owned['slurm_job_id']==session['slurm_job_id'] and owned['frozen_source_identity']==session['frozen_source_identity']
        assert role['gpu_uuid']==pair['roles'][role['role']]['gpu_uuid']
        native=load(Path(role['run']))
        if dependencies is None:dependencies=native['optional_dependencies']
        assert native['optional_dependencies']==dependencies
    for provenance in cert['reference_process_provenance']:
        for field in ('process','monitor','log','idle_check'):assert sha(Path(provenance[field]))==provenance[field+'_sha256']
        assert sha(Path(provenance['entry_snapshot']))==provenance['entry_sha256']
    assert cert['implementation_sha256']==sha(Path(session['frozen_source_root'])/'scripts/h200_review_native_model.py')
    return dependencies


def check_trials(run,pair,expected):
    role=run['role'];count=pair['warmup_trials']+pair['timed_trials']
    assert run['status']=='complete' and len(run['trials'])==count
    assert run['mode']==pair['mode'] and run['protocol']==pair['protocol']
    assert run['acquisition']==pair['acquisition']
    primary=[];first=[];decoder=[];readiness=[];energy=[]
    for index,row in enumerate(run['trials']):
        assert row['trial']==index and row['warmup'] is (index<3)
        assert row['status']=='exact_output_passed' and row['observer_present'] is (index<3)
        steps=row['host_step_intervals'];outputs=row['outputs']
        assert len(steps)==len(outputs)==(63 if role=='consumer' else 1)
        assert row['submit_unix_ns']<=steps[0]['start_unix_ns']
        assert row['final_step_return_unix_ns']==steps[-1]['end_unix_ns']
        assert row['submit_to_final_step_return_ns']>0
        energy.append(raw_energy(row))
        for step,output in zip(steps,outputs):
            assert step['start_unix_ns']<=step['end_unix_ns']==output['observed_unix_ns']
            assert step['elapsed_ns']>0
            checked_outputs(output['output_ids'],expected,role)
        assert all(a['end_unix_ns']<=b['start_unix_ns'] for a,b in zip(steps,steps[1:]))
        if role=='consumer':
            assert expected[:1]+outputs[-1]['output_ids']==expected
            assert row['first_token_file_read_unix_ns']<=row['submit_unix_ns']
        elif role=='producer':
            assert row['producer_first_step_return_unix_ns']==row['final_step_return_unix_ns']
            assert row['all_native_sends_observed_unix_ns']>=row['final_step_return_unix_ns']
            assert row['consumer_completion_unix_ns']>=row['final_step_return_unix_ns']
            assert row['native_sent_tensor_count']==64
        if index<3:
            validate_queries(row['warmup_scheduler_rows'],'producer' if role=='stock' else role,row['request_id'],pair['context'])
        else:
            assert row['warmup_scheduler_rows'] is None
            primary.append(row['submit_to_final_step_return_ns']/1e6)
            if role=='consumer':
                first.append(steps[0]['elapsed_ns']/1e6)
                decoder.extend(s['elapsed_ns']/1e6 for s in steps[1:])
            if role=='producer':readiness.append((row['all_native_sends_observed_unix_ns']-row['submit_unix_ns'])/1e6)
    return {'submit_to_final_step_return_host_ms':primary,'consumer_first_handoff_query_host_ms':first,
        'consumer_following62query_host_ms':decoder,'producer_all_native_sends_observed_host_ms':readiness,
        'raw_energy_per_warmup_and_timed_request':energy,
        'scope':'Raw host intervals. Consumer first query includes native handoff/load; following62are separate decode calls. No GPU-event timing, HTTP TTFT, serving goodput, physical link/HBM bytes or primary energy.'}


def check_pair(producer,consumer):
    assert len(producer['trials'])==len(consumer['trials'])
    for source,target in zip(producer['trials'],consumer['trials']):
        assert source['request_id']==target['request_id']
        assert source['producer_first_step_return_unix_ns']<=target['first_token_file_read_unix_ns']
        assert source['consumer_completion_unix_ns']==target['final_step_return_unix_ns']


def review(root):
    root=Path(root).resolve();captures=load(root/'results/captures.json');rows=[];errors=[];runs={}
    expected_keys={('stock_before','stock'),('native','producer'),('native','consumer'),('stock_after','stock')}
    assert captures['mode']=='pilot'
    for capture in captures['captures']:
        key=(capture['condition'],capture['role'])
        try:
            assert key in expected_keys and key not in runs
            base=Path(capture['base']);session=load(base/'session.json')
            assert sha(Path(capture['run']))==capture['run_sha256']
            assert sha(Path(capture['process']))==capture['process_sha256']
            provenance=verified_process(base,LABEL,ENTRY,session)
            run=load(Path(capture['run']));descriptor=base.parent/'pair.json' if capture['condition']=='native' else base/'pair.json'
            assert sha(descriptor)==run['pair_sha256'];pair=load(descriptor)
            descriptor_scope(pair,session,capture['role'],session['gpu_uuid'],session['slurm_job_id'],session['expected_hostname'])
            assert run['source_identity']==session['frozen_source_identity'] and run['gpu_uuid']==session['gpu_uuid']
            assert run['role']==capture['role'] and pair['mode']=='pilot'
            source=Path(pair['reference_run']);assert sha(source)==pair['reference_run_sha256']
            reference=load(source);assert valid_tokens(reference)
            assert reference['config']['model_id']=='Qwen/Qwen2.5-32B-Instruct'
            assert reference['config']['model_revision']=='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
            assert reference['config']['seed']==20261013 and not reference['config'].get('decode_conditioning')
            assert run['hardware']['uuid']==run['gpu_uuid']
            for metric in ('sm_count','l2_bytes','power_limit_w'):
                assert run['hardware'][metric]==reference['hardware'][metric]
            for trial in run['trials']:
                for endpoint in ('hardware_before','hardware_after'):
                    assert trial[endpoint]['uuid']==run['gpu_uuid']
                    assert trial[endpoint]['power_limit_w']==run['hardware']['power_limit_w']
            hf=Path(pair['strict_HF_reference']);assert sha(hf)==pair['strict_HF_reference_sha256']
            assert strict_reference_record(load(hf),sha(source))
            assert run['engine_settings_without_KV_object']==reference['engine_settings']
            assert run['versions']==reference['versions']
            cert=Path(pair['functional_CPU_review']);assert sha(cert)==pair['functional_CPU_review_sha256']
            assert run['optional_dependencies']==checked_certificate(cert,pair,session)
            metrics=check_trials(run,pair,reference['cases'][0]['trials'][0]['output_ids'][0])
            timed=[t for t in run['trials'] if not t['warmup']]
            windows=[dict(t,wall_start_unix_ns=t['submit_unix_ns'],wall_end_unix_ns=t['final_step_return_unix_ns']) for t in timed]
            from shoreline_campaign import clock_qualification
            clocks=clock_qualification({'cases':[{'trials':windows}]},provenance['monitor'])
            rows.append({'condition':key[0],'role':key[1],'gpu_uuid':run['gpu_uuid'],'source_identity':run['source_identity'],
                'run':capture['run'],'run_sha256':capture['run_sha256'],'provenance':provenance,'metrics':metrics,'clock':clocks})
            runs[key]=run
        except (OSError,ValueError,KeyError,AssertionError) as error:errors.append({'condition':key,'error':str(error)})
    if set(runs)==expected_keys:
        try:
            check_pair(runs[('native','producer')],runs[('native','consumer')])
            assert len({r['source_identity'] for r in runs.values()})==1
            assert len({r['reference_run_sha256'] for r in runs.values()})==1
            assert len({r['functional_CPU_review_sha256'] for r in runs.values()})==1
            producer=runs[('native','producer')]['gpu_uuid']
            assert all(runs[(name,'stock')]['gpu_uuid']==producer for name in ('stock_before','stock_after'))
            assert runs[('native','consumer')]['gpu_uuid']!=producer
        except (ValueError,KeyError,AssertionError) as error:errors.append({'scope':'paired completion/causality','error':str(error)})
    else:errors.append({'scope':'missing or rejected conditions','expected':sorted(expected_keys),'present':sorted(runs)})
    clocks_by_UUID={}
    for row in rows:
        if 'clocks' in row['clock']:
            for metric,value in row['clock']['clocks'].items():
                clocks_by_UUID.setdefault(row['gpu_uuid'],{}).setdefault(metric,[]).extend((value['minimum'],value['maximum']))
    pooled={uuid:{metric:max(v)/min(v) for metric,v in metrics.items()} for uuid,metrics in clocks_by_UUID.items()}
    clocks_ok=len(rows)==4 and all(r['clock']['passed'] for r in rows) and all(v<=1.05 for m in pooled.values() for v in m.values())
    controls={r['condition']:statistics.median(r['metrics']['submit_to_final_step_return_host_ms']) for r in rows if r['role']=='stock'}
    drift=max(controls.values())/min(controls.values()) if len(controls)==2 else None
    complete=set(runs)==expected_keys and not errors
    energy_status_counts={}
    for row in rows:
        for diagnostic in row['metrics']['raw_energy_per_warmup_and_timed_request']:
            name=diagnostic['status'];energy_status_counts[name]=energy_status_counts.get(name,0)+1
    result={'classification':'CPU review of producer host pilot; timed rows have no scheduler/connector hooks, but retain CPU timing/output bookkeeping',
        'status':'pilot_review_passed' if complete and clocks_ok and drift<=1.05 else 'pilot_partial_or_rejected',
        'capture_complete_and_correct':complete,'clock_gate_passed':clocks_ok,'pooled_clock_ratios_by_UUID':pooled,
        'same_GPU_stock_before_after_drift_ratio':drift,'rows':rows,'errors':errors,
        'performance_comparison_qualified':False,'qualified_producer_penalty_ratio':None,
        'reason':'Pilot3warm+3timed only; even passed pilot gates do not replace3independent acquisitions/10timed per condition',
        'serving_qualified':False,'primary_J_per_token':None,
        'raw_energy_status_counts':energy_status_counts,
        'next_required':'After actual pilot time/clock/source/drift review, budget a fresh complete3x10matched stock/native/stock cohort on the same physical producer UUID. Keep connector profiling separate.'}
    write(root/'results/producer_timing_CPU_review.json',result)
    lines=['# Producer host-timing pilot CPU review','',result['status']+'. '+result['reason'],
        '', '| Condition | Role | Raw median submit-to-final-return host ms | Clock gate |','|---|---|---:|---|']
    for row in rows:lines.append('| '+row['condition']+' | '+row['role']+' | '+str(statistics.median(row['metrics']['submit_to_final_step_return_host_ms']))+' | '+str(row['clock']['passed'])+' |')
    lines.extend(['','Same-GPU stock before/after drift ratio: '+str(drift)+'. Qualified producer penalty ratio: NA.',
        '', 'Raw per-role lifecycle energy-counter dispositions: '+str(energy_status_counts)+'. Primary J/token remains NA: aligned all-board windows, counter cadence and agreement with sampled power are not established.',
        '', 'Consumer first handoff query and following62host decode calls are separate in the JSON. File IPC is not HTTP/deployed TTFT or serving goodput; no physical bytes or primary energy are inferred.'])
    (root/'analysis/producer_timing_CPU_review.md').write_text('\n'.join(lines)+'\n')
    if result['status']!='pilot_review_passed':
        limitation(root,'producer_timing_CPU_review','Producer pilot incomplete or original source/clock/isolation/drift gates rejected',
            'results/producer_timing_CPU_review.json','Same-GPU before/native/after with all raw trials; exact outputs and source/provenance checks; original5percent clock/drift gates',
            'Inspect exact errors and all raw clock windows. Preserve rejected results; no favorable resampling or pooling of physical UUIDs. Bound any repair before a fresh complete cohort.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(review(parser.parse_args().root)['status'])
