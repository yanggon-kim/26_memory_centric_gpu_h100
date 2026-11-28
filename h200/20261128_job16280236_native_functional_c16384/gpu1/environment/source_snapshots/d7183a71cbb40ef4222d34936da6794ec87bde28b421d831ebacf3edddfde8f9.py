#!/usr/bin/env python3
"""Review resident-prefill acquisitions without turning capture into qualification."""
import argparse
import copy
import math
from pathlib import Path
import statistics
from h200_priority_campaign import load,write,sha,valid_tokens,strict_reference_record,limitation
from h200_review_functional_pilots import prefix_coverage

CONDITIONS=('whole','chunk256','chunk1024','chunk4096','whole_end')


def pooled_gate(rows,clock_key):
    if len(rows)!=15 or len({r['gpu_uuid'] for r in rows})!=1 or len({r['source_identity'] for r in rows})!=1:return False,{}
    if any(sorted(r['acquisition'] for r in rows if r['condition']==name)!=[0,1,2] for name in CONDITIONS):return False,{}
    if not all(r['source_verified'] and r[clock_key].get('passed') and
        all(len(r[key])==10 and all(math.isfinite(x) and x>0 for x in r[key])
            for key in ('service_wall_ms','service_gpu_event_ms','submitted_first_token_wall_ms')) for r in rows):return False,{}
    clocks=[r[clock_key]['clocks'] for r in rows]
    ratios={metric:max(c[metric]['maximum'] for c in clocks)/min(c[metric]['minimum'] for c in clocks)
            for metric in ('sm_clock_mhz','memory_clock_mhz')}
    return all(value<=1.05 for value in ratios.values()),ratios


def review(root):
    root=Path(root).resolve()
    session=load(root/'session.json');rows=[];errors=[];expected=None;prompt=None
    smoke_path=root/'results/prefill_smoke.json'
    try:
        smoke=load(smoke_path)
        assert smoke['status']=='passed' and sha(smoke['source'])==smoke['source_sha256']
        source=load(smoke['source']);assert valid_tokens(source)
        expected=source['cases'][0]['trials'][0]['output_ids'];prompt=source['cases'][0]['prompt_ids']
        label=Path(smoke['source']).parent.name;hf=root/'results'/(label+'_hf.json')
        reference=load(hf);assert strict_reference_record(reference,smoke['source_sha256'])
        assert reference['gpu_uuid']==session['gpu_uuid']
        for name in (label,label+'_hf'):
            process=load(root/'results/processes'/(name+'.json'))
            assert process['status']=='complete' and process['source_freeze_identity']==session['frozen_source_identity']
            assert process['monitor'] and not process['monitor'].get('failure') and Path(process['monitor']['source']).is_file()
    except (OSError,ValueError,KeyError,AssertionError) as error:
        expected=None;errors.append({'scope':'smoke','error':str(error)})
    progress=root/'results/prefill_acquisitions.json'
    captures=load(progress)['records'] if progress.exists() else []
    for capture in captures:
        try:
            assert expected is not None and capture['status']=='capture_passed_pending_CPU_review'
            assert capture['gpu_uuid']==session['gpu_uuid'] and capture['source_identity']==session['frozen_source_identity']
            assert sha(capture['run'])==capture['run_sha256'] and sha(capture['manifest'])==capture['manifest_sha256']
            process=load(capture['process']);assert process['status']=='complete'
            assert process['source_freeze_identity']==session['frozen_source_identity']
            assert process['monitor'] and not process['monitor'].get('failure')
            monitor=Path(process['monitor']['source']);assert monitor.is_file()
            manifest=load(capture['manifest']);assert manifest['status']=='complete' and manifest['protocol']=='resident_prefill_v2'
            assert set(manifest['conditions'])==set(CONDITIONS)
            output=Path(capture['run']).parent
            acquisition_rows=[]
            for name in CONDITIONS:
                condition=manifest['conditions'][name];native_path=output/name/'run.json';native=load(native_path)
                assert valid_tokens(native,expected) and native['cases'][0]['prompt_ids']==prompt
                assert all(native['config'][key]==source['config'][key]
                    for key in ('model_id','model_revision','precision','generated_tokens','seed'))
                assert native.get('baseline_comparison_eligible') is False and not native['config'].get('decode_conditioning')
                assert native['cases'][0]['trials']==condition['trials']
                assert len(condition['warmups'])==3 and len(condition['trials'])==10
                for trial in condition['warmups']+condition['trials']:
                    assert trial['output_ids']==expected and trial['decode_iterations']==63
                    p=trial['prefill_and_first_token'];prefix_coverage(p,capture['context'])
                    assert p['wall_ms']>0 and p['gpu_event_ms']>0 and p['submit_to_first_token_wall_ms']>0
                    assert p['submit_start_unix_ns']<=p['submit_end_unix_ns']<=p['first_token_observed_unix_ns']
                service=[t['prefill_and_first_token'] for t in condition['trials']]
                submitted=[]
                for item in service:
                    window=copy.deepcopy(item)
                    window.update(hardware_before=item['submission_hardware_before'],
                        wall_start_unix_ns=item['submit_start_unix_ns'],wall_end_unix_ns=item['first_token_observed_unix_ns'])
                    submitted.append(window)
                from shoreline_campaign import clock_qualification
                row={'acquisition':capture['acquisition'],'condition':name,'source_verified':True,
                    'gpu_uuid':session['gpu_uuid'],'source_identity':session['frozen_source_identity'],
                    'run':str(native_path),'run_sha256':sha(native_path),'process':capture['process'],
                    'process_sha256':sha(capture['process']),'monitor':str(monitor),'monitor_sha256':sha(monitor),
                    'service_wall_ms':[p['wall_ms'] for p in service],
                    'service_gpu_event_ms':[p['gpu_event_ms'] for p in service],
                    'submitted_first_token_wall_ms':[p['submit_to_first_token_wall_ms'] for p in service],
                    'prefill_steps':[p['actual_prefill_steps'] for p in service],
                    'service_clock':clock_qualification({'cases':[{'trials':service}]},monitor),
                    'submit_clock':clock_qualification({'cases':[{'trials':submitted}]},monitor)}
                acquisition_rows.append(row)
            rows.extend(acquisition_rows)
        except (OSError,ValueError,KeyError,AssertionError) as error:
            errors.append({'scope':capture.get('acquisition'),'error':str(error),'capture':capture})
    service_ok,service_ratios=pooled_gate(rows,'service_clock');submit_ok,submit_ratios=pooled_gate(rows,'submit_clock')
    complete=len(rows)==15 and not errors
    result={'classification':'CPU review of measured instrumented resident-prefill acquisitions',
        'session':str(root/'session.json'),'session_sha256':sha(root/'session.json'),
        'smoke_receipt':str(smoke_path),'smoke_receipt_sha256':sha(smoke_path) if smoke_path.exists() else None,
        'workload':{'model_id':source['config']['model_id'],'model_revision':source['config']['model_revision'],
            'precision':source['config']['precision'],'seed':source['config']['seed'],
            'batch':len(prompt),'context':len(prompt[0]),'generated_tokens':source['config']['generated_tokens']}
            if expected is not None else None,
        'complete_acquisition_set':complete,'instrumented_service_qualified':service_ok and complete,
        'instrumented_submitted_first_token_qualified':submit_ok and complete,
        'ordinary_TTFT_qualified':False,'physical_traffic_measured':False,
        'service_pooled_clock_ratios':service_ratios,'submitted_pooled_clock_ratios':submit_ratios,
        'rows':rows,'errors':errors,'conditions':[],
        'implementation_sha256':sha(Path(__file__)),
        'scope':'Instrumented pretokenized engine submission and prefill service only. Policy/observer/event costs remain included; ordinary TTFT, offered-load SLOs, physical bytes and energy remain separate gates. All rejected observations are retained.'}
    for name in CONDITIONS:
        selected=sorted([r for r in rows if r['condition']==name],key=lambda r:r['acquisition'])
        point={'condition':name,'acquisitions':[r['acquisition'] for r in selected],
            'raw_median_service_wall_ms':[statistics.median(r['service_wall_ms']) for r in selected],
            'raw_median_submitted_first_token_ms':[statistics.median(r['submitted_first_token_wall_ms']) for r in selected]}
        from shoreline_statistics import hierarchical_interval
        if service_ok and complete:
            point['instrumented_service_wall_ms']=hierarchical_interval([r['service_wall_ms'] for r in selected])
            point['instrumented_service_gpu_event_ms']=hierarchical_interval([r['service_gpu_event_ms'] for r in selected])
        if submit_ok and complete:point['instrumented_submitted_first_token_ms']=hierarchical_interval([r['submitted_first_token_wall_ms'] for r in selected])
        result['conditions'].append(point)
    write(root/'results/prefill_CPU_review.json',result)
    lines=['# Resident prefill comparison: CPU review','',result['scope'],'',
        f'Complete acquisition set: **{complete}**. Instrumented service qualified: **{result["instrumented_service_qualified"]}**. Instrumented submitted first token qualified: **{result["instrumented_submitted_first_token_qualified"]}**. Ordinary TTFT remains unqualified.','',
        '| Condition | Acquisitions | Raw median service wall ms | Raw median submitted first token ms |',
        '|---|---|---|---|']
    for p in result['conditions']:lines.append(f"| {p['condition']} | {p['acquisitions']} | {p['raw_median_service_wall_ms']} | {p['raw_median_submitted_first_token_ms']} |")
    lines+=['','All source/monitor hashes, phase-specific clock windows, partial/error records and any qualified intervals are in `../results/prefill_CPU_review.json`.']
    (root/'analysis/prefill_CPU_review.md').write_text('\n'.join(lines)+'\n')
    if not complete or not service_ok or not submit_ok:
        limitation(root,'prefill_CPU_review','Prefill capture/source/phase-clock qualification is incomplete or rejected',
            'results/prefill_CPU_review.json','Own smoke/HF, exact prefix and63decode checks, phase-specific clock sampling and original3x10/source/isolation gates',
            'Inspect exact errors and raw windows; do not pool UUIDs or resample for favorable clocks. Preserve instrumented scope and resume a fresh complete acquisition set when incomplete')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();r=review(a.root)
    print({k:r[k] for k in ('complete_acquisition_set','instrumented_service_qualified','instrumented_submitted_first_token_qualified')})
