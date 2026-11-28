#!/usr/bin/env python3
"""CPU review of actual serving-policy and resident-prefill functional evidence."""
import argparse
from pathlib import Path
from h200_priority_campaign import load,write,sha,valid_tokens,strict_reference_record


def prefix_coverage(prefill,context):
    assert prefill['input_prompt_tokens']==context and prefill['first_selected_tokens']==1
    rows=prefill['scheduler_rows'];assert len(rows)==prefill['actual_prefill_steps']
    offset=0
    for step in rows:
        assert step['composition']=='prefill_only' and not step['any_preempted_request']
        assert step['post_prompt_input_queries']==0 and len(step['requests'])==1
        request=step['requests'][0]
        assert request['computed_tokens_before_step']==offset and request['original_prompt_tokens']==context
        assert 0<request['prompt_input_queries']<=prefill['token_budget']
        offset+=request['prompt_input_queries']
    assert offset==context
    if prefill['whole']:assert len(rows)==1
    return True


def review(root):
    session=load(root/'session.json');capture=load(root/'results/functional_pilots.json');results=[]
    assert len(capture['results'])==2
    for row in capture['results']:
        assert row['status']=='functional_passed' and row['source_identity']==session['frozen_source_identity']
        native=load(row['run']);assert sha(row['run'])==row['run_sha256'] and valid_tokens(native)
        expected=native['cases'][0]['trials'][0]['output_ids'][0]
        assert sha(row['receipt'])==row['receipt_sha256']
        hf=root/'results'/(row['id']+'_hf.json');reference=load(hf)
        assert strict_reference_record(reference,row['run_sha256'])
        assert Path(reference['source_run']).resolve()==Path(row['run']).resolve()
        provenance=[]
        for label in (row['id'],row['id']+'_hf'):
            path=root/'results/processes'/(label+'.json');process=load(path)
            assert process['status']=='complete' and process['source_freeze_identity']==session['frozen_source_identity']
            assert process['monitor'] and not process['monitor'].get('failure')
            monitor=Path(process['monitor']['source']);assert monitor.is_file()
            provenance.append({'process':str(path),'process_sha256':sha(path),'monitor':str(monitor),'monitor_sha256':sha(monitor)})
        item={'id':row['id'],'status':'functional_review_passed','strict_dynamic_and_StaticCache_passed':True,
              'run_sha256':row['run_sha256'],'receipt_sha256':row['receipt_sha256'],
              'HF_reference':str(hf),'HF_reference_sha256':sha(hf),'provenance':provenance,
              'qualified_quantitative_comparison':False}
        if row['id'].startswith('P3'):
            policies=[];output=Path(row['run']).parent
            for name in ('whole_separate','chunked_separate','chunked_mixed'):
                path=output/(name+'.json');ledger=load(path)
                assert ledger['status']=='functional_policy_and_exact_outputs_passed'
                assert len(ledger['requests'])==3
                assert all(r['output_ids']==expected and len(r['inter_token_s'])==63 for r in ledger['requests'].values())
                compositions={r['composition'] for r in ledger['scheduler_rows']}
                assert ('mixed' in compositions)==(name=='chunked_mixed')
                assert not any(r['any_preempted_request'] for r in ledger['scheduler_rows'])
                assert sum(r['prompt_input_queries'] for r in ledger['scheduler_rows'])==3*row['context']
                assert sum(r['post_prompt_input_queries'] for r in ledger['scheduler_rows'])==3*63
                policies.append({'policy':name,'requests':3,'steps':len(ledger['steps']),
                    'compositions':sorted(compositions),'source':str(path),'sha256':sha(path)})
            item['policies']=policies
        else:
            manifest=load(row['receipt']);assert manifest['status']=='complete'
            conditions=[]
            for name,condition in manifest['conditions'].items():
                trials=condition['warmups']+condition['trials'];assert len(trials)==6
                for trial in trials:
                    assert trial['output_ids']==[expected] and trial['decode_iterations']==63
                    prefix_coverage(trial['prefill_and_first_token'],row['context'])
                conditions.append({'condition':name,'warmups':3,'timed_trials':3,
                    'prefill_steps':sorted({t['prefill_and_first_token']['actual_prefill_steps'] for t in trials})})
            assert {c['condition'] for c in conditions}=={'whole','chunk256'}
            item['conditions']=conditions
        results.append(item)
    result={'classification':'CPU review of measured instrumented functional pilots',
            'new_GPU_measurements':False,'gpu_uuid':session['gpu_uuid'],
            'source_identity':session['frozen_source_identity'],'results':results,
            'scope':'Functional correctness and actual scheduler/prefix composition only. No3independent acquisitions, loaded serving, held-out tuning/SLO/energy, physical traffic or ordinary TTFT qualification.'}
    write(root/'results/functional_pilots_CPU_review.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();result=review(a.root);print({r['id']:r['status'] for r in result['results']})
