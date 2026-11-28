#!/usr/bin/env python3
"""Read-only-by-default CPU review of fixed3x10producer acquisitions.

The primary denominator is the same-acquisition stock-before control;
stock-after is an independent drift guard. No pilot samples enter the CI.
"""
import argparse
import json
from pathlib import Path
import statistics
from h200_priority_campaign import load,sha,valid_tokens,strict_reference_record
from h200_review_native_model import verified_process
from h200_review_producer_timing import check_trials,check_pair
from h200_producer_timing_role import descriptor_scope

KEYS={('stock_before','stock'),('native','producer'),('native','consumer'),('stock_after','stock')}


def certificate_dependencies(path,pair,session):
    cert=load(path);assert cert['status']=='native_model_functional_review_passed' and cert['job_id']==session['slurm_job_id']
    original=load(path.parents[1]/'pair.json')
    assert original['context']==pair['context'] and original['reference_run_sha256']==pair['reference_run_sha256']
    assert len(cert['roles'])==2 and len(cert['reference_process_provenance'])==2
    dependencies=None
    for role in cert['roles']:
        for field in ('session','run','process','monitor'):assert sha(Path(role[field]))==role[field+'_sha256']
        owned=load(Path(role['session']));native=load(Path(role['run']));proc=load(Path(role['process']))
        assert owned['slurm_job_id']==session['slurm_job_id'] and owned['frozen_source_identity']==session['frozen_source_identity']
        assert role['gpu_uuid']==pair['roles'][role['role']]['gpu_uuid']
        digest=cert['implementation_sha256']
        assert proc['script_sha256']['h200_review_native_model.py']==digest
        snapshot=Path(role['session']).parent/'environment/source_snapshots'/(digest+'.py')
        assert sha(snapshot)==digest # portable after the allocation's /tmp tree disappears
        if dependencies is None:dependencies=native['optional_dependencies']
        assert native['optional_dependencies']==dependencies
    for provenance in cert['reference_process_provenance']:
        for field in ('process','monitor','log','idle_check'):assert sha(Path(provenance[field]))==provenance[field+'_sha256']
        assert sha(Path(provenance['entry_snapshot']))==provenance['entry_sha256']
    return dependencies


def read_capture(capture):
    base=Path(capture['base']);session=load(base/'session.json')
    assert sha(Path(capture['run']))==capture['run_sha256'] and sha(Path(capture['process']))==capture['process_sha256']
    provenance=verified_process(base,'matched_producer_timing','h200_producer_timing_role.py',session)
    run=load(Path(capture['run']))
    descriptor=base.parent/'pair.json' if capture['condition']=='native' else base/'pair.json'
    assert sha(descriptor)==run['pair_sha256'];pair=load(descriptor)
    descriptor_scope(pair,session,capture['role'],session['gpu_uuid'],session['slurm_job_id'],session['expected_hostname'])
    assert pair['mode']=='comparison' and run['role']==capture['role'] and pair['acquisition']==capture['acquisition']
    assert run['source_identity']==session['frozen_source_identity'] and run['gpu_uuid']==session['gpu_uuid']==run['hardware']['uuid']
    source=Path(pair['reference_run']);assert sha(source)==pair['reference_run_sha256']==run['reference_run_sha256']
    reference=load(source);assert valid_tokens(reference)
    assert reference['config']['model_id']=='Qwen/Qwen2.5-32B-Instruct' and reference['config']['seed']==20261013
    assert reference['config']['model_revision']=='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
    assert not reference['config'].get('decode_conditioning')
    assert (reference['cases'][0]['batch_size'],reference['cases'][0]['context_length'],reference['cases'][0]['generated_tokens'])==(1,pair['context'],64)
    for metric in ('sm_count','l2_bytes','power_limit_w'):
        assert isinstance(run['hardware'][metric],(int,float)) and run['hardware'][metric]>0
        assert run['hardware'][metric]==reference['hardware'][metric]
    for trial in run['trials']:
        for endpoint in ('hardware_before','hardware_after'):
            assert trial[endpoint]['uuid']==run['gpu_uuid'] and trial[endpoint]['power_limit_w']==run['hardware']['power_limit_w']
    hf=Path(pair['strict_HF_reference']);assert sha(hf)==pair['strict_HF_reference_sha256']
    assert strict_reference_record(load(hf),sha(source))
    assert run['engine_settings_without_KV_object']==reference['engine_settings'] and run['versions']==reference['versions']
    cert=Path(pair['functional_CPU_review']);assert sha(cert)==pair['functional_CPU_review_sha256']==run['functional_CPU_review_sha256']
    assert run['optional_dependencies']==certificate_dependencies(cert,pair,session)
    metrics=check_trials(run,pair,reference['cases'][0]['trials'][0]['output_ids'][0])
    windows=[dict(t,wall_start_unix_ns=t['submit_unix_ns'],wall_end_unix_ns=t['final_step_return_unix_ns']) for t in run['trials'] if not t['warmup']]
    from shoreline_campaign import clock_qualification
    return {'acquisition':pair['acquisition'],'condition':capture['condition'],'role':capture['role'],'context':pair['context'],
        'gpu_uuid':run['gpu_uuid'],'source_identity':run['source_identity'],'reference_run_sha256':sha(source),
        'functional_CPU_review_sha256':sha(cert),'source_verified':True,'metrics':metrics,
        'clock':clock_qualification({'cases':[{'trials':windows}]},provenance['monitor']),
        'run':capture['run'],'run_sha256':capture['run_sha256'],'provenance':provenance},run


def analyze(rows,errors,draws=10000):
    expected={(a,c,r) for a in range(3) for c,r in KEYS}
    keys=[(r['acquisition'],r['condition'],r['role']) for r in rows]
    complete=set(keys)==expected and len(keys)==len(expected) and not errors
    sources_ok=bool(rows) and len({r['source_identity'] for r in rows})==1 and len({r['reference_run_sha256'] for r in rows})==1 and len({r['functional_CPU_review_sha256'] for r in rows})==1 and len({r['context'] for r in rows})==1 and all(r['source_verified'] for r in rows)
    producer_UUIDs={r['gpu_uuid'] for r in rows if r['role']!='consumer'}
    consumer_UUIDs={r['gpu_uuid'] for r in rows if r['role']=='consumer'}
    pair_ok=len(producer_UUIDs)==len(consumer_UUIDs)==1 and producer_UUIDs.isdisjoint(consumer_UUIDs)
    grouped={};clocks_ok=bool(rows) and all(r['clock']['passed'] for r in rows)
    for row in rows:
        for metric,value in row['clock'].get('clocks',{}).items():grouped.setdefault(row['gpu_uuid'],{}).setdefault(metric,[]).extend((value['minimum'],value['maximum']))
    pooled={u:{m:max(v)/min(v) for m,v in metrics.items()} for u,metrics in grouped.items()}
    clocks_ok=clocks_ok and len(pooled)==2 and all(set(m)=={'sm_clock_mhz','memory_clock_mhz'} and all(v<=1.05 for v in m.values()) for m in pooled.values())
    lookup={(r['acquisition'],r['condition'],r['role']):r for r in rows}
    drift={};all_controls=[]
    for a in range(3):
        control=[lookup.get((a,c,'stock')) for c in ('stock_before','stock_after')]
        if all(control):
            medians=[statistics.median(r['metrics']['submit_to_final_step_return_host_ms']) for r in control]
            drift[str(a)]=max(medians)/min(medians);all_controls+=medians
    pooled_drift=max(all_controls)/min(all_controls) if len(all_controls)==6 else None
    counts_ok=all(len(r['metrics']['submit_to_final_step_return_host_ms'])==10 for r in rows)
    qualified=complete and sources_ok and pair_ok and clocks_ok and counts_ok and len(drift)==3 and all(v<=1.05 for v in drift.values()) and pooled_drift<=1.05
    result={'classification':'Fixed3independent-acquisition/10timed stock/native/stock producer host comparison',
        'complete':complete,'source_gate':sources_ok,'same_physical_pair_gate':pair_ok,'clock_gate':clocks_ok,
        'pooled_clock_ratios_by_UUID':pooled,'stock_before_after_drift_by_acquisition':drift,'pooled_stock_drift':pooled_drift,
        'performance_qualified':qualified,'qualified_producer_ratio':None,'qualified_stock_host_ms':None,'qualified_native_host_ms':None,
        'rows':rows,'errors':errors,'pilot_samples_in_estimator':0,'primary_J_per_token':None,'serving_qualified':False,
        'primary_anchor':'Same-acquisition stock-before; stock-after is a <=5percent within/across-acquisition drift guard',
        'scope':'Producer first step return host latency only. Actual send-readiness observation, consumer first handoff query and following62host calls remain separate raw fields. File IPC/CPU bookkeeping is not HTTP/deployed TTFT, serving goodput, GPU-event elapsed or all-board energy.'}
    if qualified:
        from shoreline_statistics import paired_ratio,hierarchical_interval
        native=[lookup[(a,'native','producer')]['metrics']['submit_to_final_step_return_host_ms'] for a in range(3)]
        stock=[lookup[(a,'stock_before','stock')]['metrics']['submit_to_final_step_return_host_ms'] for a in range(3)]
        result.update(qualified_producer_ratio=paired_ratio(native,stock,draws=draws),
            qualified_stock_host_ms=hierarchical_interval(stock,draws=draws),qualified_native_host_ms=hierarchical_interval(native,draws=draws))
    return result


def review(root,draws=10000):
    root=Path(root).resolve();manifest_path=root/'results/acquisitions.json';manifest=load(manifest_path)
    assert manifest['mode']=='comparison'
    rows=[];errors=[];runs={}
    for capture in manifest['captures']:
        try:
            key=(capture['acquisition'],capture['condition'],capture['role']);assert key not in runs
            row,run=read_capture(capture);rows.append(row);runs[key]=run
        except (OSError,ValueError,KeyError,AssertionError,TypeError,IndexError) as error:errors.append({'capture':capture,'error':str(error)})
    for acquisition in range(3):
        try:check_pair(runs[(acquisition,'native','producer')],runs[(acquisition,'native','consumer')])
        except (KeyError,ValueError,AssertionError) as error:errors.append({'scope':'paired completion/causality acquisition'+str(acquisition),'error':str(error)})
    result=analyze(rows,errors,draws)
    result.update(implementation_sha256=sha(Path(__file__)),capture_manifest=str(manifest_path),capture_manifest_sha256=sha(manifest_path),
        statistics_source_sha256=sha(Path(__file__).with_name('shoreline_statistics.py')))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Optional new output path; existing certificates are never overwritten')
    args=parser.parse_args();result=review(args.root)
    if args.output:
        with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('complete','performance_qualified','qualified_producer_ratio')}))
