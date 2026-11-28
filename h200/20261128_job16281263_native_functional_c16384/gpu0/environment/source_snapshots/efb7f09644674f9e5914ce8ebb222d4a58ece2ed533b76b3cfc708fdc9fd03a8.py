#!/usr/bin/env python3
"""Runtime admission only: verified producer pilot -> fixed3x10cohort.

Pilot latency/clock/UUID data never enter the new performance estimator.
The pilot may be on an earlier allocation, but the exact role source and
model/settings must match. Each new acquisition gets its own correctness,
source, physical-pair, clock, isolation and drift gates.
"""
import math
from pathlib import Path
from h200_priority_campaign import PACKAGE,load,sha
from h200_review_native_model import verified_process
from h200_review_producer_timing import check_trials,check_pair


def role_bound(elapsed_s,request_seconds):
    assert type(elapsed_s) in (int,float) and math.isfinite(elapsed_s) and elapsed_s>0
    assert len(request_seconds)==6 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in request_seconds)
    assert sum(request_seconds)<=elapsed_s+.001,'inconsistent pilot process/lifecycle clocks'
    timed=request_seconds[3:]
    oneoff=max(0,elapsed_s-sum(timed))
    bound=max(120,math.ceil(1.25*(oneoff+10*max(timed))+30))
    return {'timeout_s':bound,'observed_process_s':elapsed_s,'observed_initialization_warmups_and_other_s':oneoff,
        'observed_slowest_timed_request_s':max(timed),'predicted_warmups':3,'predicted_timed_requests':10,
        'formula':'ceil(1.25*(observed initialization/warmup/other time +10*slowest of3timed whole requests)+30s), floor120s',
        'within_existing_role_watchdog':bound<=660}


def budget(pilot):
    pilot=Path(pilot).resolve()
    outcome=load(pilot/'results/worker_outcome.json');review_path=pilot/'results/producer_timing_CPU_review.json'
    assert sha(review_path)==outcome['review_sha256']
    reviewed=load(review_path)
    assert reviewed['capture_complete_and_correct'] is True and reviewed['errors']==[]
    captures=load(pilot/'results/captures.json');assert captures['mode']=='pilot'
    expected={('stock_before','stock'),('native','producer'),('native','consumer'),('stock_after','stock')}
    rows={};native={};settings=[];contexts=[]
    for capture in captures['captures']:
        key=(capture['condition'],capture['role']);assert key in expected and key not in rows
        base=Path(capture['base']);session=load(base/'session.json')
        provenance=verified_process(base,'matched_producer_timing','h200_producer_timing_role.py',session)
        assert provenance['entry_sha256']==sha(PACKAGE/'scripts/h200_producer_timing_role.py')
        path=Path(capture['run']);proc_path=Path(capture['process'])
        assert sha(path)==capture['run_sha256'] and sha(proc_path)==capture['process_sha256']
        run=load(path);proc=load(proc_path)
        assert run['source_identity']==session['frozen_source_identity'] and run['gpu_uuid']==session['gpu_uuid']
        pair_path=base.parent/'pair.json' if key[0]=='native' else base/'pair.json'
        assert sha(pair_path)==run['pair_sha256'];pair=load(pair_path)
        assert pair['mode']=='pilot' and pair['context'] in (16384,32704)
        source=Path(pair['reference_run']);assert sha(source)==pair['reference_run_sha256']
        reference=load(source);check_trials(run,pair,reference['cases'][0]['trials'][0]['output_ids'][0])
        settings.append(run['engine_settings_without_KV_object']);contexts.append(pair['context'])
        times=[(t['completed_unix_ns']-t['raw_energy_counter']['before']['query_start_unix_ns'])/1e9 for t in run['trials']]
        rows[key]=dict(role_bound(proc['elapsed_s'],times),run=str(path),run_sha256=sha(path),
            process=str(proc_path),process_sha256=sha(proc_path),pilot_gpu_uuid=run['gpu_uuid'],
            source_identity=run['source_identity'],entry_sha256=provenance['entry_sha256'])
        if key[0]=='native':native[key[1]]=run
    assert set(rows)==expected and len(set(contexts))==1 and all(s==settings[0] for s in settings)
    check_pair(native['producer'],native['consumer'])
    phases={'stock_before':rows[('stock_before','stock')]['timeout_s'],
        'native':max(rows[('native','producer')]['timeout_s'],rows[('native','consumer')]['timeout_s']),
        'stock_after':rows[('stock_after','stock')]['timeout_s']}
    required=3*sum(phases.values())+360
    return {'classification':'runtime budget from preserved pilot, not a performance prediction or qualification',
        'pilot':str(pilot),'pilot_CPU_review':str(review_path),'pilot_CPU_review_sha256':sha(review_path),
        'role_source_sha256':sha(PACKAGE/'scripts/h200_producer_timing_role.py'),'context':contexts[0],
        'engine_settings':settings[0],'phase_timeout_s':phases,'required_s':required,
        'acquisitions':3,'warmups_per_condition':3,'timed_per_condition':10,
        'all_bounds_within_existing660second_watchdog':all(row['within_existing_role_watchdog'] for row in rows.values()),
        'roles':[dict(condition=c,role=r,**row) for (c,r),row in rows.items()],
        'margin_scope':'Each role bound includes25percent allowance plus30s; another360s covers cohort setup, per-acquisition publication and archive/closeout. Actual allocation/source/idle checks still gate every launch.',
        'pilot_performance_data_used_in_comparison':False}
