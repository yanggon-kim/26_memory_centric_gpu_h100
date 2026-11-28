"""Versioned orchestration for the independently labeled resident eager curve."""
import datetime as dt
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from h200_priority_campaign import PACKAGE,load,write,sha,config,publish,limitation,valid_tokens,analyze_curve
from h200_resident_sweep import conditions
SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def verify_published_core(path):
    """Reuse a completed broad-curve gate, never its latency across devices."""
    from h200_review_allocation import case_records
    from h200_priority_campaign import strict_reference_record
    review=load(path);root=Path(review['source_session']).parent
    assert review['complete_acquisition_set'] and (review['batch'],review['context'])==(1,4096)
    assert sha(review['source_session'])==review['source_session_sha256']
    assert sha(review['strict_HF_reference'])==review['strict_HF_reference_sha256']
    reference=load(review['strict_HF_reference'])
    assert strict_reference_record(reference,sha(reference['source_run']))
    assert len(review['cases'])==27
    for case in review['cases']:assert sha(case['run'])==case['run_sha256']
    rows,issues,interrupted=case_records(root);assert not issues and not interrupted
    rows=[r for r in rows if (r['batch'],r['context'])==(1,4096)]
    clean=[r for r in rows if not r['pilot']]
    assert len(clean)==27 and all(r['status']=='complete' and len(r.get('step_ms',[]))==10 for r in clean)
    for acquisition in range(3):
        selected=[r for r in clean if r['acquisition']==acquisition]
        assert len(selected)==9 and sum(r['stock'] for r in selected)==1
        assert {r['actual_sm_count'] for r in selected}=={132,112,96,80,56,40}
        assert {r['suffix'] for r in selected}=={'','_interleaved','_end'}
    proofs=[r for r in rows if r['pilot'] and '_proof_' in r['id']]
    assert len(proofs)==7 and all(r['qualified'] and r['placement_passed'] for r in proofs)
    return {'status':'completed_broad_capture_verified','review':str(path),'review_sha256':sha(path),
        'protocol':review['protocol'],'original_qualified_timing':review['qualified_timing_curve'],
        'scope':'Order/eligibility gate only: broad eager fallback curve was published. No old latency, clock qualification, source/UUID measurement or optimized restriction success transfers to this allocation.'}


def build_environment(worker):
    assert hashlib.sha256(SOURCE_AT_IMPORT).hexdigest()==sha(worker.script/Path(__file__).name),'loaded resident orchestration differs from frozen source'
    import h200_resident_sweep
    assert hashlib.sha256(h200_resident_sweep.SOURCE_AT_IMPORT).hexdigest()==sha(worker.script/'h200_resident_sweep.py')
    wrappers={}
    for variable,name in (('CC','h200_host_cc.sh'),('CXX','h200_host_cxx.sh')):
        path=worker.script/name
        path.chmod(0o700)
        os.environ[variable]=str(path)
        wrappers[variable]={'path':str(path),'sha256':sha(path)}
    write(worker.root/'environment/host_build_wrappers.json',{'wrappers':wrappers,
        'actual_loaded_resident_driver_sha256':hashlib.sha256(SOURCE_AT_IMPORT).hexdigest(),
        'scope':'Same pinned GCC12.3 frontend, native /usr/bin linker via -B/usr/bin; fresh source cohort only; no system changes',
        'prior_repair_evidence':'h200/20261127_job16210814_priority/gpu1_native_linker/results/P0_native_L2_counter_review.json'})


def acquire(worker,acquisition,proof=False,pilot=False,batch=1,context=4096,conditioned=False):
    label=f'resident_b{batch}_c{context}_a{acquisition}'+('_pilot' if pilot else '_proof' if proof else '')
    cfg=config(worker.root,'eager',context,3 if proof else 10,batch)
    conditioning_declaration=None
    if conditioned:
        data=load(cfg)
        conditioning_declaration={'protocol':'memory_settle_v3','seconds':1.0,
            'buffer_bytes':256*1024**2,'scope':'conditioned_decode_only'}
        data['decode_conditioning']=conditioning_declaration
        cfg=cfg.with_name(cfg.stem+'_memory_settle_v3.json');write(cfg,data)
    output=worker.root/'runs'/label
    command=[worker.py,worker.script/'h200_resident_sweep.py','--config',cfg,'--gpu-uuid',worker.uuid,
             '--output',output,'--acquisition',str(acquisition)]
    if pilot:command+=['--pilot']
    if proof:
        command+=['--external-profiler']
        command=[worker.cuda/'bin/nsys','profile','--sample=none','--cpuctxsw=none',
                 '--trace=cuda,nvtx','--cuda-graph-trace=node','--capture-range=cudaProfilerApi',
                 '--capture-range-end=repeat:'+str(len(conditions(pilot,True))),
                 '--output='+str(worker.root/'profiles'/label)]+command
    proc=worker.execute(label,command,timeout=1500)
    manifest=output/'resident_sweep.json'
    rows=[];data=load(manifest) if manifest.exists() else {}
    databases=[]
    if proof and proc['status']=='complete':
        for index,report in enumerate(sorted((worker.root/'profiles').glob(label+'*.nsys-rep'))):
            db=report.with_suffix('.sqlite')
            result=worker.execute(label+f'_export{index}',[worker.cuda/'bin/nsys','export','--type=sqlite',
                '--output='+str(db),report],gpu=False,timeout=180)
            if result['status']=='complete':databases.append(db)
    for name,fraction,green_path in conditions(pilot,proof):
        run=output/name/'run.json'
        row={'id':label+'_'+name,'producer_label':label,'engine':'eager','batch':batch,'context':context,
             'fraction':fraction,'acquisition':acquisition,'pilot':proof,'stock':not green_path,
             'suffix':'_interleaved' if name.endswith('_interleaved') else '_end' if name.endswith('_end') else '',
             'repair_route':'resident_eager_memory_settle_v3' if conditioned else 'resident_eager_v1','gpu_uuid':worker.uuid,'source_identity':worker.settings['frozen_source_identity'],
             'run':str(run),'status':'failed','process_status':proc['status'],'qualified':False,
             'producer_elapsed_s':proc.get('elapsed_s'),'protocol_scope':'one full resident model per acquisition; sequential conditions, never pooled with fresh-model-per-point',
             'conditioning':conditioning_declaration}
        record=data.get('conditions',{}).get(name,{})
        row['allocation']=record.get('allocation',{})
        row['actual_sm_count']=row['allocation'].get('actual_sm_count')
        row['physical_sm_count']=record.get('physical_sm_count')
        from h200_priority_campaign import sm_grant_matches_case
        row['sm_grant_matches_requested_scope']=sm_grant_matches_case(not green_path,fraction,row['actual_sm_count'],row['physical_sm_count'])
        if run.exists():
            native=load(run);row['run_sha256']=sha(run)
            row['exact_outputs']=bool(valid_tokens(native,worker.expected))
            from h200_clock_conditioning_pilot import conditioning_gate
            row['conditioning_scope_verified']=(conditioning_gate(native,conditioned,3 if proof else 10)
                and conditioning_gate({'config':native['config'],'cases':[{'trials':record.get('warmups',[])}]},conditioned,3))
            trials=native['cases'][0]['trials']
            if proc['status']=='complete' and data.get('status')=='complete' and row['exact_outputs'] and row['conditioning_scope_verified']:
                row['status']='complete'
                row['step_ms']=[t['wall_ms']/t['decode_iterations'] for t in trials]
                row['gpu_event_step_ms']=[t['cuda_event_ms']/t['decode_iterations'] for t in trials]
                row['tokens_s']=[t['tokens_per_s'] for t in trials]
                if not proof:
                    from shoreline_campaign import clock_qualification
                    row['clock']=clock_qualification(native,proc['monitor']['source'])
                    row['qualified']=row['clock']['passed'] and len(trials)==10 and row['sm_grant_matches_requested_scope']
        if proof:
            try:
                from shoreline_timeline import read
                windows=[]
                for db in databases:
                    with sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True) as connection:
                        found=connection.execute('SELECT count(*) FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL',
                                                 (f'resident_{name}_ticks_4_6',)).fetchone()[0]
                    if found:
                        windows.extend(read(db,row['allocation']['green_context_id'] or 0,
                                            f'resident_{name}_ticks_4_6')['windows'])
                row['timeline']={'status':'complete' if windows else 'missing','windows':windows,
                    'restriction_passed':len(windows)==1 and all(w['restriction_passed'] and w['graph_kernel_count']==0 for w in windows)}
                row['placement_passed']=row['timeline']['restriction_passed']
                row['qualified']=row['status']=='complete' and row['placement_passed'] and row['sm_grant_matches_requested_scope']
            except (OSError,ValueError,KeyError,AssertionError,sqlite3.Error) as error:row['proof_error']=str(error)
        write(worker.root/'results'/(row['id']+'.json'),row);rows.append(row)
    good=all(r['qualified'] if proof else r['status']=='complete' and r['sm_grant_matches_requested_scope'] for r in rows)
    if not good or (not proof and any(not r['qualified'] for r in rows)):
        limitation(worker.root,label,'Resident producer/output/placement/conditioning-receipt gate failed' if not good else 'Resident curve clocks fail the predeclared envelope; all trials retained',
            str(manifest),'New separate protocol; bounded actual-kernel proof, independent HF exact reference, original3x10 clock/source/isolation gates',
            'Inspect the saved failure; do not pool protocols or resample for favorable clocks. Continue eligible full-GPU cases when the captured curve is quality-limited.')
    return rows,proc,good


def analyze(worker,conditioned,batch=1,context=4096):
    result=analyze_curve(worker.root,worker.records,'eager',batch=batch,context=context)
    result['protocol']='resident_eager_memory_settle_v3' if conditioned else 'resident_eager_v1'
    result['conditioning']={'protocol':'memory_settle_v3','seconds':1.0,'buffer_bytes':256*1024**2,
        'scope':'conditioned decode only; changes cache history and adds untimed TTFT; not the original unconditioned latency'} if conditioned else None
    if conditioned:result['scope']+='; all thresholds conditional on fixed memory_settle_v3; unconditioned thresholds remain separate'
    write(worker.root/'results'/f'curve_eager_b{batch}_c{context}.json',result)
    if (batch,context)==(1,4096):write(worker.root/'results/curve_eager.json',result)
    return result


def conditioned_followup(worker,observed,proof_elapsed):
    available=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time()
    estimate=(3*(max(observed)+117)+2*proof_elapsed+240)*1.15
    decision={'status':'admitted' if available>estimate+60 else 'deferred_budget',
        'estimate_s':estimate,'available_before_closeout_s':available,
        'classification':'analytical scheduling only; preemption may happen earlier',
        'selection':'one prospectively declared fixed memory_settle_v3 curve after complete unconditioned clock rejection; no original trials replaced or selected by latency',
        'scope':'new root/workspace/frozen cohort, own stock/full/cut NSYS and dynamic/StaticCache gates; no pooled protocols or original TTFT claim'}
    write(worker.root/'results/conditioned_followup_budget.json',decision)
    if decision['status']!='admitted':return decision
    root=worker.root.parent/'gpu0_conditioned'
    for folder in ('environment','results','logs','configs','runs','profiles','analysis'):(root/folder).mkdir(parents=True,exist_ok=False)
    (root/'analysis/report.md').write_text('# Conditioned eager decode: memory_settle_v3\n\nThis is a separate protocol and source cohort. One fixed one-second private-buffer workload follows each prefill barrier, outside decode timing. It changes cache history and TTFT. Unconditioned results are retained separately; no original clock failure is promoted. Own placement and strict dynamic/StaticCache checks precede3fresh acquisitions.\n')
    session=dict(worker.settings)
    session.update(workspace=worker.settings['workspace']+'_conditioned',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip(),
        protocol='resident_eager_memory_settle_v3',parent_unconditioned_root=str(worker.root))
    Path(session['workspace']).mkdir(exist_ok=False)
    path=root/'session.json';write(path,session)
    subprocess.run([worker.py,PACKAGE/'scripts/freeze_shoreline.py','--session',path,'--activate'],check=True)
    with (root/'logs/orchestrator.log').open('x') as stream:
        result=subprocess.run([worker.py,'-u',PACKAGE/'scripts/h200_conditioned_followup.py',
            '--root',root,'--session',path],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        limitation(root,'conditioned_followup_driver','Conditioned driver exited with code '+str(result.returncode),
            'logs/orchestrator.log; results/worker_outcome.json',
            'New root/workspace/source cohort; original unconditioned evidence retained',
            'Inspect the bounded failure before any new identified attempt; no failed gate promotion')
    publish(root,'H200 conditioned follow-up: durable complete/failed driver receipt')
    decision.update(status='orchestrator_finished' if result.returncode==0 else 'orchestrator_failed',
                    exit_code=result.returncode,root=str(root),scientific_gate='see separate curve/disposition; not inferred from exit0')
    write(worker.root/'results/conditioned_followup_budget.json',decision)
    return decision


def primary(worker,conditioned=False,batch=1,context=4096):
    assert batch in (1,8,32) and context in (4096,16384,32704), 'unsupported resident follow-up shape'
    build_environment(worker)
    inventory=worker.execute('inventory',[worker.py,worker.script/'inventory.py','--gpu-uuid',worker.uuid,
                   '--check-cuda','--output',worker.root/'environment/inventory.json'],timeout=180)
    if inventory['status']!='complete':
        result={'status':'blocked_inventory','complete_acquisition_set':False,'allow_context_expansion':False}
        write(worker.root/'results/eager_core_disposition.json',result)
        limitation(worker.root,'resident_inventory','Inventory did not complete; no resident producer launched',
            'logs/inventory.log','Allocation/identity-bound inventory before the pilot','Inspect the preserved inventory failure before a new identified attempt')
        publish(worker.root,'H200 resident route: preserve inventory block before model launch')
        return result
    limitation(worker.root,'resident_protocol_scope','Prior optimized graph restriction remains blocked; repeated model initialization prevented a complete fresh-model-per-point curve before preemption.',
        'h200/20261127_job16210814_priority/analysis/current_work_20261127.md',
        'Versioned one-resident-model-per-acquisition eager protocol; bounded stock/full-green/40SM pilot before full proof and3fresh acquisitions',
        'Never pool prior timings; retain exact outputs, actual decoder placement, all references and clock/isolation gates; optimized restriction remains independently blocked')
    if conditioned:
        limitation(worker.root,'conditioned_scope','Fixed memory_settle_v3 changes cache history and adds untimed TTFT; it does not repair the original unconditioned latency scope.',
            'session.json; configs/*memory_settle_v3.json',
            'Separate prospectively fixed protocol, own actual-decoder proof and dynamic/unchanged StaticCache references',
            'Report thresholds only for this conditioned decode scope, preserving all rejected unconditioned trials')
    pilot,process,good=acquire(worker,0,proof=True,pilot=True,batch=batch,context=context,conditioned=conditioned)
    if good:good=worker.reference(pilot[0]['run'],'resident_pilot')
    if not good:
        result={'status':'blocked_resident_pilot','allow_context_expansion':True,'complete_acquisition_set':False}
        write(worker.root/'results/eager_core_disposition.json',result)
        publish(worker.root,'H200 resident eager pilot: preserve blocked proof/reference and advance eligible full-GPU cases')
        return result
    worker.expected=load(pilot[0]['run'])['cases'][0]['trials'][0]['output_ids']
    proofs,proof_process,good=acquire(worker,0,proof=True,batch=batch,context=context,conditioned=conditioned)
    if good:worker.records.extend(proofs)
    observed=[]
    for acquisition in range(3):
        if not good:break
        available=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time()
        estimate=max(300,max(observed,default=proof_process.get('elapsed_s') or 300)*1.25+30)
        write(worker.root/'results/resident_budget.json',{'acquisition':acquisition,'available_s':available,
            'next_producer_estimate_s':estimate,'observed_s':observed,'admitted':available>=estimate,
            'scope':'analytical scheduling only; preemption can occur earlier than scheduler EndTime'})
        if available<estimate:break
        rows,proc,good=acquire(worker,acquisition,batch=batch,context=context,conditioned=conditioned)
        worker.records.extend(rows);observed.append(proc.get('elapsed_s') or estimate)
        result=analyze(worker,conditioned,batch=batch,context=context)
        from h200_plot_sm_curve import plot
        plot(worker.root,worker.records)
        publish(worker.root,f'H200 resident eager acquisition{acquisition}: publish broad curve and unchanged qualification gates')
    result=analyze(worker,conditioned,batch=batch,context=context)
    complete=all(sum(r['acquisition']==a and not r['pilot'] and r['status']=='complete' and len(r.get('step_ms',[]))==10
                     for r in worker.records)==9 for a in range(3))
    disposition={'status':'resident_curve_disposition','protocol':'resident_eager_memory_settle_v3' if conditioned else 'resident_eager_v1',
        'complete_acquisition_set':complete,'complete_qualified_eager_curve':result['complete_qualified_curve'],
        'allow_context_expansion':True,'reason':'broad curve published; advance eligible full-GPU work, retaining any blocked or clock-unqualified restriction result'}
    write(worker.root/'results/eager_core_disposition.json',disposition)
    publish(worker.root,'H200 resident eager curve: publish final captured/qualified disposition before expansion')
    if not conditioned and complete and not result['complete_qualified_curve'] and (batch,context)==(1,4096):
        disposition['conditioned_followup']=conditioned_followup(worker,observed,proof_process.get('elapsed_s') or max(observed))
        write(worker.root/'results/worker_followup_disposition.json',disposition)
    return disposition


def secondary(worker,conditioned=False):
    """Reuse prior triplicate counters; capture missing full-device context cases."""
    build_environment(worker)
    inventory=worker.execute('inventory',[worker.py,worker.script/'inventory.py','--gpu-uuid',worker.uuid,
                   '--check-cuda','--output',worker.root/'environment/inventory.json'],timeout=180)
    if inventory['status']!='complete':
        limitation(worker.root,'spare_inventory','Inventory did not complete; no spare model producer launched',
            'logs/inventory.log','Allocation/identity-bound inventory before model work','Inspect inventory and preserve the failed attempt')
        return {'status':'blocked_inventory'}
    write(worker.root/'results/reused_work.json',{'scope':'Prior DRAM/L2 and observer pilots retained on original UUID/source cohorts; not repeated by default or pooled with this allocation',
        'DRAM_L2':'h200/20261127_job16210814_priority/gpu1_native_linker/results/P2a_L2_disposition.json',
        'scheduler_pilot':'h200/20261127_job16210814_priority/gpu1_native_linker/results/P3a_observer_pilot.json'})
    from h200_clock_conditioning_pilot import run as clock_pilot
    clock_pilot(worker)
    if conditioned:
        limitation(worker.root,'conditioned_full_context_scope',
            'Full-device context comparisons use the prospectively fixed memory_settle_v3 scope; original unconditioned timings remain separate.',
            'configs/*memory_settle_v3.json; results/clock_conditioning_pilot.json',
            'Own stock placement/HF proof and3x10acquisitions per shape, with all warmup/timed conditioning receipts verified',
            'Interpret decode only; added untimed conditioning changes cache history and TTFT. No deployment optimization or original unconditioned latency claim')
    previous=worker.settings.get('published_core_gate');previous_verified=False
    if previous:
        try:
            assert sha(previous['review'])==previous['review_sha256'],'published core certificate changed'
            verified=verify_published_core(Path(previous['review']))
            write(worker.root/'results/published_core_gate.json',verified);previous_verified=True
        except (OSError,ValueError,KeyError,AssertionError) as error:
            limitation(worker.root,'published_core_gate','Published core could not be reverified: '+str(error),
                str(previous.get('review')),'Fail closed to the current allocation core gate',
                'Inspect source/checksum failure; no old qualification or timing transfers')
    gate=worker.root.parent/'gpu0/results/eager_core_disposition.json'
    while not previous_verified and not gate.exists():
        outcome=worker.root.parent/'gpu0/results/worker_outcome.json'
        if outcome.exists():return {'status':'partial','reason':'core worker ended without an explicit published disposition; context expansion deferred'}
        if time.time()+900>=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']:
            return {'status':'partial','reason':'core disposition pending before context budget'}
        time.sleep(15)
    if not previous_verified and not load(gate)['allow_context_expansion']:
        return {'status':'partial','reason':'core disposition defers context expansion'}
    from h200_eager_first import shape_summary
    outcomes=[]
    for context in (4096,16384,32704):
        worker.expected=None
        if time.time()+900>=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']:break
        proof=worker.case('graph',1.,0,pilot=True,stock=True,context=context,suffix='_resident_followup_shape_proof',conditioned=conditioned)
        if not proof['qualified'] or not worker.reference(proof['run'],f'full_graph_c{context}'):
            outcomes.append({'context':context,'status':'blocked proof/HF'});continue
        worker.expected=load(proof['run'])['cases'][0]['trials'][0]['output_ids']
        rows=[]
        for acquisition in range(3):
            if time.time()+max(180,proof['producer_elapsed_s']*1.5)>=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']:break
            rows.append(worker.case('graph',1.,acquisition,stock=True,context=context,suffix='_resident_followup_full_shape',conditioned=conditioned))
        outcomes.append(shape_summary(worker,rows,'graph',context))
        publish(worker.root,f'H200 full-GPU context{context}: independent matched-shape acquisition bundle')
        if len(rows)<3:break
    return {'status':'partial_scientific_gates_separate','full_GPU_shapes':outcomes}
