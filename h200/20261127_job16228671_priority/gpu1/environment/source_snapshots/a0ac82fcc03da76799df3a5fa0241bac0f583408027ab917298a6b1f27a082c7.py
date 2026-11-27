"""Interruptible two-H200 continuation after recorded graph-repair failures."""
import datetime as dt
import json
from pathlib import Path
import re
import statistics
import time
from h200_priority_campaign import load,write,publish,limitation,config,valid_tokens,sha


def shape_summary(worker,rows,engine,context):
    qualified=(len(rows)==3 and {r['acquisition'] for r in rows}=={0,1,2}
               and all(r.get('qualified') and len(r.get('step_ms',[]))==10 for r in rows)
               and len({r['source_identity'] for r in rows})==1)
    pooled={}
    if len(rows)==3 and all('clock' in r and 'clocks' in r['clock'] for r in rows):
        pooled={m:max(r['clock']['clocks'][m]['maximum'] for r in rows)/min(r['clock']['clocks'][m]['minimum'] for r in rows)
                for m in ('sm_clock_mhz','memory_clock_mhz')}
        qualified=qualified and max(pooled.values())<=1.05
    else:qualified=False
    from shoreline_statistics import hierarchical_interval
    result={'engine':engine,'batch':1,'context':context,'qualified_timing':qualified,
            'raw_acquisition_medians_ms':[statistics.median(r['step_ms']) if r.get('step_ms') else None for r in rows],
            'latency_ms':hierarchical_interval([r['step_ms'] for r in rows]) if qualified else None,
            'pooled_clock_ratios':pooled,'sources':[r['id'] for r in rows],
            'scope':'full-GPU shape only; clock-rejected rows retained; no restricted threshold'}
    write(worker.root/'results'/f'full_graph_b1_c{context}.json',result)
    return result


def scheduler_pilot(worker):
    """Three independent clean/observed pairs; exact outputs and overhead separate."""
    records=[];expected=None
    for acq in range(3):
        for observed in (False,True):
            label=f'P3a_observer_{"observed" if observed else "clean"}_a{acq}'
            cfg=config(worker.root,'graph',4096,10,1);output=worker.root/'runs'/label
            command=[worker.py,worker.script/'h200_scheduler_pilot.py','--config',cfg,
                     '--gpu-uuid',worker.uuid,'--output',output]
            if observed:command+=['--observe']
            proc=worker.execute(label,command,timeout=1200)
            path=output/'run.json';data=load(path) if path.exists() else {}
            passed=proc['status']=='complete' and valid_tokens(data,expected) and len(data['cases'][0]['trials'])==10
            if passed and expected is None:
                if not worker.reference(path,'P3a_clean'):passed=False
                else:expected=data['cases'][0]['trials'][0]['output_ids']
            row={'acquisition':acq,'observed':observed,'producer_complete':proc['status']=='complete',
                 'exact_outputs':bool(passed),'run':str(path),'ordinary_latency_eligible':False,
                 'gpu_uuid':worker.uuid,'source_freeze_identity':worker.settings['frozen_source_identity'],
                 'scope':'observer/input-composition pilot; no loaded serving or mixed-policy comparison'}
            if passed:
                import shoreline_campaign
                row['clock']=shoreline_campaign.clock_qualification(data,proc['monitor']['source'])
                row['raw_step_ms']=[t['wall_ms']/t['decode_iterations'] for t in data['cases'][0]['trials']]
                row['run_sha256']=sha(path)
                if not row['clock']['passed']:
                    limitation(worker.root,label,'Observer pilot clock envelope failed; all trials retained and no clean/observed latency ratio qualified',
                                str(path),'Passive clock evidence; matched clean and observed producers retained',
                                'Review pooled acquisition gates and observer intrusion separately; no favorable resampling')
            records.append(row);write(worker.root/'results/P3a_observer_pilot.json',{'status':'partial','records':records})
            if not passed:
                limitation(worker.root,label,'Scheduling observer producer/exact-output gate failed',str(path),
                            'Bounded matched clean/observed pilot with unmodified output gate',
                            'Inspect raw scheduling decisions; do not claim tuned loaded serving or intrusion-neutral timing')
                publish(worker.root,'H200 scheduling observer pilot: preserve blocked gate');return
        publish(worker.root,f'H200 scheduler observer paired acquisition{acq}: instrumented scope explicit')
    write(worker.root/'results/P3a_observer_pilot.json',{'status':'six_producers_completed_exact_outputs',
          'records':records,'qualified_clean_latency_ratio':None,
          'scope':'B1/C4096 observer/intrusion pilot only. Mixed policies/load/SLO/held-out traces remain unmeasured; pooled clocks and intrusion interpretation require review.'})
    publish(worker.root,'H200 scheduler observer: publish three paired acquisitions, retain interpretation limits')


def primary(worker):
    inv=worker.execute('inventory',[worker.py,worker.script/'inventory.py','--gpu-uuid',worker.uuid,
                       '--check-cuda','--output',worker.root/'environment/inventory.json'],timeout=180)
    if inv['status']!='complete':
        write(worker.root/'results/eager_core_disposition.json',{'status':'blocked','reason':'inventory','allow_context_expansion':False})
        return {'status':'blocked','reason':'inventory'}
    limitation(worker.root,'optimized_restriction_prior_block','Earlier job16184819 measured graph context escape and a bounded capture repair failed during graph-memory profiling; this cohort prioritizes the explicitly separate eager sweep.',
                'h200/20261127_job16184819_priority/analysis/report.md','Preserve graph failures; no repeated graph repair on this allocation',
                'Require separate eager stock/full-green/cut proofs on this UUID; optimized restriction stays blocked')
    worker.rebind_step=True;worker.capture_bind=False
    stock=worker.case('eager',1.,0,pilot=True,stock=True)
    if not stock['qualified'] or not worker.reference(stock['run'],'eager_target'):
        write(worker.root/'results/eager_core_disposition.json',{'status':'blocked','reason':'eager stock NSYS/HF gate','allow_context_expansion':False})
        return {'status':'blocked','reason':'eager stock NSYS/HF gate'}
    worker.expected=load(stock['run'])['cases'][0]['trials'][0]['output_ids']
    qualified=worker.curve('eager',stock_proof=stock)
    disposition={'status':'eager_curve_attempted','complete_qualified_eager_curve':qualified,
                 'stop':worker.last_curve_stop,'allow_context_expansion':qualified or worker.last_curve_stop in ('proof','quality'),
                 'scope':'Eager same-engine controls; never optimized decoder normalization'}
    publish(worker.root,'H200 first-priority separate eager curve: publish disposition before expansion')
    write(worker.root/'results/eager_core_disposition.json',disposition)
    if time.time()+1800<dt.datetime.fromisoformat(worker.settings['launch_cutoff']).timestamp():scheduler_pilot(worker)
    return disposition


def secondary(worker):
    inv=worker.execute('inventory',[worker.py,worker.script/'inventory.py','--gpu-uuid',worker.uuid,
                       '--check-cuda','--output',worker.root/'environment/inventory.json'],timeout=180)
    if inv['status']!='complete':return {'status':'blocked','reason':'inventory'}
    # Add the missing L2 observation at the original target before expanding.
    query=worker.execute('P2a_query_L2_metric',[worker.cuda/'bin/ncu','--query-metrics',
                '--query-metrics-mode','suffix','--metrics','lts__t_bytes'],timeout=120)
    text=(worker.root/'logs/P2a_query_L2_metric.log').read_text()
    supported=query['status']=='complete' and re.search(r'(?<![\w.])lts__t_bytes\.sum(?![\w.])',text) is not None
    if supported:
        worker.execute('P2a_ncu_version',[worker.cuda/'bin/ncu','--version'],gpu=False,timeout=30)
        label='P0_native_L2_counter'
        report=worker.root/'profiles'/(label+'.ncu-rep');output=worker.root/'runs'/label
        command=[worker.cuda/'bin/ncu','--replay-mode','kernel','--clock-control','none',
            '--cache-control','none','--profile-from-start','off',
            '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_bytes.sum',
            '--export',report,worker.py,worker.script/'shoreline_native.py','--gpu-uuid',worker.uuid,
            '--output',output,'--acquisition','0','--pilot','--profile-case','stream_512mib']
        proc=worker.execute(label,command,timeout=worker.settings.get('native_counter_timeout_s',600))
        valid=False;values={};error=None
        try:
            from h200_ncu_read import resolve_report,parse_csv
            raw=resolve_report(report)
            imported=worker.execute(label+'_import',[worker.cuda/'bin/ncu','--import',raw,
                       '--page','raw','--print-units','base','--csv'],gpu=False,timeout=180)
            values=parse_csv((worker.root/'logs'/(label+'_import.log')).read_text(),include_l2=True)
            native=load(output/'run.json')
            valid=(proc['status']=='complete' and imported['status']=='complete' and native['status']=='complete'
                   and len(native['cases'])==1 and native['cases'][0]['id']=='stream_512mib'
                   and native['cases'][0]['correctness']=='passed' and len(values)==1
                   and all(v['dram__bytes_read.sum']>0 and v['lts__t_bytes.sum']>0 for v in values.values()))
        except (ValueError,OSError,KeyError) as exception:error=str(exception)
        write(worker.root/'results/P0_native_L2_counter_review.json',{'status':'permission_functional_counter_pilot_passed' if valid else 'blocked',
              'ranges':values,'error':error,'scope':'single known-value512MiB native control; not calibrated physical service or decoder throughput'})
        if valid:
            traffic=worker.traffic(include_l2=True)
            write(worker.root/'results/P2a_L2_disposition.json',traffic)
        else:
            limitation(worker.root,'P0_native_L2_counter','Known-payload DRAM/L2 pilot failed a required producer/output/reader gate',
                        'results/P0_native_L2_counter_review.json','Single native capture before expensive model counters; matching allocated reader',
                        'Inspect metric units/permission/output evidence; full-GPU unprofiled work remains eligible')
    else:
        limitation(worker.root,'P2a_L2_metric','Allocated profiler did not advertise the exact requested L2 total-byte metric',
                    'logs/P2a_query_L2_metric.log','Read-only allocated metric query; no guessed sector-to-byte conversion',
                    'Inspect supported names/units before a versioned L2 capture; prior DRAM-only evidence stays separate')
    publish(worker.root,'H200 full-target L2 counter comparison or explicit blocking evidence')
    if worker.settings.get('spare_scheduler_pilot') and time.time()+1800<dt.datetime.fromisoformat(worker.settings['launch_cutoff']).timestamp():
        scheduler_pilot(worker)
    gate=worker.root.parent/'gpu0/results/eager_core_disposition.json'
    while not gate.exists():
        failed=worker.root.parent/'gpu0/results/worker_outcome.json'
        if failed.exists():return {'status':'partial','reason':'core worker exited without curve disposition; contexts deferred'}
        allocation=worker.hs.verify_allocation()
        if time.time()+1200>=allocation['effective_launch_cutoff_unix_s']:
            return {'status':'partial','reason':'core disposition pending before remaining-case budget'}
        time.sleep(30)
    if not load(gate)['allow_context_expansion']:
        return {'status':'partial','reason':'core curve unfinished at budget boundary; contexts deferred'}
    # The broad core has a published disposition before any longer context.
    outcomes=[]
    for context in (4096,16384,32704):
        worker.expected=None
        proof=worker.case('graph',1.,0,pilot=True,stock=True,context=context,suffix='_long_shape_proof')
        if not proof['qualified'] or not worker.reference(proof['run'],f'full_graph_c{context}'):
            outcomes.append({'context':context,'status':'blocked proof/HF'});continue
        worker.expected=load(proof['run'])['cases'][0]['trials'][0]['output_ids']
        rows=[]
        for acq in range(3):
            remaining=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time()
            if remaining<max(300,proof['producer_elapsed_s']*(13/6)*1.25):break
            rows.append(worker.case('graph',1.,acq,stock=True,context=context,suffix='_full_shape'))
        outcome=shape_summary(worker,rows,'graph',context);outcomes.append(outcome)
        publish(worker.root,f'H200 full-GPU context{context}: preserve independent shape acquisitions')
        if len(rows)<3:break
    return {'status':'partial_scientific_gates_separate','full_GPU_shapes':outcomes,'core_gate':str(gate)}
