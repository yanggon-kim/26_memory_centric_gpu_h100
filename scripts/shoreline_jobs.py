#!/usr/bin/env python3
"""Bounded native J-job acquisitions with preserved attempts and per-job pushes."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def load(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attempt(label,command,gpu=True,timeout=900):
    saved=ROOT/'results/processes'/(label+'.json')
    return load(saved) if saved.exists() else run(ROOT,label,command,timeout,gpu=gpu)


def publish(message):campaign.checkpoint(ROOT,message)


def disposition(job,status,reason):
    campaign.update_job(ROOT,job,status,reason)
    publish(f'{job}: {reason}')


def reference(model,engine,batch,context):
    name=f'{model}_{engine}_b{batch}_c{context}_acq00.json'
    q=ROOT/'results/qualifications'/name
    if not q.exists():return None
    row=load(q);p=ROOT/row['output']/'run.json'
    if not p.exists():return None
    data=load(p)
    if data['status']!='complete' or not data['cases'][0]['output_digests_identical']:return None
    return p


def profile_config(model,engine,batch,context,label):
    base=ROOT/'configs'/f'{model}_{engine}_b{batch}_c{context}_t10.json'
    config=load(base);config['trials']=3
    config['shoreline_protocol'].update(trials_per_acquisition=3,
        scope='instrumented counter/timeline process; not ordinary timing; first measured trial ticks 4/5/6 selected')
    target=ROOT/'configs'/(label+'.json')
    if target.exists():assert load(target)==config
    else:dump(target,config)
    return target


def records(path):
    lines=Path(path).read_text(errors='replace').splitlines()
    i=next((i for i,s in enumerate(lines) if s.startswith('"ID",')),None)
    if i is None:return []
    reader=csv.DictReader(io.StringIO('\n'.join(lines[i:])))
    rows=list(reader)
    if 'Metric Name' in (reader.fieldnames or []):return [r for r in rows if r.get('Metric Name')]
    # Newer Nsight raw CSV is wide: one units row, then one row per result.
    if not rows:return []
    units=rows[0] if not rows[0].get('ID') else {}
    output=[]
    for row in rows:
        if not row.get('ID'):continue
        common={k:row[k] for k in ('ID','Process ID','Kernel Name','Context','Stream') if k in row}
        for metric in reader.fieldnames:
            if '__' in metric and not metric.startswith('device__'):
                output.append({**common,'Metric Name':metric,'Metric Unit':units.get(metric,''),
                               'Metric Value':row.get(metric),'CSV Layout':'wide normalized to metric rows'})
    return output


def value(raw):
    try:return float(raw.replace(',',''))
    except (TypeError,ValueError,AttributeError):return None


def counter_case(model,engine,batch,context,acquisition,scope='full',layer=None,job='J2',pressure=None,fraction=None):
    protocol=load(ROOT/'session.json').get('profile_protocol','range_api_v2')
    treatment=(f'_{pressure}' if pressure else '')+(f'_f{int(fraction*100)}' if fraction is not None else '')
    label=f'{job}_{model}_{engine}_b{batch}_c{context}_{scope}{treatment}_acq{acquisition:02d}_{protocol}'
    result_path=ROOT/'profiles'/(label+'_reader_v3.json')
    if result_path.exists():return load(result_path)
    ref=reference(model,engine,batch,context)
    if ref is None and job=='J7':
        candidate=ROOT/'runs'/f'J7_{model}_eager_b{batch}_c{context}_normal_eventsnone_acq00/run.json'
        if candidate.exists() and load(candidate)['status']=='complete':ref=candidate
    if ref is None:
        result={'id':label,'status':'unavailable','reason':'no correct full-length baseline for exact output comparison'}
        dump(result_path,result);return result
    if not (ROOT/'configs'/f'{model}_{engine}_b{batch}_c{context}_t10.json').exists():campaign.make_config(ROOT,model,engine,batch,context)
    config=profile_config(model,engine,batch,context,label)
    base=ROOT/'runs'/label
    report=ROOT/'profiles'/(label+'.ncu-rep')
    include='shoreline_ticks_4_6/' if scope=='full' else f'layer_{layer}/'
    cmd=[NCU,'--replay-mode','app-range','--clock-control','none',
         '--cache-control','none','--nvtx','--nvtx-include',include,
         '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
         '--export',str(report),str(PYTHON),str(PROGRAMS/'shoreline_replay.py'),
         '--config',str(config),'--gpu-uuid',UUID,'--mode','timing','--output',str(base),
         '--external-profiler','--generated-tokens','64']
    if pressure:cmd+=['--pressure',pressure]
    if fraction is not None:cmd+=['--fraction',str(fraction)]
    outcome=attempt(label,cmd,timeout=900)
    passed=[];problems=[]
    expected=load(ref)['cases'][0]
    from shoreline_evidence import qualify_replays
    replay_proof=qualify_replays(base,expected,ROOT)
    passed=replay_proof['passes']
    if not replay_proof['passed']:problems.append('incomplete/mismatched/unpublished application replay')
    if report.exists():
        attempt(label+'_import_base',[NCU,'--import',str(report),'--page','raw','--print-units','base','--csv'],gpu=False,timeout=120)
    raw=ROOT/'logs'/(label+'_import_base.log')
    rows=records(raw) if raw.exists() else []
    ids=sorted(set(r['ID'] for r in rows))
    counts={}
    for ident in ids:
        counts[ident]={r['Metric Name']:value(r.get('Metric Value')) for r in rows if r['ID']==ident}
    expected_ranges=1 if scope=='full' else 3
    metrics_valid=(len(ids)==expected_ranges and all(v.get('dram__bytes_read.sum') is not None and
                    v['dram__bytes_read.sum']>0 and v.get('dram__bytes_write.sum') is not None and
                    v['dram__bytes_write.sum']>=0 for v in counts.values()))
    valid=outcome['status']=='complete' and replay_proof['passed'] and metrics_valid
    result={'id':label,'status':'complete' if valid else 'failed','model':model,'engine':engine,
            'batch':batch,'context':context,'acquisition_id':acquisition,'scope':scope,'layer':layer,
            'classification':'measured-real-GPU-instrumented','ticks':[4,5,6],'token_denominator':3*batch,
            'pressure':pressure,'requested_sm_fraction':fraction,
            'source_config_sha256':digest(config),'replay_mode':'app-range','cache_control':'none','clock_control':'none',
            'native_replay_publications':passed,'metrics':rows,'ranges':counts,'expected_range_count':expected_ranges,
            'hbm_read_bytes':sum(v['dram__bytes_read.sum'] for v in counts.values()) if metrics_valid else None,
            'hbm_write_bytes':sum(v['dram__bytes_write.sum'] for v in counts.values()) if metrics_valid else None,
            'errors':problems,'missing_reason':None if valid else 'counter/replay/output/range qualification failed; raw attempts preserved',
            'clock_join_qualification':'pending pooled per-trial/replay endpoint checks; no timing join inferred',
            'ordinary_latency':None,'ordinary_latency_reason':'profile time is instrumented and not ordinary decode latency'}
    result['reader_version']=3
    result['profile_protocol']=protocol
    result['raw_capture_reused_without_recapture']=report.exists()
    dump(result_path,result)
    publish(f'{job} acquisition checkpoint: {label} {result["status"]}')
    return result


def traffic():
    campaign.update_job(ROOT,'J2','in_progress')
    # First prove the whole-range producer and reader, then expand scope.
    first=counter_case('bridge','eager',1,1024,0)
    if first['status']!='complete':
        disposition('J2','partial','whole-range pilot failed; raw failure retained; further range expansion deferred pending diagnosis')
        return
    if args.pilot:
        publish('J2 checkpoint: bounded whole-decode producer/reader pilot passed; expansion pending')
        return
    results=[first]
    for model,batch,context in CONTRASTS:
        for engine in ('eager','graph'):
            for acq in range(3):results.append(counter_case(model,engine,batch,context,acq))
        selected=load(ROOT/'session.json').get('traffic_layer_contrasts',CONTRASTS)
        if [model,batch,context] not in [list(c) for c in selected]:continue
        layout=load(ROOT/'configs'/f'{model}_smoke.json')['layout']
        for layer in sorted(set([0,layout['num_hidden_layers']//2,layout['num_hidden_layers']-1])):
            for acq in range(3):results.append(counter_case(model,'eager',batch,context,acq,'layer'+str(layer),layer))
    dump(ROOT/'results/shoreline_traffic_qualification.json',{'cases':results,
        'capture_coverage_complete':all(r['status']=='complete' for r in results),
        'comparative_clock_timing_join':'separate pooled clock/source/isolation analysis required'})
    disposition('J2','partial',
                'ticks-4/5/6 physical traffic and exact replay qualification published; comparative clock/timing joins remain separate scientific gates')


def native():
    campaign.update_job(ROOT,'J4','in_progress')
    label='J4_native_pilot_indices_v4'
    out=ROOT/'runs'/label
    outcome=attempt(label,python_command('shoreline_native.py','--gpu-uuid',UUID,'--output',out,
                                      '--acquisition','0','--pilot'),timeout=300)
    if outcome['status']!='complete':
        disposition('J4','partial','native service pilot failed; preserve failure before expanding')
        return
    publish('J4 checkpoint: native service producer and reference pilot passed')
    for acq in range(3):
        label=f'J4_native_v4_acq{acq:02d}';out=ROOT/'runs'/label
        outcome=attempt(label,python_command('shoreline_native.py','--gpu-uuid',UUID,'--output',out,
                                            '--acquisition',str(acq)),timeout=900)
        publish(f'J4 acquisition checkpoint: {label} {outcome["status"]}')
    disposition('J4','partial','native service acquisitions saved; physical-counter, mixed/reduction and clock joins retain separate qualification')


def service_extensions():
    campaign.update_job(ROOT,'J4','in_progress','read-only service and six-condition overlap calibration extension')
    label='J4_service_extension_pilot_v1';out=ROOT/'runs'/label
    result=attempt(label,python_command('shoreline_service_extensions.py','--gpu-uuid',UUID,
                 '--output',out,'--acquisition','0','--pilot'),timeout=300)
    publish(f'J4 extension pilot checkpoint: {result["status"]}')
    if result['status']!='complete':
        disposition('J4','partial','read-only native extension pilot failed; raw failure retained before expansion');return
    outcomes=[]
    for acq in range(3):
        label=f'J4_service_extension_acq{acq:02d}';out=ROOT/'runs'/label
        result=attempt(label,python_command('shoreline_service_extensions.py','--gpu-uuid',UUID,
                     '--output',out,'--acquisition',str(acq)),timeout=600)
        outcomes.append({'id':label,'status':result['status']})
        publish(f'J4 extension acquisition checkpoint: {label} {result["status"]}')
    label='J4_same_pass_overlap';out=ROOT/'runs'/label;report=ROOT/'profiles'/label
    result=attempt(label,[str(CUDA/'bin/nsys'),'profile','--sample=none','--cpuctxsw=none',
        '--trace=cuda,nvtx','--cuda-graph-trace=node','--capture-range=cudaProfilerApi',
        '--capture-range-end=stop','--output='+str(report)]+python_command('shoreline_service_extensions.py',
        '--gpu-uuid',UUID,'--output',out,'--acquisition','0','--timeline'),timeout=600)
    proof={'status':'failed','producer_status':result['status']}
    db=report.with_suffix('.sqlite')
    if report.with_suffix('.nsys-rep').exists() and not db.exists():
        attempt(label+'_export',[str(CUDA/'bin/nsys'),'export','--type','sqlite','--output',str(db),str(report.with_suffix('.nsys-rep'))],gpu=False,timeout=120)
    if db.exists() and result['status']=='complete':
        from shoreline_timeline import native_overlap
        try:proof.update(native_overlap(db))
        except Exception as error:proof['reason']=str(error)
    dump(ROOT/'results/J4_same_pass_overlap.json',proof)
    publish(f'J4 same-pass overlap checkpoint: {proof["status"]}')
    physical=[]
    for case in ('read_only_512mib','read_stride17_512mib'):
        for acq in range(3):
            label=f'J4_counter_{case}_acq{acq:02d}';out=ROOT/'runs'/label;report=ROOT/'profiles'/(label+'.ncu-rep')
            command=[NCU,'--replay-mode','kernel','--profile-from-start','off','--clock-control','none',
                     '--cache-control','none','--nvtx','--nvtx-include',case+'/',
                     '--launch-count','1','--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
                     '--export',str(report)]+python_command('shoreline_service_extensions.py','--gpu-uuid',UUID,
                     '--output',out,'--acquisition',str(acq),'--profile-case',case)
            result=attempt(label,command,timeout=300)
            if report.exists():attempt(label+'_import',[NCU,'--import',str(report),'--page','raw','--print-units','base','--csv'],gpu=False,timeout=120)
            csv_path=ROOT/'logs'/(label+'_import.log');rows=records(csv_path) if csv_path.exists() else []
            selected={r['Metric Name']:value(r.get('Metric Value')) for r in rows}
            valid=result['status']=='complete' and len(set(r['ID'] for r in rows))==1 and selected.get('dram__bytes_read.sum',0)>0
            entry={'id':label,'status':'complete' if valid else 'failed','case':case,'acquisition_id':acq,
                   'metrics':rows,'physical_read_bytes':selected.get('dram__bytes_read.sum'),
                   'physical_write_bytes':selected.get('dram__bytes_write.sum'),
                   'classification':'measured-real-GPU-instrumented','scope':'single independent native service launch; kernel replay cache none',
                   'ordinary_time':'separate native acquisition; profiled duration excluded'}
            dump(ROOT/'profiles'/(label+'.json'),entry);physical.append(entry)
            publish(f'J4 physical service checkpoint: {label} {entry["status"]}')
    dump(ROOT/'results/shoreline_service_extension_qualification.json',{'ordinary':outcomes,'physical':physical,
        'native_fit':'requires common-family held-out validation and clock joins; no transfer to decoder assumed'})
    from shoreline_fit import fit
    fit(ROOT)
    native_counter_suite()
    disposition('J4','partial','native service, read-only physical bytes and overlap-family acquisitions published; derived clock/fit gates tracked separately')


def native_counter_suite(job='J4',fraction=1,selected=None):
    results=[]
    selected=selected or load(ROOT/'session.json').get('native_counter_conditions',['stream_512mib','gather_512mib'])
    suffix=f'_f{int(fraction*100)}' if fraction!=1 else ''
    for case,acq in [('stream_512mib',0)]+[(case,acq) for case in selected for acq in range(3)]:
        label=f'{job}_native_counter_{case}{suffix}_acq{acq:02d}'
        target=ROOT/'profiles'/(label+'.json')
        if target.exists():results.append(load(target));continue
        out=ROOT/'runs'/label;report=ROOT/'profiles'/(label+'.ncu-rep')
        cmd=[NCU,'--replay-mode','kernel','--profile-from-start','off','--clock-control','none',
             '--cache-control','none','--nvtx','--nvtx-include',case+'/','--launch-count','1',
             '--section','LaunchStats','--section','Occupancy',
             '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
             '--export',str(report)]+python_command('shoreline_native.py','--gpu-uuid',UUID,'--output',out,
                '--acquisition',str(acq),'--profile-case',case)
        if fraction!=1:cmd+=['--fraction',str(fraction)]
        outcome=attempt(label,cmd,timeout=600)
        if report.exists():
            attempt(label+'_import',[NCU,'--import',str(report),'--page','raw','--print-units','base','--csv'],gpu=False,timeout=120)
            attempt(label+'_sass',[NCU,'--import',str(report),'--page','source','--print-source','sass'],gpu=False,timeout=120)
        csv_path=ROOT/'logs'/(label+'_import.log');rows=records(csv_path) if csv_path.exists() else []
        metrics={r['Metric Name']:value(r.get('Metric Value')) for r in rows}
        valid=(outcome['status']=='complete' and len({r['ID'] for r in rows})==1 and
               all(metrics.get(m) is not None and metrics[m]>=0 for m in ('dram__bytes_read.sum','dram__bytes_write.sum')))
        native=out/'run.json'
        if native.exists():
            d=load(native)
            valid=valid and d['status']=='complete' and len(d['cases'])==1 and d['cases'][0].get('correctness')=='passed'
        entry={'id':label,'case':case,'acquisition_id':acq,'status':'complete' if valid else 'failed',
               'requested_sm_fraction':fraction,
               'classification':'measured-real-GPU-instrumented','metrics':rows,
               'scope':'one native launch, kernel replay cache/clock none; ordinary time comes from independent native suite',
               'physical_read_bytes':metrics.get('dram__bytes_read.sum'),'physical_write_bytes':metrics.get('dram__bytes_write.sum'),
               'loaded_symbols':sorted({r.get('Kernel Name','') for r in rows}),
               'dynamic_issued_work':None,'issued_work_reason':'loaded SASS export and useful operation counts do not prove issued dynamic tensor work'}
        if native.exists():
            entry['allocation']=load(native)['allocation']
            entry['actual_sm_case_id']=label+'_sm'+str(entry['allocation']['actual_sm_count'])
        dump(target,entry);results.append(entry);publish(f'{job} native counter checkpoint: {label} {entry["status"]}')
        if len(results)==1 and not valid:break
    target_name='shoreline_native_counter_qualification'+(f'_{job}'+suffix if job!='J4' else '')+'.json'
    dump(ROOT/'results'/target_name,{'cases':list({r['id']:r for r in results}.values()),
         'selection':selected,'residency_interpretation':'zero physical reads are permitted for warmed below-L2 controls; never substitute logical payload bytes'})


def timeline_case(model,engine,batch,context,acq):
    label=f'J1_timeline_{model}_{engine}_b{batch}_c{context}_acq{acq:02d}'
    if engine=='graph':label+='_graph_nodes_v2'
    original=label
    for retry in range(3):
        label=original+(f'_attempt{retry}' if retry else '')
        result_path=ROOT/'profiles'/(label+'.json')
        if not result_path.exists():break
        previous=load(result_path)
        producer=ROOT/'results/processes'/(label+'.json')
        if previous['status']=='complete' or not producer.exists() or not load(producer).get('monitor',{}).get('failure') or retry==2:
            return previous
    ref=reference(model,engine,batch,context)
    if ref is None:
        result={'status':'unavailable','id':label,'reason':'correct baseline unavailable'}
        dump(result_path,result);return result
    config=profile_config(model,engine,batch,context,label)
    output=ROOT/'runs'/label;report=ROOT/'profiles'/label
    command=[str(CUDA/'bin/nsys'),'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--cuda-graph-trace=node',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(report),
             str(PYTHON),str(PROGRAMS/'shoreline_decode.py'),'--config',str(config),
             '--gpu-uuid',UUID,'--output',str(output),'--mode','timing','--external-profiler']
    outcome=attempt(label,command,timeout=600)
    db=report.with_suffix('.sqlite')
    if report.with_suffix('.nsys-rep').exists() and not db.exists():
        attempt(label+'_export',[str(CUDA/'bin/nsys'),'export','--type','sqlite','--output',str(db),
                                str(report.with_suffix('.nsys-rep'))],gpu=False,timeout=180)
    result={'id':label,'status':'failed','model':model,'engine':engine,'batch':batch,'context':context,
            'acquisition_id':acq,'classification':'measured-real-GPU-instrumented','ticks':[4,5,6]}
    if db.exists() and (output/'run.json').exists():
        from shoreline_timeline import read
        try:
            result.update(read(db))
            if engine=='graph' and not all(w['graph_kernel_count']>0 for w in result['windows']):
                result.update(status='failed',reason='graph-node activity missing; outside-graph kernels are not complete decode coverage')
            data=load(output/'run.json');expected=load(ref)['cases'][0]['trials'][0]['output_ids']
            exact=data['status']=='complete' and all(t['output_ids']==expected for c in data['cases'] for t in c['trials'])
            result['exact_reference_ids']=exact
            if not exact or outcome['status']!='complete':result['status']='failed'
            result['database']={'path':str(db.relative_to(ROOT)),'sha256':digest(db)}
        except Exception as error:result.update(status='failed',reason=str(error))
    dump(result_path,result);publish(f'J1 timeline checkpoint: {label} {result["status"]}')
    return result


def timelines():
    campaign.update_job(ROOT,'J1','in_progress','collect bounded actual decode timelines; ordinary acquisitions already preserved')
    outcomes=[]
    for model,b,c in CONTRASTS:
        for engine in ('graph','eager'):
            for acq in range(3):outcomes.append(timeline_case(model,engine,b,c,acq))
    dump(ROOT/'results/shoreline_timeline_qualification.json',{'cases':outcomes})
    ordinary=load(ROOT/'results/baseline_qualification.json')
    disposition('J1','complete' if ordinary['all_conditions_qualified'] and all(r['status']=='complete' for r in outcomes) else 'partial',
                'focused ordinary baselines and actual decode timelines published; rejected clocks/outputs remain excluded')


def representative_text():
    """Small functional saved-text pilot; no three-acquisition timing claim."""
    results=[];expected=None
    for engine in ('graph','eager'):
        label=f'J1_text_bridge_{engine}_b1_c128_pilot_v1';base=campaign.make_config(ROOT,'bridge',engine,1,128,3)
        config=load(base);config['workload_file']='configs/representative_text_v1.json'
        config['shoreline_protocol']['scope']='functional saved-text pilot only; not a3x10 timing comparison'
        config.pop('config_sha256',None);config['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        path=ROOT/'configs'/(label+'.json');dump(path,config);out=ROOT/'runs'/label
        row=attempt(label,python_command('controlled_vllm_experiments.py','--config',path,'--gpu-uuid',UUID,
                    '--mode','timing','--output',out),timeout=300)
        result={'id':label,'status':row['status'],'scope':'functional saved-text exact-output pilot; timing not a qualified comparison'}
        if (out/'run.json').exists() and row['status']=='complete':
            d=load(out/'run.json');ids=d['cases'][0]['trials'][0]['output_ids']
            if expected is None:expected=ids
            result['exact_mode_match']=all(t['output_ids']==expected for c in d['cases'] for t in c['trials'])
            if not result['exact_mode_match']:result['status']='failed'
            reference_path=ROOT/'results'/(label+'_hf_dynamic.json')
            ref=attempt(label+'_hf_dynamic',python_command('hf_full_reference.py','--source-run',out/'run.json',
                        '--gpu-uuid',UUID,'--output',reference_path),timeout=300)
            result['dynamic_reference_status']=ref['status']
            if ref['status']!='complete':result['status']='failed'
            result['static_cache_scope']='existing synthetic model smoke gate retained; this additional reference is explicitly dynamic and correctness-only'
        dump(ROOT/'profiles'/(label+'.json'),result);results.append(result)
        publish(f'J1 saved-text functional checkpoint: {label} {result["status"]}')
    dump(ROOT/'results/shoreline_representative_text_qualification.json',{'cases':results,
        'scope':'saved original text, B1/C128, BF16, fixed64 outputs; three trials are a functional pilot and not quantitative acquisition replication'})
    publish('J1: preserve saved-text functional controls and separate reference scope')


def kernel_profiles():
    campaign.update_job(ROOT,'J3','in_progress')
    from profile_h200 import METRICS,BASE_METRICS
    extra=['smsp__issue_active.avg.pct_of_peak_sustained_active',
           'smsp__warps_eligible.avg.per_cycle_active']
    catalog_label='J3_metric_catalog'
    attempt(catalog_label,[NCU,'--query-metrics','--query-metrics-mode','suffix','--metrics',
                          ','.join(BASE_METRICS+['smsp__issue_active','smsp__warps_eligible'])],gpu=False,timeout=120)
    catalog=(ROOT/'logs'/(catalog_label+'.log')).read_text()
    requested=[m for m in METRICS+extra if re.search(r'^'+re.escape(m)+r'\s',catalog,re.M)]
    unavailable=sorted(set(METRICS+extra)-set(requested))
    results=[]
    for model,batch,context in load(ROOT/'session.json').get('kernel_profile_contrasts',CONTRASTS):
        ref=reference(model,'eager',batch,context)
        if ref is None:
            results.append({'model':model,'status':'unavailable','reason':'correct eager baseline unavailable'});continue
        timeline=timeline_case(model,'eager',batch,context,0)
        symbols=[s for w in timeline.get('windows',[]) for s in w['kernels']]
        layers=load(ROOT/'configs'/f'{model}_smoke.json')['layout']['num_hidden_layers']
        mid=layers//2
        roles=[('qkv',f'operation_layer_{mid}_self_attn_qkv_proj/',None),
               ('gate_up',f'operation_layer_{mid}_mlp_gate_up_proj/',None),
               ('down',f'operation_layer_{mid}_mlp_down_proj/',None)]
        for role,pattern in [('attention_forward',lambda s:('flash' in s.lower() or 'attn' in s.lower()) and 'combine' not in s.lower()),
                             ('attention_combine',lambda s:'combine' in s.lower()),
                             ('kv_update',lambda s:'reshape_and_cache' in s.lower())]:
            symbol=next((s for s in symbols if pattern(s)),None)
            if symbol:roles.append((role,f'layer_{mid}/',symbol))
            else:results.append({'model':model,'role':role,'status':'unavailable','reason':'not observed in current decode timeline'})
        for role,nvtx,symbol in roles:
            for acq in range(3):
                label=f'J3_{model}_b{batch}_c{context}_{role}_acq{acq:02d}'
                target=ROOT/'profiles'/(label+'.json')
                if target.exists():results.append(load(target));continue
                config=profile_config(model,'eager',batch,context,label)
                output=ROOT/'runs'/label;report=ROOT/'profiles'/(label+'.ncu-rep')
                command=[NCU,'--replay-mode','kernel','--profile-from-start','off',
                         '--clock-control','none','--cache-control','none','--nvtx','--nvtx-include',nvtx,
                         '--launch-count','1','--kernel-name-base','demangled',
                         '--section','LaunchStats','--section','Occupancy','--metrics',','.join(requested),
                         '--export',str(report)]
                if symbol:command+=['--kernel-name','regex:^'+re.escape(symbol)+'$']
                command+=[str(PYTHON),str(PROGRAMS/'shoreline_decode.py'),'--config',str(config),
                          '--gpu-uuid',UUID,'--mode','timing','--output',str(output),'--external-profiler']
                outcome=attempt(label,command,timeout=600)
                if report.exists():
                    attempt(label+'_import',[NCU,'--import',str(report),'--page','raw','--print-units','base','--csv'],gpu=False,timeout=120)
                    attempt(label+'_sass',[NCU,'--import',str(report),'--page','source','--print-source','sass'],gpu=False,timeout=120)
                raw=ROOT/'logs'/(label+'_import.log');rows=records(raw) if raw.exists() else []
                symbols_observed=sorted(set(r.get('Kernel Name','') for r in rows))
                exact=False
                if (output/'run.json').exists():
                    d=load(output/'run.json');expected=load(ref)['cases'][0]['trials'][0]['output_ids']
                    exact=d['status']=='complete' and all(t['output_ids']==expected for c in d['cases'] for t in c['trials'])
                role_ok=(bool(symbols_observed) and (symbol is None or symbols_observed==[symbol]))
                if role in ('qkv','gate_up','down'):
                    role_ok=role_ok and not any('flash' in s.lower() or 'attn' in s.lower() or 'combine' in s.lower() for s in symbols_observed)
                result={'id':label,'status':'complete' if outcome['status']=='complete' and rows and exact and role_ok else 'failed',
                    'model':model,'batch':batch,'context':context,'engine':'eager','layer':mid,'role':role,
                    'acquisition_id':acq,'actual_nvtx_filter':nvtx,'symbols':symbols_observed,'exact_reference_ids':exact,
                    'metrics':rows,'unavailable_metrics':unavailable,'cache_control':'none','clock_control':'none',
                    'replay_mode':'kernel','classification':'measured-real-GPU-instrumented',
                    'scope':'one observed operation/layer launch at early decode; kernel replay changes cache state',
                    'dynamic_tensor_work':None,'dynamic_work_reason':'static tile names and occupancy do not establish dynamic issued work',
                    'full_step_totals':None,'full_step_reason':'bounded kernel samples are not whole-model traffic'}
                sass=ROOT/'logs'/(label+'_sass.log')
                result['loaded_source_export']={'path':str(sass.relative_to(ROOT)),'sha256':digest(sass),
                    'semantic_validation':'pending instruction/async coverage review; text export alone is not dynamic work'} if sass.exists() else None
                dump(target,result);results.append(result)
                publish(f'J3 acquisition checkpoint: {label} {result["status"]}')
                if model=='bridge' and batch==1 and context==1024 and role=='qkv' and acq==0 and result['status']!='complete':
                    dump(ROOT/'results/shoreline_kernel_qualification.json',{'cases':results,'pilot_passed':False})
                    disposition('J3','partial','bounded operation-range producer/reader pilot failed; preserve failure and diagnose before expansion')
                    return
    dump(ROOT/'results/shoreline_kernel_qualification.json',{'cases':results})
    disposition('J3','partial','operation-verified kernel evidence published; dynamic SASS/work and unavailable paths remain explicit')


def restricted_case(model,engine,batch,context,fraction,acq,pilot=False,attempt_id=0):
    label=f'J5_{model}_{engine}_b{batch}_c{context}_f{int(fraction*100)}_acq{acq:02d}'+(f'_pilot{attempt_id}' if pilot else '')
    target=ROOT/'profiles'/(label+'.json')
    if target.exists():return load(target)
    config=profile_config(model,engine,batch,context,label) if pilot else ROOT/'configs'/f'{model}_{engine}_b{batch}_c{context}_t10.json'
    output=ROOT/'runs'/label
    cmd=python_command('shoreline_restricted_decode.py','--config',config,'--gpu-uuid',UUID,
                       '--output',output,'--fraction',str(fraction),'--mode','timing')
    report=ROOT/'profiles'/label
    if pilot:
        cmd=[str(CUDA/'bin/nsys'),'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--cuda-graph-trace=node',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(report)]+cmd+['--external-profiler']
    outcome=attempt(label,cmd,timeout=600)
    result={'id':label,'status':'failed','model':model,'batch':batch,'context':context,'engine':engine,
            'requested_fraction':fraction,'acquisition_id':acq,'instrumented_pilot':pilot}
    allocation=output/'green_allocation.json';native=output/'run.json'
    if allocation.exists():
        result['allocation']=load(allocation)
        result['actual_sm_case_id']=label+'_sm'+str(result['allocation']['allocation']['actual_sm_count'])
    ref=reference(model,engine,batch,context)
    if native.exists() and ref:
        d=load(native);expected=load(ref)['cases'][0]['trials'][0]['output_ids']
        result['exact_reference_ids']=d['status']=='complete' and all(t['output_ids']==expected for c in d['cases'] for t in c['trials'])
    if pilot and report.with_suffix('.nsys-rep').exists():
        db=report.with_suffix('.sqlite')
        if not db.exists():attempt(label+'_export',[str(CUDA/'bin/nsys'),'export','--type','sqlite','--output',str(db),str(report.with_suffix('.nsys-rep'))],gpu=False,timeout=120)
        if db.exists() and allocation.exists():
            from shoreline_timeline import read
            try:
                result['timeline']=read(db,load(allocation)['allocation']['green_context_id'])
                graph_proof=engine!='graph' or all(w['graph_kernel_count']>0 for w in result['timeline']['windows'])
                result['status']='complete' if outcome['status']=='complete' and result.get('exact_reference_ids') and result['timeline']['restriction_passed'] and graph_proof else 'failed'
            except Exception as error:result['reason']=str(error)
    elif not pilot and outcome['status']=='complete' and result.get('exact_reference_ids'):
        result['status']='complete'
        result['clock_qualification']=campaign.clock_qualification(load(native),outcome['monitor']['source'])
        if not result['clock_qualification']['passed']:result['status']='rejected'
    dump(target,result);publish(f'J5 checkpoint: {label} {result["status"]}')
    return result


def sm_allocations():
    campaign.update_job(ROOT,'J5','in_progress')
    results=[];qualified={}
    for b,c in [(1,1024),(8,4096)]:
        for engine in ('graph','eager'):
            for attempt_id in range(2):
                result=restricted_case('bridge',engine,b,c,.75,0,True,attempt_id)
                results.append(result)
                if result['status']=='complete':break
            qualified[(engine,b,c)]=result['status']=='complete'
    counter_proofs={}
    for (engine,b,c),supported in qualified.items():
        if not supported:continue
        for fraction in (.75,.5,.3):
            # Each resource setting needs its own all-kernel proof; a 75% pilot does not prove 30%.
            proof=restricted_case('bridge',engine,b,c,fraction,0,True)
            results.append(proof)
            if proof['status']!='complete':continue
            counter_proofs[(b,c,fraction)]=engine if engine=='graph' else counter_proofs.get((b,c,fraction),engine)
            for acq in range(3):results.append(restricted_case('bridge',engine,b,c,fraction,acq))
    for (b,c,fraction),engine in counter_proofs.items():
        for acq in range(3):results.append(counter_case('bridge',engine,b,c,acq,job='J5',fraction=fraction))
    for fraction in (.75,.5,.3):
        for acq in range(3):
            label=f'J5_native_f{int(fraction*100)}_acq{acq:02d}';out=ROOT/'runs'/label
            outcome=attempt(label,python_command('shoreline_native.py','--gpu-uuid',UUID,'--output',out,
                         '--acquisition',str(acq),'--fraction',str(fraction)),timeout=900)
            results.append({'id':label,'status':outcome['status'],'family':'independent service under allocation'})
            publish(f'J5 service checkpoint: {label} {outcome["status"]}')
        native_counter_suite('J5',fraction,load(ROOT/'session.json').get('native_sm_counter_conditions',
                             ['stream_512mib','gather_512mib','pointer_96mib_chains1']))
    dump(ROOT/'results/shoreline_sm_allocation_qualification.json',{'cases':results,
         'interpretation':'software SM restriction removes arithmetic, issue, LSU and MLP together; it creates no SRAM or physical area'})
    disposition('J5','partial','publish actual SM allocations, all-kernel restriction proofs and service sweeps; CI/traffic/unsupported gates remain explicit')


def treatment_case(job,model,batch,context,acq,pressure='normal',events=False):
    event_mode='complete' if events is True else 'empty' if events=='empty' else 'none'
    label=f'{job}_{model}_eager_b{batch}_c{context}_{pressure}_events{event_mode}_acq{acq:02d}'
    target=ROOT/'profiles'/(label+'.json')
    if target.exists():return load(target)
    base=campaign.make_config(ROOT,model,'eager',batch,context)
    config=load(base);config['shoreline_control']={'pressure':pressure,'complete_layer_events':events,
        'pressure_buffer_bytes':256*1024**2,'pressure_ticks':[4,5,6],
        'observation_windows':[[4,5,6],[31,32,33],[60,61,62]],
        'window_scope':'same hooks and event markers across windows; separately instrumented stationarity diagnostic',
        'allocation':'both treatments allocate after engine KV pool creation and before warmup'}
    path=ROOT/'configs'/(label+'.json');dump(path,config)
    out=ROOT/'runs'/label
    cmd=python_command('shoreline_decode.py','--config',path,'--gpu-uuid',UUID,'--output',out,
                        '--mode','timing','--pressure',pressure,'--ordinary-control')
    if events is True:cmd+=['--layer-events']
    elif events=='empty':cmd+=['--empty-layer-events']
    outcome=attempt(label,cmd,timeout=600)
    result={'id':label,'status':'failed','pressure':pressure,'layer_events':events,'model':model,
            'batch':batch,'context':context,'acquisition_id':acq,
            'timing_scope':'matched treatment includes full decode and direct intervention/callback/event costs'}
    if (out/'run.json').exists():
        d=load(out/'run.json')
        if outcome['status']=='complete' and d['status']=='complete':
            ref=reference(model,'eager',batch,context)
            if ref is None:
                candidate=ROOT/'runs'/f'{job}_{model}_eager_b{batch}_c{context}_normal_eventsnone_acq00/run.json'
                ref=candidate if candidate.exists() else None
            expected=load(ref)['cases'][0]['trials'][0]['output_ids'] if ref else None
            exact=expected is not None and all(t['output_ids']==expected for c in d['cases'] for t in c['trials'])
            cq=campaign.clock_qualification(d,outcome['monitor']['source'])
            result.update(status='complete' if cq['passed'] and exact else 'rejected',clock_qualification=cq,exact_reference_ids=exact,
                          output_ids=d['cases'][0]['trials'][0]['output_ids'],
                          step_ms=[t['wall_ms']/t['decode_iterations'] for t in d['cases'][0]['trials']],
                          run_source=str((out/'run.json').relative_to(ROOT)),run_sha256=digest(out/'run.json'))
    dump(target,result);publish(f'{job} acquisition checkpoint: {label} {result["status"]}')
    return result


def retention():
    campaign.update_job(ROOT,'J7','in_progress')
    results=[]
    for b,c in [(1,1024),(1,2048),(1,4096),(8,4096)]:
        for acq in range(3):
            # Predetermined alternating order; never choose by observed latency.
            for pressure in (('normal','pressure') if acq%2==0 else ('pressure','normal')):
                results.append(treatment_case('J7','bridge',b,c,acq,pressure))
    physical=[]
    for b,c in [(1,1024),(1,2048),(1,4096),(8,4096)]:
        for pressure in ('normal','pressure'):
            for acq in range(3):physical.append(counter_case('bridge','eager',b,c,acq,job='J7',pressure=pressure))
    dump(ROOT/'results/shoreline_retention_qualification.json',{'cases':results,'physical':physical,
        'physical_traffic':'full decode ticks include pressure and its direct read/write cost; no logical byte subtraction',
        'interpretation':'pressure is cache-history competition; no guaranteed flush or measured larger-SRAM benefit'})
    disposition('J7','partial','matched retention-pressure timing and exact outputs saved; physical read/write attribution remains a separate gate')


def layer_events():
    campaign.update_job(ROOT,'J8','in_progress')
    results=[]
    for model,b,c in CONTRASTS:
        for acq in range(3):
            for events in (False,'empty',True):results.append(treatment_case('J8',model,b,c,acq,events=events))
    dump(ROOT/'results/shoreline_layer_timing_qualification.json',{'cases':results,
        'scope':'disjoint first/middle/last complete layers at early4/5/6, middle31/32/33 and late60/61/62; controls retain direct costs',
        'local_event_intrusion':'whole-model perturbation does not bound local layer intrusion',
        'overlap':'service-equivalent bounds require independent physical demand/service and clock joins'})
    disposition('J8','partial','complete-layer event and perturbation controls published; bounded decomposition awaits qualified service/traffic joins')


def energy():
    campaign.update_job(ROOT,'J10','in_progress')
    results=[];energy_sensor_family_allowed=True
    for model,b,c in CONTRASTS:
        if reference(model,'graph',b,c) is None:continue
        for acq in range(3):
            if not energy_sensor_family_allowed:continue
            for count in ([2048] if model=='bridge' else [1024]):
                label=f'J10_{model}_graph_b{b}_c{c}_n{count}_acq{acq:02d}'
                config=campaign.make_config(ROOT,model,'graph',b,c)
                out=ROOT/'runs'/label
                outcome=attempt(label,python_command('controlled_vllm_experiments.py','--config',config,
                      '--gpu-uuid',UUID,'--mode','energy','--output',out,'--generated-tokens',str(count)),timeout=1200)
                result={'id':label,'command_status':outcome['status'],'qualified_energy':False}
                if (out/'run.json').exists():
                    d=load(out/'run.json');trials=[t for case in d['cases'] for t in case['trials']]
                    result['raw_sensors']=[t.get('telemetry') for t in trials]
                    expected=load(reference(model,'graph',b,c))['cases'][0]['trials'][0]['output_ids']
                    prefix_match=all([row[:64] for row in t['output_ids']]==expected for t in trials)
                    result['exact_64_token_baseline_prefix']=prefix_match
                    result['clock_qualification']=campaign.clock_qualification(d,outcome['monitor']['source']) if outcome.get('monitor') and trials else {'passed':False}
                    result['qualified_energy']=outcome['status']=='complete' and d['status']=='complete' and prefix_match and len(trials)==10 and all(isinstance(t.get('telemetry',{}).get('joules_per_token'),(int,float)) for t in trials)
                results.append(result);publish(f'J10 energy checkpoint: {label} qualified={result["qualified_energy"]}')
                if model=='bridge' and b==1 and acq==0 and not result['qualified_energy']:
                    diag=attempt('J10_raw_sensor_diagnostic_v1',python_command('shoreline_energy_diagnostic.py',
                        '--gpu-uuid',UUID,'--output',ROOT/'results/J10_raw_sensor_diagnostic_v1.json'),timeout=120)
                    energy_sensor_family_allowed=False
                    publish(f'J10 raw sensor diagnostic: {diag["status"]}; unqualified energy sweep stopped, prefill guardrails continue')
        for acq in range(3):
            label=f'J10_prefill_{model}_graph_b{b}_c{c}_acq{acq:02d}'
            out=ROOT/'runs'/label;config=campaign.make_config(ROOT,model,'graph',b,c)
            outcome=attempt(label,python_command('shoreline_prefill.py','--config',config,'--gpu-uuid',UUID,
                        '--mode','timing','--output',out),timeout=600)
            results.append({'id':label,'family':'prefill guardrail','status':outcome['status']})
            publish(f'J10 prefill checkpoint: {label} {outcome["status"]}')
    dump(ROOT/'results/shoreline_energy_qualification.json',{'cases':results,
        'scope':'whole-board energy; raw counter and sampled power retained without rescaling or idle subtraction',
        'absolute_component_calibration':None})
    disposition('J10','partial' if any(not r.get('qualified_energy',r.get('status')=='complete') for r in results) else 'complete',
                'publish qualified-or-NA board energy and separate prefill/first-token guardrails')


def competition():
    campaign.update_job(ROOT,'J6','in_progress')
    probe=ROOT/'runs/J6_disjoint_probe_v1'
    outcome=attempt('J6_disjoint_probe_v1',python_command('shoreline_competition.py','--gpu-uuid',UUID,
                    '--output',probe,'--probe'),timeout=180)
    publish(f'J6 resource probe checkpoint: {outcome["status"]}')
    if outcome['status']!='complete':
        disposition('J6','unavailable','native disjoint-context allocation probe failed; owned competition not expanded');return
    results=[]
    for batch,context in [(1,1024),(8,4096)]:
        supported=[]
        for engine in ('graph','eager'):
            proof=ROOT/'profiles'/f'J5_bridge_{engine}_b{batch}_c{context}_f75_acq00_pilot0.json'
            if proof.exists() and load(proof)['status']=='complete':supported.append(engine)
        if not supported:
            results.append({'batch':batch,'context':context,'status':'unavailable',
                'reason':'no all-kernel decoder restriction proof from J5'});continue
        engine=supported[0]
        label=f'J6_concurrent_pilot_{engine}_b{batch}_c{context}'
        config=profile_config('bridge',engine,batch,context,label)
        output=ROOT/'runs'/label;report=ROOT/'profiles'/label
        cmd=[str(CUDA/'bin/nsys'),'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--cuda-graph-trace=node',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(report)]
        cmd+=python_command('shoreline_competition.py','--gpu-uuid',UUID,'--output',output,
                            '--config',config,'--treatment','memory100','--external-profiler')
        pilot=attempt(label,cmd,timeout=600)
        db=report.with_suffix('.sqlite')
        if report.with_suffix('.nsys-rep').exists() and not db.exists():
            attempt(label+'_export',[str(CUDA/'bin/nsys'),'export','--type','sqlite','--output',str(db),str(report.with_suffix('.nsys-rep'))],gpu=False,timeout=120)
        valid=False;proof={}
        if db.exists() and (output/'disjoint_allocation.json').exists():
            from shoreline_timeline import read
            timeline=read(db);allocation=load(output/'disjoint_allocation.json')
            expected=sorted([allocation['decoder']['green_context_id'],allocation['competitor']['green_context_id']])
            valid=pilot['status']=='complete' and all(w['green_context_ids']==expected and
                    w['distinct_green_context_kernel_overlap_ms']>0 for w in timeline['windows'])
            if engine=='graph':valid=valid and all(w['graph_kernel_count']>0 for w in timeline['windows'])
            proof={'timeline':timeline,'allocation':allocation,'both_contexts_observed':valid}
        dump(ROOT/'profiles'/(label+'.json'),{'status':'complete' if valid else 'failed',**proof})
        publish(f'J6 concurrency pilot checkpoint: {label} proved={valid}')
        if not valid:
            results.append({'id':label,'status':'unavailable','reason':'actual disjoint concurrent decode not proved'});continue
        for acq in range(3):
            for treatment in ('idle','memory50','memory100','arithmetic50','arithmetic100'):
                label=f'J6_{engine}_b{batch}_c{context}_{treatment}_acq{acq:02d}'
                output=ROOT/'runs'/label
                config=ROOT/'configs'/f'bridge_{engine}_b{batch}_c{context}_t10.json'
                row=attempt(label,python_command('shoreline_competition.py','--gpu-uuid',UUID,'--output',output,
                    '--config',config,'--treatment',treatment),timeout=600)
                result={'id':label,'status':row['status'],'treatment':treatment}
                if (output/'run.json').exists() and row['status']=='complete':
                    d=load(output/'run.json');expected=load(reference('bridge',engine,batch,context))['cases'][0]['trials'][0]['output_ids']
                    result['exact_reference_ids']=d['status']=='complete' and all(t['output_ids']==expected for c in d['cases'] for t in c['trials'])
                    result['clock_qualification']=campaign.clock_qualification(d,row['monitor']['source'])
                    if not result['exact_reference_ids'] or not result['clock_qualification']['passed']:result['status']='rejected'
                results.append(result)
                dump(ROOT/'profiles'/(label+'.json'),result)
                publish(f'J6 acquisition checkpoint: {label} {result["status"]}')
    dump(ROOT/'results/shoreline_competition_qualification.json',{'cases':results,
         'scope':'owned shared-service sensitivity, not added physical memory-interface bandwidth',
         'concurrent_counter_replay':'unvalidated; no physical controller-capacity change inferred'})
    disposition('J6','partial','publish owned disjoint-allocation/concurrency outcomes and service-sensitivity limits')


def traces():
    campaign.update_job(ROOT,'J11','in_progress')
    micro=attempt('J11_native_micro_v1',python_command('h200_native_micro.py'),gpu=False,timeout=600)
    publish(f'J11 native micro checkpoint: {micro["status"]}')
    if micro['status']!='complete':
        disposition('J11','partial','native micro capture/postprocess/replay failed; preserve evidence before model tracing');return
    if args.micro_only:
        publish('J11 checkpoint: early native trace/postprocess/replay pilot passed; model semantic pilot remains pending')
        return
    session=load(ROOT/'session.json');accel=Path(session['accel_root'])
    # Separate eager same-program semantic pilot before any complete-model capture.
    config=campaign.make_config(ROOT,'bridge','eager',1,128,3)
    output=ROOT/'traces/bridge_layer0_semantic_v1'
    archive=ROOT/'profiles/bridge_layer0_semantic_v1.tar'
    label='J11_bridge_layer0_semantic_v1'
    outcome=attempt(label,python_command('run_trace.py','--accel-root',accel,'--config',config,
        '--gpu-uuid',UUID,'--engine','vllm','--output',output,'--batch','1','--context','128',
        '--steps','2','--layers','model.layers.0','--budget-gib','4','--timeout-s','900',
        '--replay-smoke','--replay-timeout-s','300','--archive-path',archive),timeout=1500)
    publish(f'J11 semantic pilot checkpoint: {outcome["status"]}')
    warnings=[]
    for path in output.rglob('*.log') if output.exists() else []:
        text=path.read_text(errors='replace')
        warnings += [line for line in text.splitlines() if any(word in line.lower() for word in
                      ('unsupported','dropping','not implemented','trywait','warning'))]
    pilot={'status':outcome['status'],'semantic_warnings':warnings,'full_model_gate':'pending'}
    semantic_review=output/'semantic_review.json'
    review=load(semantic_review) if semantic_review.exists() else {'status':'pending','reason':'async descriptors, waits, masks and opcode mapping require actual trace/source review'}
    pilot['semantic_review']=review
    if outcome['status']=='complete' and not warnings and review.get('status')=='passed':
        import time,datetime
        cutoff=datetime.datetime.fromisoformat(session['launch_cutoff']).timestamp()
        if cutoff-time.time()>2400:
            full=ROOT/'traces/bridge_full_two_steps_v1';full_archive=ROOT/'profiles/bridge_full_two_steps_v1.tar'
            result=attempt('J11_bridge_full_two_steps_v1',python_command('run_trace.py',
                '--accel-root',accel,'--config',config,'--gpu-uuid',UUID,'--engine','vllm',
                '--output',full,'--batch','1','--context','128','--steps','2','--layers','full',
                '--budget-gib','10','--timeout-s','1800','--replay-smoke','--replay-timeout-s','600',
                '--archive-path',full_archive),timeout=2400)
            pilot['full_model_gate']=result['status'];publish(f'J11 full-model trace checkpoint: {result["status"]}')
        else:pilot['full_model_gate']='pending: insufficient capture/postprocess/transfer closeout budget'
    elif warnings:pilot['full_model_gate']='pending: native semantic warnings require investigation; no calibrated full-model claim'
    elif review.get('status')!='passed':pilot['full_model_gate']='pending: explicit trace semantic coverage gate has not passed'
    dump(ROOT/'results/shoreline_trace_qualification.json',pilot)
    for archive in [ROOT/'environment/native_trace_micro_v1.tar',*sorted((ROOT/'profiles').glob('bridge_*.tar'))]:
        if not archive.exists():continue
        label='J11_upload_'+archive.stem
        proof=ROOT/'results'/(label+'.json')
        outcome=attempt(label,python_command('shoreline_archive.py','--file',archive,'--proof',proof,
            '--remote-path','archives/20261016_job14100402/'+archive.name),gpu=False,timeout=1200)
        publish(f'J11 archive readback checkpoint: {label} {outcome["status"]}')
    disposition('J11','partial','native trace/parser, semantic warnings, recurrence and durable readback individually recorded; H200 timing/power calibration remains separate')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--job',choices=['J1','J2','J3','J4','J5','J6','J7','J8','J10','J11'],required=True)
    p.add_argument('--pilot',action='store_true',help='stop J2 after its bounded full-decode pilot')
    p.add_argument('--extensions',action='store_true',help='J4 read-only/overlap and physical-service extension')
    p.add_argument('--micro-only',action='store_true',help='J11 early bounded native tracer feasibility check')
    p.add_argument('--representative-only',action='store_true',help='J1 bounded saved-text functional control')
    args=p.parse_args();os.environ['H200_SESSION_CONFIG']=str(args.session.resolve())
    from h200_session import DEFAULT_ROOT as ROOT,PACKAGE,PYTHON,UUID,dump,run,python_command,CUDA,verify_allocation
    import shoreline_campaign as campaign
    for key in ('PACKAGE','PYTHON','UUID','dump','run','python_command'):
        setattr(campaign,key,globals()[key])
    verify_allocation()
    NCU=str(CUDA/'bin/ncu')
    PROGRAMS=Path(load(args.session).get('frozen_source_root',PACKAGE))/'scripts'
    CONTRASTS=[('bridge',1,1024),('bridge',8,4096),('mistral7b',1,4096),('mistral7b',8,4096),('qwen14b',1,4096)]
    {'J1':representative_text if args.representative_only else timelines,'J2':traffic,'J3':kernel_profiles,'J4':service_extensions if args.extensions else native,'J5':sm_allocations,
     'J6':competition,'J7':retention,'J8':layer_events,'J10':energy,'J11':traces}[args.job]()
