"""Flat, source-bound measurement export; explicit scope and missing units/rates."""
import hashlib
import json


def emit(root):
    rows=[]
    def source(path):return {'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def hardware(h):return {k:v for k,v in (h or {}).items() if k in ('uuid','sm_clock_mhz','memory_clock_mhz','temperature_c','power_w')}
    accounting=root/'results/shoreline_traffic_accounting.json'
    if accounting.exists():
        for entry in json.loads(accounting.read_text())['cases']:
            logical=entry['logical']
            values={k:logical.get(k) for k in ('U_tick_sum','U_window_unique','U_window_initial_external_main')}
            values['useful_linear_flops']=sum(t['useful_linear_flops'] for t in logical['ticks'])
            values['useful_attention_flops']=sum(t['useful_attention_flops'] for t in logical['ticks'])
            for kind,v in values.items():
                rows.append({'kind':kind,'value':v,'unit':'FLOP' if 'flops' in kind else 'byte',
                    'classification':'analytical-logical-demand','case_id':entry['id'],'source':source(accounting),
                    'geometry':logical['geometry'],'ticks':[t['tick'] for t in logical['ticks']],
                    'missing_reason':None if v is not None else 'complete logical inputs or aliases unavailable',
                    'issued_work':None,'issued_work_reason':'useful FLOPs are not dynamic issued tensor work',
                    'scope':'logical input/value accounting; never substituted for physical traffic'})
    for path in sorted((root/'profiles').glob('*_reader_v3.json')):
        d=json.loads(path.read_text())
        if d.get('status')!='complete':continue
        for metric in ('hbm_read_bytes','hbm_write_bytes'):
            rows.append({'kind':metric,'value':d[metric],'unit':'byte','classification':'measured-real-GPU-instrumented',
                'model':d['model'],'engine':d['engine'],'batch':d['batch'],'context':d['context'],
                'acquisition_id':d['acquisition_id'],'ticks':d['ticks'],'scope':d['scope'],'layer':d['layer'],
                'pressure':d.get('pressure'),'requested_sm_fraction':d.get('requested_sm_fraction'),
                'replay_mode':d['replay_mode'],'cache_control':d['cache_control'],'clock_control':d['clock_control'],
                'full_output_replay_qualified':True,'clock_join_qualified':False,
                'clock_join_missing_reason':d['clock_join_qualification'],'source':source(path)})
    for pattern in ('J3_*.json','J4_native_counter_*.json','J4_counter_*.json','J5_native_counter_*.json'):
        for path in sorted((root/'profiles').glob(pattern)):
            d=json.loads(path.read_text())
            for metric in d.get('metrics',[]):
                rows.append({'kind':'raw_hardware_metric','metric_name':metric['Metric Name'],
                    'value_raw':metric.get('Metric Value'),'unit':metric.get('Metric Unit'),
                    'classification':'measured-real-GPU-instrumented','capture_qualification':d['status'],
                    'case_id':d.get('actual_sm_case_id',d['id']),'source':source(path),
                    'scope':d['scope'],'metric_result_id':metric['ID'],
                    'wall_time_partition':None,'wall_time_missing_reason':'raw activity, stall and occupancy ratios are not exact wall-time partitions'})
    prefixes=('J4_native_v4_acq','J4_service_extension_acq','J5_native_f','J5_bridge_',
              'J7_bridge_eager_','J8_','J10_','J6_graph_','J6_eager_','J11_eager_bridge_',
              'bridge_graph_b','bridge_eager_b','mistral7b_graph_b','mistral7b_eager_b','qwen14b_graph_b','qwen14b_eager_b')
    for path in sorted((root/'runs').glob('*/run.json')):
        name=path.parent.name
        if not name.startswith(prefixes) or 'pilot' in name:continue
        d=json.loads(path.read_text())
        if d.get('status')!='complete':continue
        src=source(path)
        native='allocation' in d and name.startswith(('J4_','J5_native'))
        allocation=d.get('allocation',{})
        green=path.parent/'green_allocation.json'
        if green.exists():allocation=json.loads(green.read_text())['allocation']
        actual_sms=allocation.get('actual_sm_count',d.get('hardware',{}).get('sm_count'))
        for case in d['cases']:
            for trial_id,t in enumerate(case['trials']):
                repetitions=t.get('repetitions',1)
                event=t.get('gpu_event_ms',t.get('cuda_event_ms',t.get('gpu_ms')))
                wall=t.get('wall_ms')
                common={'case_id':case.get('id',name)+(f'_sm{actual_sms}' if actual_sms else ''),'acquisition_name':name,'trial':trial_id,
                    'source':src,'classification':'measured-real-GPU',
                    'warmup_trials':3,'timed_trials_in_acquisition':len(case['trials']),
                    'full_output_reference_qualification':'see matched profile/qualification record',
                    'clock_join_qualification':'separate pooled endpoint/interior monitor gate',
                    'hardware_before':hardware(t.get('hardware_before')),'hardware_after':hardware(t.get('hardware_after')),
                    'actual_sm_count':actual_sms,
                    'scope':'independent native pattern' if native else 'complete-model decode after one-token barrier; see source for ordinary/intervention/trace-control variant',
                    'repetitions':repetitions,'decode_ticks':t.get('decode_iterations'),
                    'aggregate_generated_tokens':t.get('generated_tokens',len(t.get('output_ids',[]))*case.get('generated_tokens',0) if 'output_ids' in t else None),
                    'timed_generated_tokens':t.get('timed_generated_tokens')}
                for kind,v in [('gpu_event_elapsed',event),('synchronized_wall_elapsed',wall)]:
                    if v is not None:rows.append({**common,'kind':kind,'value':v,'unit':'ms'})
                if native and event:
                    rows.append({**common,'kind':'native_event_per_launch','value':event/repetitions,'unit':'ms',
                        'metadata':case['metadata'],'physical_bytes':None,
                        'physical_bytes_missing_reason':'logical/useful payload is not actual HBM traffic; separate scope-matched counters required'})
                if 'prefill_and_first_token' in t:
                    rows.append({**common,'kind':'prefill_and_first_selected_token','unit':'ms',
                        'value':t['prefill_and_first_token']['gpu_event_ms'],
                        'scope':t['prefill_and_first_token']['scope']})
                for window in t.get('ordinary_decode_windows',[]):
                    rows.append({**common,'kind':'early_middle_late_decode_window','unit':'ms',
                        'classification':'measured-real-GPU-event-instrumented','value':window['gpu_event_ms'],
                        'ticks':window['ticks'],'scope':window['scope']})
                if 'telemetry' in t:
                    s=t['telemetry'];value=s.get('joules_per_token')
                    rows.append({**common,'kind':'whole_board_energy_per_timed_token','unit':'J/token',
                        'value':value if isinstance(value,(int,float)) else None,
                        'missing_reason':None if isinstance(value,(int,float)) else s.get('qualification_reason',s.get('reason','sensor qualification failed')),
                        'counter_j':s.get('counter_j'),'sampled_power_integral_j':s.get('integrated_power_j'),
                        'rescaled':False,'absolute_component_calibration':False})
        metadata=path.parent/'shoreline_worker_metadata.json'
        if metadata.exists():
            for worker in json.loads(metadata.read_text()):
                for e in worker.get('layer_events',[]):
                    if 'measured' not in (e.get('trial') or ''):continue
                    rows.append({'kind':'disjoint_layer_event','value':e['elapsed_ms'],'unit':'ms',
                        'classification':'measured-real-GPU-event-instrumented','source':source(metadata),
                        'acquisition_name':name,'layer':e['layer'],'tick':e['tick'],'trial':e['trial'],'event_mode':e['mode'],
                        'scope':'one contiguous layer invocation; local event intrusion not automatically bounded by whole-model controls'})
    out=root/'results/shoreline_measurements.jsonl'
    out.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    # Bounded shards remain reviewable in Git even after the full export grows.
    shards=out.parent/'measurement_shards';shards.mkdir(exist_ok=True)
    for index in range(0,len(rows),1000):
        (shards/f'measurements_{index//1000:04d}.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows[index:index+1000]))
    return len(rows)
