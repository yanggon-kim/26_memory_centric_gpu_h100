"""Join independent native timing and physical demand with strict source/clock gates."""
import hashlib
import json
import statistics


def analyze(root):
    from shoreline_campaign import clock_qualification
    from shoreline_statistics import hierarchical_interval
    def load(p):return json.loads(p.read_text())
    def source(p):return {'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    results=[]
    first=root/'results/shoreline_native_counter_qualification.json'
    selected=load(first).get('selection',[]) if first.exists() else []
    definitions=[(c,'J4_native_v4_acq','J4_native_counter_') for c in selected]
    definitions += [(c,'J4_service_extension_acq','J4_counter_') for c in ('read_only_512mib','read_stride17_512mib')]
    for condition,ordinary_prefix,counter_prefix in definitions:
        entry={'condition':condition,'sources':[],'clock_qualified':False,'physical_service_bytes_per_second':None,
               'classification':'independently measured native-pattern service; no automatic decoder transfer'}
        timings=[];reads=[];writes=[];clock_checks=[];hashes=[];operations=[]
        for acq in range(3):
            ordinary=root/'runs'/f'{ordinary_prefix}{acq:02d}/run.json'
            profile=root/'profiles'/f'{counter_prefix}{condition}_acq{acq:02d}.json'
            profiled_run=root/'runs'/f'{counter_prefix}{condition}_acq{acq:02d}/run.json'
            if not all(p.exists() for p in (ordinary,profile,profiled_run)):continue
            d,p,pd=load(ordinary),load(profile),load(profiled_run)
            if d['status']!='complete' or p['status']!='complete' or pd['status']!='complete':continue
            case=next(c for c in d['cases'] if c['id']==condition)
            pcase=next(c for c in pd['cases'] if c['id']==condition)
            if case.get('correctness')!='passed' or pcase.get('correctness')!='passed':continue
            hashes.extend(json.dumps(x['source_sha256'],sort_keys=True) for x in (d,pd))
            timings.append([t['gpu_event_ms']/t.get('repetitions',1) for t in case['trials']])
            reads.append(p['physical_read_bytes']);writes.append(p['physical_write_bytes'])
            operations.append(case['metadata'].get('useful_flops_per_launch',case['metadata'].get('useful_operations_per_launch')))
            for path,selected_case in ((ordinary,case),(profiled_run,pcase)):
                process=load(root/'results/processes'/(path.parent.name+'.json'))
                if process['status']!='complete' or process.get('monitor',{}).get('failure'):continue
                clock_checks.append(clock_qualification({'cases':[selected_case]},process['monitor']['source']))
            entry['sources'] += [source(p) for p in (ordinary,profile,profiled_run)]
        entry.update(acquisition_count=len(timings),raw_event_ms_acquisitions=timings,
                     physical_read_bytes_acquisitions=reads,physical_write_bytes_acquisitions=writes)
        valid=(len(timings)==3 and len(hashes)==6 and len(set(hashes))==1 and len(clock_checks)==6 and all(q['passed'] for q in clock_checks))
        pooled={}
        if valid:
            pooled={m:max(q['clocks'][m]['maximum'] for q in clock_checks)/min(q['clocks'][m]['minimum'] for q in clock_checks)
                    for m in ('sm_clock_mhz','memory_clock_mhz')}
            valid=all(v<=1.05 for v in pooled.values())
        entry.update(clock_qualified=valid,pooled_clock_ratio=pooled,
                     missing_reason=None if valid else 'three source-matched physical/ordinary acquisitions and pooled clock/isolation gate required')
        if valid:
            summary=hierarchical_interval(timings);entry['ordinary_event_ms']=summary
            # Each independent capture is joined only to its matching acquisition ID.
            rates=[[(r+w)/(t/1000) for t in ts] for r,w,ts in zip(reads,writes,timings)]
            if all(r+w>0 for r,w in zip(reads,writes)):
                entry['physical_service_bytes_per_second']=hierarchical_interval(rates)
            else:entry['physical_service_missing_reason']='warmed native condition has zero physical bytes; a positive HBM service rate is unidentified'
            if all(o is not None and o>0 for o in operations):
                entry['useful_operations_per_second']=hierarchical_interval([[o/(t/1000) for t in ts] for o,ts in zip(operations,timings)])
            entry['prediction_transfer']='not qualified for decoder: pattern/shape/local-resource match and model error must be independently validated'
        results.append(entry)
    fit=root/'results/shoreline_native_heldout_fit.json'
    output={'patterns':results,'native_held_out_fit':load(fit) if fit.exists() else None,
            'decoder_service_prediction':None,'missing_reason':'native service does not establish shape-matched decoder P/beta or physical architecture feasibility'}
    (root/'results/shoreline_independent_service_qualification.json').write_text(json.dumps(output,indent=2)+'\n')
    return output
