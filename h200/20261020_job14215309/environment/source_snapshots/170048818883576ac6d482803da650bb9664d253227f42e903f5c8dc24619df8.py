#!/usr/bin/env python3
"""Consolidate source-bound H200 cohorts, keeping allocations and rejected gates distinct."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

PACKAGE=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def compact_energy(root):
    raw=root/'results/shoreline_energy_qualification.json'
    compact=root/'results/energy_qualification_compact_v1.json'
    if compact.exists():return compact
    ledger=read(raw)
    for row in ledger['cases']:
        if 'raw_sensors' in row:
            row['raw_sensors']=[{k:v for k,v in sensor.items() if k!='samples'} for sensor in row['raw_sensors']]
    ledger['parent']={'path':str(raw.relative_to(PACKAGE)),'sha256':sha(raw),
        'storage':'persistent shared workspace; duplicate aggregate may be excluded from Git; per-acquisition run.json retains raw samples'}
    write(compact,ledger)
    return compact


def energy_family(root):
    from shoreline_statistics import hierarchical_interval
    ledger=read(root/'results/energy_qualification_compact_v1.json')
    groups=collections.defaultdict(list)
    for row in ledger['cases']:
        if 'qualified_energy' in row:groups[row['id'].rsplit('_acq',1)[0]].append(row)
    output=[]
    for name,rows in sorted(groups.items()):
        sources=[];samples=[];digests=set();identities=set();clocks=[];valid=len(rows)==3
        for row in sorted(rows,key=lambda r:r['id']):
            run=root/'runs'/row['id']/'run.json';process=root/'results/processes'/(row['id']+'.json')
            data=read(run);producer=read(process);trials=data['cases'][0]['trials']
            valid=valid and row['qualified_energy'] and producer['status']=='complete' and not producer.get('monitor',{}).get('failure')
            valid=valid and len(trials)==10 and data['cases'][0]['output_digests_identical']
            valid=valid and all(len(ids)==data['cases'][0]['generated_tokens'] for t in trials for ids in t['output_ids'])
            digests.update(t['output_sha256'] for t in trials)
            identities.add(producer.get('source_freeze_identity'))
            clocks.append(row['clock_qualification'])
            samples.append([t['telemetry']['joules_per_token'] for t in trials])
            sources.extend([{'path':str(p.relative_to(PACKAGE)),'sha256':sha(p)} for p in (run,process)])
        valid=valid and len(digests)==1 and len(identities)==1 and None not in identities
        pooled={}
        if valid:
            pooled={m:max(q['clocks'][m]['maximum'] for q in clocks)/min(q['clocks'][m]['minimum'] for q in clocks)
                    for m in ('sm_clock_mhz','memory_clock_mhz')}
            valid=all(v<=1.05 for v in pooled.values())
        output.append({'id':name,'qualified':bool(valid),'acquisitions':len(rows),'trials_per_acquisition':[len(a) for a in samples],
            'joules_per_timed_token':hierarchical_interval(samples) if valid else None,'pooled_clock_ratio':pooled,
            'source_freeze_identities':sorted(str(x) for x in identities),'sources':sources,
            'scope':'whole-board decode; raw energy counter/sample estimators retained in source; no idle subtraction or external absolute calibration',
            'missing_reason':None if valid else 'three complete exact-output/source/isolation/sensor-qualified 10-trial acquisitions and pooled <=5% clocks required'})
    return output


def main(destination,repair_root=None):
    from shoreline_treatment_analysis import analyze as treatment_analysis
    from shoreline_timeline import read as timeline,competition_gate
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    sources=[]
    def artifact(name):
        path=PACKAGE/name;value=read(path)
        sources.append({'path':name,'sha256':sha(path)})
        return value
    cohorts={
        'engine':artifact('h200/20261017_job14133085/results/J1_b1_engine_comparison_v1.json'),
        'traffic':artifact('h200/20261017_job14131987/results/shoreline_traffic_accounting.json'),
        'native':artifact('h200/20261019_job14171804/results/J5_native_service_review_v1.json'),
        'sm':artifact('h200/20261018_job14150903/results/J5_partial_completed_v1/shoreline_sm_statistics.json'),
        'fit':artifact('h200/20261017_job14133085/results/J4_completed_v1/shoreline_independent_service_qualification.json'),
        'trace':artifact('h200/20261017_job14133085/results/J11_semantic_review_v1.json'),
        'archive':artifact('h200/20261019_job14180351/results/cpu_archive_00_v1_readback.json')}
    treatment_root=PACKAGE/'h200/20261019_job14177052'
    treatment_analysis(treatment_root)
    treatments=artifact('h200/20261019_job14177052/results/shoreline_treatment_statistics.json')
    energy_root=PACKAGE/'h200/20261019_job14180351'
    compact=compact_energy(energy_root)
    artifact('h200/20261019_job14180351/results/energy_qualification_compact_v1.json')
    energy=energy_family(energy_root)
    range_diagnostics=[]
    for p in sorted((treatment_root/'profiles').glob('J7*_reader_v3.json')):
        row=read(p);publication=row['native_replay_publications']
        process=treatment_root/'results/processes'/(row['id']+'.json')
        range_diagnostics.append({'source':str(p.relative_to(PACKAGE)),'sha256':sha(p),
            'original_status':row['status'],'producer_status':read(process)['status'],
            'expected_ranges':row['expected_range_count'],'observed_ranges':len(row['ranges']),
            'replay_outputs_exact':bool(publication) and all(x['passed'] for x in publication),
            'interpretation':'six outer intervals include three warmups and three measured trials; not one selected trial; original rejection retained'})
    concurrency=[]
    for shape in ('b1_c1024','b8_c4096'):
        label='J6_concurrent_pilot_eager_'+shape
        allocation=read(energy_root/'runs'/label/'disjoint_allocation.json')
        process=read(energy_root/'results/processes'/(label+'.json'))
        observed=timeline(energy_root/'profiles'/(label+'.sqlite'))
        run=read(energy_root/'runs'/label/'run.json')
        placement=competition_gate(observed,allocation,True,run['status']=='complete')
        concurrency.append({'id':label,'producer_status':process['status'],'exit_code':process.get('exit_code'),
            'native_run_status':run['status'],'instrumented_role_placement_and_overlap_observed':placement,
            'actual_overlap_ms':[w['distinct_green_context_kernel_overlap_ms'] for w in observed['windows']],
            'reason':'process failed after complete native outputs with cudaErrorContextIsDestroyed during interpreter/allocator teardown; both contexts lacked J5-style lifetime retention',
            'accepted_quantitative_measurement':False,
            'sources':[{'path':str(p.relative_to(PACKAGE)),'sha256':sha(p)} for p in
                       (energy_root/'results/processes'/(label+'.json'),energy_root/'profiles'/(label+'.json'))]})
    diagnoses={
        'J7_counter_scope':{'captures':range_diagnostics,'repair':'ticks_only_v5 emits outer counter NVTX only for selected measured trial; new pilot and captures required'},
        'J7_c4096_correctness':artifact('h200/20261019_job14177052/results/J7_functional_reference_gates.json'),
        'J6_context_lifetime':concurrency,
        'clock_rejections':{'reason':'exact outputs retained but pooled/per-trial SM clock ratio exceeds1.05; no relaxed gates, forced clocks or favorable resampling',
            'new_gpu_retry_planned':False},
        'J11_semantics':{'status':cohorts['trace']['status'],'full_model_capture_allowed':False,'new_capture_planned':False,
            'reason':'safe wait/arrival/TMA correspondence, descriptors/aliases and H200 simulator calibration missing; never prune waits'},
        'private_archive':{'status':cohorts['archive']['status'],'http_status':cohorts['archive']['http_status'],
            'reason':'configured private dataset unavailable to existing authentication (HTTP404); cannot distinguish missing repository from access denial; GPU rerun cannot repair credentials'}}
    smoke=[]
    for engine in ('hf_eager','eager'):
        q=read(treatment_root/'results/qualifications'/f'bridge_{engine}_b1_c4096_smoke.json')
        p=treatment_root/q['output']/'run.json';d=read(p);case=d['cases'][0]
        smoke.append({'engine':engine,'source':str(p.relative_to(PACKAGE)),'sha256':sha(p),
            'model_revision':d['config']['model_revision'],'prompt_sha256':case['prompt_sha256'],
            'within_engine_deterministic':bool(case['trials']) and all(
                t['output_ids']==case['trials'][0]['output_ids'] for t in case['trials']),
            'output':case['trials'][0]['output_ids']})
    differences=[{'zero_based_position':i,'hf_token':x,'vllm_token':y}
        for i,(x,y) in enumerate(zip(smoke[0]['output'][0],smoke[1]['output'][0])) if x!=y]
    diagnoses['J7_c4096_correctness'].update(sources=smoke,token_differences=differences,
        interpretation='same prompt and immutable model, each engine internally deterministic; first mismatch at generated token6. HF SDPA and vLLM FA3 are different implementations; logits were not saved, so numerical versus cache/semantic cause is unresolved. Exact gate remains closed; no timing expansion.')
    clock_rows=[]
    for job in ('J7','J8'):
        for p in sorted((treatment_root/'profiles').glob(job+'*.json')):
            row=read(p)
            if row.get('status')=='rejected':clock_rows.append({'id':row['id'],'source':str(p.relative_to(PACKAGE)),
                'sha256':sha(p),'exact_outputs':row.get('exact_reference_ids'),'clock':row.get('clock_qualification')})
    for row in read(compact)['cases']:
        if 'qualified_energy' in row and not row['qualified_energy']:
            clock_rows.append({'id':row['id'],'source':str(compact.relative_to(PACKAGE)),
                'exact_outputs':row.get('exact_64_token_baseline_prefix'),'clock':row.get('clock_qualification')})
    diagnoses['clock_rejections']['cases']=clock_rows
    auth=destination/'results/archive_access_diagnostic.json'
    if auth.exists():diagnoses['private_archive']['existing_credential_checks']=read(auth)
    accepted={
        'scope':'independent H200 allocation cohorts, BF16/TP1/seed20261013; no cross-allocation acquisition pooling',
        'cohort_namespaces':{'engine':'h200/20261017_job14133085','sm':'h200/20261018_job14150903',
            'native':'h200/20261019_job14171804','traffic':'h200/20261017_job14131987',
            'treatments':'h200/20261019_job14177052','energy':'h200/20261019_job14180351'},
        'engine_comparison':cohorts['engine'],
        'sm_comparisons':[x for x in cohorts['sm']['comparisons'] if x['qualified']],
        'native_service':cohorts['native'],
        'treatment_comparisons':[x for x in treatments['comparisons'] if x['qualified']],
        'energy_families':energy,
        'traffic_captures':cohorts['traffic']['cases'],
        'decoder_prediction':None,'physical_architecture_benefit':None,
        'model_limit':'earlier held-out native overlap fit rejected; no decoder coefficient transfer',
        'evidence_sources':sources}
    repair=None
    if repair_root is not None:
        repair_root=repair_root.resolve()
        stats=repair_root/'results/shoreline_treatment_statistics.json'
        counter_paths=sorted((repair_root/'profiles').glob('J7*_reader_v3.json'))
        repair={'session':str(repair_root.relative_to(PACKAGE)),
            'treatment_comparisons':read(stats)['comparisons'] if stats.exists() else [],
            'physical_counter_captures':[{'path':str(p.relative_to(PACKAGE)),'sha256':sha(p),
                'status':read(p)['status'],'profile_protocol':read(p).get('profile_protocol'),
                'hbm_read_bytes':read(p).get('hbm_read_bytes'),'hbm_write_bytes':read(p).get('hbm_write_bytes')} for p in counter_paths],
            'interpretation':'repair cohort remains separate from earlier timing/service acquisitions; counter replay can perturb cache history and does not establish a cross-GPU timing join'}
        accepted['repair_cohort']=repair
        if (repair_root/'results/shoreline_energy_qualification.json').exists():
            compact_energy(repair_root)
            repair['energy_families']=energy_family(repair_root)
        if read(repair_root/'session.json').get('decode_conditioning'):
            repair['decode_conditioning']=read(repair_root/'session.json')['decode_conditioning']
            repair['interpretation']+=' Explicit post-prefill memory conditioning changes cache history and adds untimed TTFT; these are conditioned-decode results, not repaired original unconditioned latency.'
        retained=[]
        for name in read(repair_root/'session.json').get('retained_repair_sessions',[]):
            prior=(PACKAGE/name).resolve()
            assert prior.is_relative_to(PACKAGE/'h200') and prior!=repair_root
            evidence=prior/'analysis/consolidated/results/qualified_evidence.json'
            cohort=read(evidence)['repair_cohort']
            retained.append({'source':str(evidence.relative_to(PACKAGE)),
                'sha256':sha(evidence),'cohort':cohort})
        accepted['retained_repair_cohorts']=retained
    write(destination/'results/diagnoses.json',diagnoses)
    write(destination/'results/qualified_evidence.json',accepted)
    dispositions=[
        ('complete','allocation-specific inventories and runtime guards passed'),
        ('partial','qualified bridge B1 graph/eager comparison; larger-clock gates remain rejected'),
        ('partial','selected qualified physical traffic; no address-level exactly-once claim'),
        ('partial','bounded operation/resource evidence; complete issued/productive work unknown'),
        ('partial','independent native service qualified; held-out overlap model rejected'),
        ('partial','eager B1 SM comparisons and native curves qualified; graph restriction unsupported'),
        ('needs_gpu_repair','corrected disjoint context lifetime requires clean-exit pilot'),
        ('needs_gpu_repair','two qualified pressure-timing comparisons; single-range counter repair pending'),
        ('partial','qualified bridge B1 whole-model perturbation controls; local intrusion and larger clocks pending'),
        ('analytical_only','conditional multipliers; no validated decoder performance/physical-feasibility prediction'),
        ('partial','one three-acquisition energy family qualified; other clocks rejected'),
        ('blocked','Hopper async/wait semantics and private archive access unresolved')]
    jobs=[{'id':f'J{i}','scientific_gate':gate,'disposition':why} for i,(gate,why) in enumerate(dispositions)]
    if repair:
        for job in jobs:
            comparisons=repair['treatment_comparisons']+[x for p in accepted.get('retained_repair_cohorts',[])
                for x in p['cohort']['treatment_comparisons']]
            if job['id']=='J6' and any(x['job']=='J6' and x['qualified'] for x in comparisons):
                job.update(scientific_gate='partial',disposition='corrected cohort has qualified fixed-partition comparisons; see per-treatment gates and shared-service limits')
            if job['id']=='J7' and len(repair['physical_counter_captures'])==18 and all(x['status']=='complete' for x in repair['physical_counter_captures']):
                job.update(scientific_gate='partial',disposition='corrected single-range counters passed; prior timing and new physical cohort remain separate')
            if job['id']=='J7' and accepted.get('retained_repair_cohorts'):
                captures=repair['physical_counter_captures']+[x for p in accepted['retained_repair_cohorts']
                    for x in p['cohort']['physical_counter_captures']]
                completed={tuple(read(PACKAGE/x['path'])[k] for k in ('batch','context','pressure','acquisition_id'))
                    for x in captures if x['status']=='complete'}
                expected={(b,c,state,acq) for b,c in ((1,1024),(1,2048),(8,4096))
                    for state in ('normal','pressure') for acq in range(3)}
                if completed==expected:
                    job.update(scientific_gate='partial',disposition='all18 individual counter captures retained across separate allocations; no pooled timing/clocks or matched cross-allocation treatment claim')
    write(destination/'STATUS.json',{'status':'consolidated_with_explicit_gaps','jobs':jobs,
        'required_gpu_repairs':[j['id'] for j in jobs if j['scientific_gate']=='needs_gpu_repair'],
        'clock_rejected_families':'retained; not automatically resampled','trace_and_archive':'blocked independently of GPU collection'})
    lines=['# Consolidated H200 evidence and gap diagnosis','',accepted['scope']+'.','',
        '## Accepted results','',
        '- Engine: bridge B1/C1024 graph/eager wall ratio0.292817; software-engine comparison only.',
        '- SM restriction: qualified eager B1/C1024 comparisons at104/72/40SMs; upper95% regression bounds1.27%/1.40%/1.18%; no iso-area inference.',
        '- Native service:12 source/clock-qualified J5 joins; streaming approximately3.97/3.97/4.01/3.71TB/s at132/104/72/40SMs. No automatic decoder transfer.',
        '- Physical traffic: Qwen14B graph B1/C4096 reads approximately1.00054× summed per-tick demand, approximately3.0016× window-unique demand across3ticks; aggregate ratios do not prove address-level exactly-once access.','',
        '| Treatment | Shape | Ratio |95%CI|','|---|---|---:|---|']
    for row in accepted['treatment_comparisons']:
        value=row['ratio_ci95'];lines.append(f'|{row["job"]} {row["numerator"]}|B{row["batch"]}/C{row["context"]}|{value["value"]:.6f}|{value["ci95"]}|')
    lines+=['','J7 pressure timing includes intervention cost; pressure is competition, not a cache flush or larger SRAM. J8 whole-model perturbation does not bound local event intrusion. No significant pressure slowdown is established for the qualified B1 controls.','',
        '## Board energy','', '| Family | Qualified | J/timed token (95% CI) |','|---|---|---|']
    for row in energy:lines.append(f'|{row["id"]}|{row["qualified"]}|{row["joules_per_timed_token"]}|')
    lines+=['','Only sensor/output/source/isolation/pooled-clock-qualified families receive a primary interval. Counter and sampled-power agreement is internal consistency, not external absolute calibration. Longer-run energy scope is separate from64-token latency.','',
        '## Diagnoses and actions','',
        '1. J7: all18 counter producers exited0 with exact replay output, but each exported6 outer ranges while the reader required1. Pressure callbacks caused warmup and other trials to emit the outer range. Keep original rejections; correct the producer and collect one bounded pilot before expansion. B1/C4096 also failed the strict HF/engine output gate and is excluded, with failure retained.',
        '2. J6: native runs reached complete output and NSYS observed both contexts and overlap, but interpreter teardown aborted with destroyed CUDA contexts. Retain both contexts/streams for process lifetime; require clean producer exit, exact outputs and correct context per kernel role before a small fixed-partition treatment family.',
        '3. J8/J10 clock failures remain rejected. No new sweep is justified merely to search for acceptable clocks. Bridge B1 energy is separately qualified after all three long-output acquisitions agree. For excluded B1/C4096, HF and vLLM have the same prompt/model and internally deterministic outputs but diverge at generated token6. Missing logits prevent a definitive numerical-versus-cache diagnosis; exact correctness stays closed.',
        '4. J11 full tracing stays blocked: preserving raw waits is mandatory, and the postprocessor/simulator semantics remain incomplete. No GPU recapture can substitute for that implementation.',
        '5. Private archive upload/readback returns HTTP404 under existing authentication. Shared raw archives and SHA manifests persist; independent private readback is unverified. Authorized dataset access or a corrected private destination is needed; no credentials are published.','',
        '## Targeted follow-up','',
        'One versioned H200 repair job will run corrected J6 pilots/treatments and J7 physical counters for already output-qualified shapes, keeping three acquisitions/ten ordinary trials and the existing correctness/clock/isolation gates. It will reuse all other job dispositions, retain failures and commit/push at acquisition boundaries. It does not repeat the completed matrix or clock-rejected families.','',
        'The architecture motivation remains compute/request-delivery co-design with workload-specific retention. Area, placeable shoreline, package, energy advantage and validated decoder speedup predictions remain unproven.']
    if repair:
        lines+=['','## Current repair/diagnostic cohort','',repair['session'],
            '', '| Comparison | Shape | Qualified | Ratio95%CI |','|---|---|---|---|']
        for row in repair['treatment_comparisons']:
            if row['job'] in ('J6','J8'):lines.append(f'|{row["job"]} {row["numerator"]}/{row["denominator"]}|B{row["batch"]}/C{row["context"]}|{row["qualified"]}|{row["ratio_ci95"]}|')
        lines+=['','Physical J7 counter dispositions: '+str(dict(collections.Counter(x['status'] for x in repair['physical_counter_captures']))),repair['interpretation']]
        for energy_row in repair.get('energy_families',[]):
            lines+=['','Current-cohort energy: '+energy_row['id']+' qualified='+str(energy_row['qualified'])+
                ', J/timed token='+str(energy_row['joules_per_timed_token'])+'.']
        for prior in accepted.get('retained_repair_cohorts',[]):
            cohort=prior['cohort']
            lines+=['','Retained allocation cohort: '+cohort['session'],
                'Source: '+prior['source']+' (SHA256 '+prior['sha256']+').',
                'Qualified J6 comparisons: '+str(sum(x['job']=='J6' and x['qualified'] for x in cohort['treatment_comparisons'])),
                'Retained physical counter dispositions: '+str(dict(collections.Counter(x['status'] for x in cohort['physical_counter_captures']))),
                'These acquisitions remain separate; no cross-allocation timing or clock pooling.']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(destination),'qualified_treatments':len(accepted['treatment_comparisons']),
        'qualified_energy_families':[x['id'] for x in energy if x['qualified']],
        'counter_range_counts':dict(collections.Counter(x['observed_ranges'] for x in range_diagnostics)),
        'concurrency_diagnoses':concurrency},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=PACKAGE/'h200/consolidated_20261019_v1')
    p.add_argument('--repair-root',type=Path)
    args=p.parse_args();main(args.output,args.repair_root)
