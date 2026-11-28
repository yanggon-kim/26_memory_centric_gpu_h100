"""Bounded acquisition of the first resident prefill comparison; CPU review follows."""
import time
import hashlib
from pathlib import Path
from h200_priority_campaign import config,load,write,sha,valid_tokens,publish,limitation
SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def capture(worker,context):
    assert context==1024,'this scheduling pilot admits only the validated1Kshape'
    assert hashlib.sha256(SOURCE_AT_IMPORT).hexdigest()==sha(worker.script/Path(__file__).name),'loaded prefill controller differs from frozen source'
    from h200_resident_campaign import build_environment
    build_environment(worker)
    budget={'estimate_s':1380.,'available_s':worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time(),
        'basis':'prior151.13second functional pilot; new owned-GPU smoke and dualHF before3x10, with conservative full-acquisition/publication margin',
        'scope':'analytical scheduling only; first-hour preemption may interrupt sooner'}
    budget['admitted']=budget['available_s']>=budget['estimate_s']+60
    write(worker.root/'results/comparison_budget.json',budget)
    if not budget['admitted']:return {'status':'deferred_budget','budget':budget}
    inventory=worker.execute('inventory',[worker.py,worker.script/'inventory.py','--gpu-uuid',worker.uuid,
        '--check-cuda','--output',worker.root/'environment/inventory.json'],timeout=180)
    assert inventory['status']=='complete'
    pilot=f'P4a_prefill_c{context}_smoke';pilot_output=worker.root/'runs'/pilot
    pilot_config=config(worker.root,'graph',context,3,1)
    proc=worker.execute(pilot,[worker.py,worker.script/'h200_prefill_resident.py','--config',pilot_config,
        '--gpu-uuid',worker.uuid,'--output',pilot_output,'--acquisition','0','--pilot'],timeout=600)
    source=pilot_output/'run.json';manifest=pilot_output/'prefill_resident.json'
    passed=proc['status']=='complete' and source.exists() and manifest.exists()
    if passed:passed=bool(valid_tokens(load(source))) and load(manifest).get('status')=='complete'
    if passed:passed=worker.reference(source,pilot)
    write(worker.root/'results/prefill_smoke.json',{'status':'passed' if passed else 'blocked',
        'source':str(source),'source_sha256':sha(source) if source.exists() else None,
        'scope':'fresh owned-GPU whole/chunk256functional and strict dualHF gate before quantitative acquisition'})
    if not passed:
        limitation(worker.root,pilot,'Fresh prefill smoke or strict reference did not pass',str(manifest),
            'Bounded600second same-GPU functional pilot before3x10capture',
            'Inspect saved failure; no baseline acquisition launched and no old-GPU qualification transferred')
        publish(worker.root,'H200 prefill: preserve blocked owned-GPU smoke');return {'status':'blocked_smoke'}
    expected=load(source)['cases'][0]['trials'][0]['output_ids']
    publish(worker.root,'H200 prefill: publish fresh owned-GPU smoke and dualHF before repeated acquisitions')
    records=[];elapsed=[]
    for acquisition in range(3):
        remaining=worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time()
        estimate=max(280,max(elapsed,default=0)*1.2)
        if remaining<(3-acquisition)*estimate+(90 if acquisition==0 else 0)+60:break
        label=f'P4a_prefill_c{context}_a{acquisition}';cfg=config(worker.root,'graph',context,10,1)
        output=worker.root/'runs'/label
        proc=worker.execute(label,[worker.py,worker.script/'h200_prefill_resident.py','--config',cfg,
            '--gpu-uuid',worker.uuid,'--output',output,'--acquisition',str(acquisition)],timeout=900)
        elapsed.append(proc.get('elapsed_s') or estimate)
        path=output/'run.json';manifest=output/'prefill_resident.json'
        passed=proc['status']=='complete' and path.exists() and manifest.exists()
        if passed:
            data=load(path);m=load(manifest)
            passed=bool(valid_tokens(data,expected)) and m.get('status')=='complete'
            passed=passed and set(m['conditions'])=={'whole','chunk256','chunk1024','chunk4096','whole_end'}
            if passed:
                for name,condition in m['conditions'].items():
                    native=load(output/name/'run.json')
                    passed=passed and len(condition['warmups'])==3 and len(condition['trials'])==10 and bool(valid_tokens(native,expected))
        row={'acquisition':acquisition,'status':'capture_passed_pending_CPU_review' if passed else 'blocked_capture_or_reference',
             'producer_status':proc['status'],'context':context,'gpu_uuid':worker.uuid,
             'source_identity':worker.settings['frozen_source_identity'],'run':str(path),
             'run_sha256':sha(path) if path.exists() else None,'manifest':str(manifest),
             'manifest_sha256':sha(manifest) if manifest.exists() else None,'process':str(worker.root/'results/processes'/(label+'.json')),
             'qualified_comparison':False,'scope':'instrumented complete prefill plus first selected token, followed by separate63decode ticks; source/output/clock/isolation/prefix coverage and cross-acquisition statistics require CPU review; no physical traffic or ordinary TTFT claim'}
        records.append(row)
        write(worker.root/'results/prefill_acquisitions.json',{'records':records,'complete_acquisition_set':len(records)==3 and all(r['status']=='capture_passed_pending_CPU_review' for r in records)})
        with (worker.root/'analysis/report.md').open('a') as stream:stream.write(f"\n## Acquisition {acquisition}\n\n{row['status']}. {row['scope']}. Evidence: `{row['manifest']}`.\n")
        if not passed:
            limitation(worker.root,label,'Prefill acquisition producer/output/reference gate failed or was incomplete',
                str(manifest),'Previously GPU-validated resident adapter; separate frozen cohort,3warmups/10trials and strict dualHFreference',
                'Preserve partial acquisition; inspect before a new identified attempt; no quantitative comparison inferred')
        publish(worker.root,f'H200 prefill context{context} acquisition{acquisition}: persist all budgets and exact-output disposition')
        if not passed:break
    return {'status':'prefill_acquisition_disposition','complete_acquisition_set':len(records)==3 and all(r['status']=='capture_passed_pending_CPU_review' for r in records),
            'records':records,'scientific_gate':'separate CPU review required; ordinary timing and physical traffic not inferred'}
