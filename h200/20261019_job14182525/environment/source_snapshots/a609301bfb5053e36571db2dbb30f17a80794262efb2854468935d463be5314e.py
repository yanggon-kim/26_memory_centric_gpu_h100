"""Matched in-process decoder SM ratios; reject unsupported or unmatched arms."""
import hashlib
import json


def analyze(root):
    from shoreline_statistics import paired_ratio
    from shoreline_treatment_analysis import provenance
    def load(p):return json.loads(p.read_text())
    def group(engine,b,c,fraction):
        rows=[]
        for acq in range(3):
            p=root/'profiles'/f'J5_bridge_{engine}_b{b}_c{c}_f{fraction}_acq{acq:02d}.json'
            if p.exists():rows.append((p,load(p)))
        return rows
    def proof(engine,b,c,fraction,sms,identity):
        for attempt in range(2):
            p=root/'profiles'/f'J5_bridge_{engine}_b{b}_c{c}_f{fraction}_acq00_pilot{attempt}.json'
            if not p.exists():continue
            d=load(p);producer=root/'results/processes'/(d['id']+'.json')
            if not producer.exists():continue
            process=load(producer)
            if (d.get('status')=='complete' and d.get('exact_reference_ids') and
                d.get('timeline',{}).get('restriction_passed') is True and
                d.get('allocation',{}).get('allocation',{}).get('actual_sm_count')==sms and
                process.get('status')=='complete' and not process.get('monitor',{}).get('failure') and
                process.get('source_freeze_identity')==identity and
                (engine!='graph' or all(w['graph_kernel_count']>0 for w in d['timeline']['windows']))):
                return {'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        return None
    comparisons=[]
    for engine in ('graph','eager'):
      for b,c in ((1,1024),(8,4096)):
        baseline=group(engine,b,c,100)
        for fraction in (75,50,30):
            treatment=group(engine,b,c,fraction);combined=baseline+treatment
            row={'engine':engine,'batch':b,'context':c,'requested_fraction':fraction/100,
                 'qualified':False,'ratio_ci95':None,'source_count':len(combined),
                 'scope':'same in-process/stream harness, whole63-tick synchronized wall per step; no compute-only/iso-area interpretation',
                 'raw_steps':{'unrestricted':[d.get('step_ms') for p,d in baseline],
                              'restricted':[d.get('step_ms') for p,d in treatment]},
                 'sources':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,d in combined]}
            identities=[provenance(root,d) for p,d in combined]
            clocks=[d.get('clock_qualification',{}) for p,d in combined]
            sms=[d.get('allocation',{}).get('allocation',{}).get('actual_sm_count') for p,d in combined]
            valid=(len(baseline)==len(treatment)==3 and all(d.get('status')=='complete' and d.get('exact_reference_ids')
                   and len(d.get('step_ms',[]))==10 and all(x>0 for x in d['step_ms']) for p,d in combined)
                   and all(identities) and len(set(identities))==1 and all(q.get('passed') for q in clocks)
                   and len({d.get('harness') for p,d in combined})==1 and all(d.get('harness') for p,d in combined)
                   and len(set(sms[:3]))==len(set(sms[3:]))==1 and all(isinstance(x,int) and x>0 for x in sms))
            if valid:
                pooled={m:max(q['clocks'][m]['maximum'] for q in clocks)/min(q['clocks'][m]['minimum'] for q in clocks)
                        for m in ('sm_clock_mhz','memory_clock_mhz')}
                primary=proof(engine,b,c,100,sms[0],identities[0]);restricted=proof(engine,b,c,fraction,sms[3],identities[0])
                valid=all(v<=1.05 for v in pooled.values()) and bool(primary and restricted) and sms[3]<sms[0]
                row.update(pooled_clock_ratio=pooled,actual_sm_count=sms[3],baseline_actual_sm_count=sms[0],
                           primary_context_proof=primary,restricted_context_proof=restricted)
            if valid:
                row.update(qualified=True,ratio_ci95=paired_ratio([d['step_ms'] for p,d in treatment],[d['step_ms'] for p,d in baseline]))
            row['missing_reason']=None if valid else 'same-harness exact3x10/source/isolation/pooled-clock and per-allocation measured-decode placement proofs required'
            comparisons.append(row)
    result={'comparisons':comparisons,'noninferiority_upper_ratio_limit':1.05,
            'interpretation':'software SM restriction also changes issue, LSU and MLP; it creates no SRAM, memory interfaces or physical area'}
    (root/'results/shoreline_sm_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
