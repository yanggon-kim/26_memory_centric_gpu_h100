"""Acquisition-matched treatment ratios; failed clock/output gates retain raw data."""
import json
import statistics


def analyze(root):
    from shoreline_statistics import paired_ratio
    def load(p):return json.loads(p.read_text())
    contrasts=[('bridge',1,1024),('bridge',8,4096),('mistral7b',1,4096),('mistral7b',8,4096),('qwen14b',1,4096)]
    def compare(job,model,b,c,left,right):
        groups={mode:[] for mode in (left,right)};sources=[];clocks=[];raw={};exact=[]
        for mode in groups:
            for acq in range(3):
                p=root/'profiles'/f'{job}_{model}_eager_b{b}_c{c}_{mode}_acq{acq:02d}.json'
                if not p.exists():continue
                d=load(p)
                if not d.get('step_ms'):continue
                groups[mode].append(d['step_ms']);sources.append(str(p.relative_to(root)))
                clocks.append(d.get('clock_qualification',{}));exact.append(d.get('exact_reference_ids',False))
            raw[mode]=[statistics.median(a) for a in groups[mode]]
        valid=(all(len(a)==3 and all(len(t)==10 for t in a) for a in groups.values()) and all(exact) and
               len(clocks)==6 and all(q.get('passed') for q in clocks))
        pooled={}
        if valid:
            pooled={m:max(q['clocks'][m]['maximum'] for q in clocks)/min(q['clocks'][m]['minimum'] for q in clocks)
                    for m in ('sm_clock_mhz','memory_clock_mhz')}
            valid=all(v<=1.05 for v in pooled.values())
        return {'job':job,'model':model,'batch':b,'context':c,'numerator':left,'denominator':right,
                'scope':'whole63-tick synchronized wall per decode step, includes intervention costs',
                'raw_acquisition_medians_ms':raw,'qualified':valid,'pooled_clock_ratio':pooled,
                'ratio_ci95':paired_ratio(groups[left],groups[right]) if valid else None,
                'missing_reason':None if valid else 'complete exact-output matched3x10 family and pooled <=5% clocks required',
                'sources':sources,'local_event_intrusion':'whole-model noninferiority does not establish a local-layer event-intrusion bound'}
    comparisons=[]
    for model,b,c in contrasts:
        for mode in ('normal_eventsempty','normal_eventscomplete'):
            comparisons.append(compare('J8',model,b,c,mode,'normal_eventsnone'))
    for b,c in [(1,1024),(1,2048),(1,4096),(8,4096)]:
        comparisons.append(compare('J7','bridge',b,c,'pressure_eventsnone','normal_eventsnone'))
    result={'comparisons':comparisons,'interpretation':'pressure is cache competition, event timing is instrumented; neither creates SRAM/PHY area'}
    (root/'results/shoreline_treatment_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
