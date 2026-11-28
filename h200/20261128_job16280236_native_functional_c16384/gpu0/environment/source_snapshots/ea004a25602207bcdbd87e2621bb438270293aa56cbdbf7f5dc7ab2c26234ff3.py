"""Bounded paired diagnostic, never a qualified three-acquisition comparison.

The existing memory_settle_v3 precondition changes cache history and TTFT.
It never changes clocks, power or driver settings. Both failed and passing
clock windows are retained; this diagnostic is not a favorable-trial retry.
"""
from pathlib import Path
import hashlib
from h200_priority_campaign import config,load,write,sha,valid_tokens,publish,limitation
SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def conditioning_gate(data,conditioned,required_trials=3):
    trials=[t for case in data.get('cases',[]) for t in case.get('trials',[])]
    if len(trials)!=required_trials:return False
    if not conditioned:
        return not data.get('config',{}).get('decode_conditioning') and all(not t.get('post_prefill_conditioning') for t in trials)
    declaration=data.get('config',{}).get('decode_conditioning',{})
    if declaration!={'protocol':'memory_settle_v3','seconds':1.0,'buffer_bytes':256*1024**2,'scope':'conditioned_decode_only'}:return False
    for trial in trials:
        record=trial.get('post_prefill_conditioning') or {}
        if not (record.get('protocol')=='memory_settle_v3' and record.get('buffer_bytes')==256*1024**2
                and isinstance(record.get('copies'),int) and 0<record['copies']<=32768 and record['copies']%32==0
                and record.get('wall_ms',0)>=1000 and record.get('gpu_event_ms',0)>0):return False
    return True


def run(worker):
    assert hashlib.sha256(SOURCE_AT_IMPORT).hexdigest()==sha(worker.script/Path(__file__).name),'loaded diagnostic driver differs from frozen source'
    rows=[];expected=None
    for conditioned in (False,True):
        label='clock_pilot_'+('memory_settle_v3' if conditioned else 'unconditioned')
        base=config(worker.root,'graph',4096,3,1)
        cfg=base
        if conditioned:
            data=load(base)
            data['decode_conditioning']={'protocol':'memory_settle_v3','seconds':1.0,
                'buffer_bytes':256*1024**2,'scope':'conditioned_decode_only'}
            cfg=worker.root/'configs/clock_pilot_memory_settle_v3.json';write(cfg,data)
        output=worker.root/'runs'/label
        proc=worker.execute(label,[worker.py,worker.script/'controlled_vllm_experiments.py',
            '--config',cfg,'--gpu-uuid',worker.uuid,'--mode','timing','--output',output,
            '--generated-tokens','64'],timeout=900)
        path=output/'run.json';data=load(path) if path.exists() else {}
        good=proc['status']=='complete' and valid_tokens(data,expected)
        if good and expected is None:
            good=worker.reference(path,'clock_pilot_stock')
            if good:expected=data['cases'][0]['trials'][0]['output_ids']
        row={'conditioned':conditioned,'producer_label':label,'producer_complete':proc['status']=='complete',
             'exact_outputs':bool(good),'gpu_uuid':worker.uuid,
             'source_freeze_identity':worker.settings['frozen_source_identity'],
             'run':str(path),'run_sha256':sha(path) if path.exists() else None,
             'classification':'functional clock diagnostic only','qualified_timing_comparison':False}
        row['conditioning_scope_verified']=conditioning_gate(data,conditioned)
        good=good and row['conditioning_scope_verified']
        if good:
            from shoreline_campaign import clock_qualification
            row['clock']=clock_qualification(data,proc['monitor']['source'])
            row['post_prefill_conditioning']=[t.get('post_prefill_conditioning') for t in data['cases'][0]['trials']]
        rows.append(row)
        write(worker.root/'results/clock_conditioning_pilot.json',{'status':'partial','records':rows})
        if not good:
            limitation(worker.root,label,'Bounded clock-conditioning diagnostic failed producer/exact-output gate',
                str(path),'Single prospectively declared pair, original strict HF reference and unchanged clock rule',
                'Retain failures and continue eligible full-GPU work; no conditioned timing claim without separate qualification')
            break
    result={'status':'paired_functional_diagnostic_captured' if len(rows)==2 and all(r['exact_outputs'] and r['conditioning_scope_verified'] for r in rows) else 'blocked',
        'records':rows,'qualified_timing_comparison':False,
        'scope':'One independent producer per arm, three warmups and three timed trials; no three-acquisition latency claim. memory_settle_v3 is a fixed one-second private-buffer workload after prefill, outside decode timing; it changes cache history and TTFT. Any future conditioned SM curve needs its own placement, HF and3x10/source/clock/isolation proof and must stay separate from unconditioned results.'}
    write(worker.root/'results/clock_conditioning_pilot.json',result)
    with (worker.root/'analysis/report.md').open('a') as stream:
        stream.write('\n## Bounded clock-conditioning diagnostic\n\n'+result['scope']+'\n\nEvidence: `../results/clock_conditioning_pilot.json`. All failed windows are retained; no clock or power settings were changed. This diagnostic does not qualify clean bandwidth, eager placement or thresholds.\n')
    publish(worker.root,'H200 fixed clock-conditioning pilot: preserve paired outcomes and changed cache/TTFT scope')
    return result
