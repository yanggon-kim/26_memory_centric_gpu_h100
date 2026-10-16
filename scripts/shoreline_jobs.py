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
    return [r for r in csv.DictReader(io.StringIO('\n'.join(lines[i:]))) if r.get('Metric Name')]


def value(raw):
    try:return float(raw.replace(',',''))
    except (TypeError,ValueError,AttributeError):return None


def counter_case(model,engine,batch,context,acquisition,scope='full',layer=None):
    label=f'J2_{model}_{engine}_b{batch}_c{context}_{scope}_acq{acquisition:02d}'
    result_path=ROOT/'profiles'/(label+'.json')
    if result_path.exists():return load(result_path)
    ref=reference(model,engine,batch,context)
    if ref is None:
        result={'id':label,'status':'unavailable','reason':'no correct full-length baseline for exact output comparison'}
        dump(result_path,result);return result
    config=profile_config(model,engine,batch,context,label)
    base=ROOT/'runs'/label
    report=ROOT/'profiles'/(label+'.ncu-rep')
    include='shoreline_ticks_4_6/' if scope=='full' else f'layer_{layer}/'
    cmd=[NCU,'--replay-mode','app-range','--profile-from-start','off','--clock-control','none',
         '--cache-control','none','--nvtx','--nvtx-include',include,
         '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
         '--export',str(report),str(PYTHON),str(PACKAGE/'scripts/shoreline_replay.py'),
         '--config',str(config),'--gpu-uuid',UUID,'--mode','timing','--output',str(base),
         '--external-profiler','--generated-tokens','64']
    outcome=attempt(label,cmd,timeout=900)
    passed=[];problems=[]
    expected=load(ref)['cases'][0]
    from shoreline_evidence import qualify_replays
    replay_proof=qualify_replays(base,expected,ROOT)
    passed=replay_proof['passes']
    if not replay_proof['passed']:problems.append('incomplete/mismatched/unpublished application replay')
    if report.exists():
        attempt(label+'_import',[NCU,'--import',str(report),'--page','raw','--csv'],gpu=False,timeout=120)
    raw=ROOT/'logs'/(label+'_import.log')
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
            'source_config_sha256':digest(config),'replay_mode':'app-range','cache_control':'none','clock_control':'none',
            'native_replay_publications':passed,'metrics':rows,'ranges':counts,'expected_range_count':expected_ranges,
            'hbm_read_bytes':sum(v['dram__bytes_read.sum'] for v in counts.values()) if metrics_valid else None,
            'hbm_write_bytes':sum(v['dram__bytes_write.sum'] for v in counts.values()) if metrics_valid else None,
            'errors':problems,'missing_reason':None if valid else 'counter/replay/output/range qualification failed; raw attempts preserved',
            'clock_join_qualification':'pending pooled per-trial/replay endpoint checks; no timing join inferred',
            'ordinary_latency':None,'ordinary_latency_reason':'profile time is instrumented and not ordinary decode latency'}
    dump(result_path,result)
    publish(f'J2 acquisition checkpoint: {label} {result["status"]}')
    return result


def traffic():
    campaign.update_job(ROOT,'J2','in_progress')
    # First prove the whole-range producer and reader, then expand scope.
    first=counter_case('bridge','eager',1,1024,0)
    if first['status']!='complete':
        disposition('J2','partial','whole-range pilot failed; raw failure retained; further range expansion deferred pending diagnosis')
        return
    results=[first]
    for model,batch,context in CONTRASTS:
        for engine in ('eager','graph'):
            for acq in range(3):results.append(counter_case(model,engine,batch,context,acq))
        layout=load(ROOT/'configs'/f'{model}_smoke.json')['layout']
        for layer in sorted(set([0,layout['num_hidden_layers']//2,layout['num_hidden_layers']-1])):
            for acq in range(3):results.append(counter_case(model,'eager',batch,context,acq,'layer'+str(layer),layer))
    dump(ROOT/'results/shoreline_traffic_qualification.json',{'cases':results})
    disposition('J2','complete' if all(r['status']=='complete' for r in results) else 'partial',
                'publish ticks-4/5/6 physical traffic and replay qualification')


def native():
    campaign.update_job(ROOT,'J4','in_progress')
    label='J4_native_pilot'
    out=ROOT/'runs'/label
    outcome=attempt(label,python_command('shoreline_native.py','--gpu-uuid',UUID,'--output',out,
                                      '--acquisition','0','--pilot'),timeout=300)
    if outcome['status']!='complete':
        disposition('J4','partial','native service pilot failed; preserve failure before expanding')
        return
    publish('J4 checkpoint: native service producer and reference pilot passed')
    for acq in range(3):
        label=f'J4_native_acq{acq:02d}';out=ROOT/'runs'/label
        outcome=attempt(label,python_command('shoreline_native.py','--gpu-uuid',UUID,'--output',out,
                                            '--acquisition',str(acq)),timeout=900)
        publish(f'J4 acquisition checkpoint: {label} {outcome["status"]}')
    disposition('J4','partial','native service acquisitions saved; physical-counter, mixed/reduction and clock joins retain separate qualification')


def timeline_case(model,engine,batch,context,acq):
    label=f'J1_timeline_{model}_{engine}_b{batch}_c{context}_acq{acq:02d}'
    result_path=ROOT/'profiles'/(label+'.json')
    if result_path.exists():return load(result_path)
    ref=reference(model,engine,batch,context)
    if ref is None:
        result={'status':'unavailable','id':label,'reason':'correct baseline unavailable'}
        dump(result_path,result);return result
    config=profile_config(model,engine,batch,context,label)
    output=ROOT/'runs'/label;report=ROOT/'profiles'/label
    command=[str(CUDA/'bin/nsys'),'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(report),
             str(PYTHON),str(PACKAGE/'scripts/shoreline_decode.py'),'--config',str(config),
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--job',choices=['J1','J2','J4'],required=True)
    args=p.parse_args();os.environ['H200_SESSION_CONFIG']=str(args.session.resolve())
    from h200_session import DEFAULT_ROOT as ROOT,PACKAGE,PYTHON,UUID,dump,run,python_command,CUDA,verify_allocation
    import shoreline_campaign as campaign
    for key in ('PACKAGE','PYTHON','UUID','dump','run','python_command'):
        setattr(campaign,key,globals()[key])
    verify_allocation()
    NCU=str(CUDA/'bin/ncu')
    CONTRASTS=[('bridge',1,1024),('bridge',8,4096),('mistral7b',1,4096),('mistral7b',8,4096),('qwen14b',1,4096)]
    {'J1':timelines,'J2':traffic,'J4':native}[args.job]()
