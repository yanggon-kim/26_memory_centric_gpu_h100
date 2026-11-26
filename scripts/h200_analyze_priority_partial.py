#!/usr/bin/env python3
"""CPU-only closeout of the preempted two-H200 priority allocation.

Preserves producer artifacts, records interrupted cases separately, and never
promotes native controls into decoder curves or clock-rejected timings.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics

from h200_priority_campaign import load, write, sha, analyze_curve, FRACTIONS, ROWS
from h200_ncu_read import parse_csv
from h200_traffic_resume import reference
from shoreline_campaign import clock_qualification
from shoreline_statistics import hierarchical_interval


def native(root):
    grouped={}
    sources=[]
    for acquisition in range(3):
        for fraction in FRACTIONS:
            label=f'native_f{round(fraction*100)}_a{acquisition}'
            path=root/'runs'/label/'run.json';data=load(path)
            process=load(root/'results/processes'/(label+'.json'))
            assert data['status']=='complete' and process['status']=='complete'
            assert not process['monitor']['failure']
            assert data['policy']['warmups']>=3 and data['policy']['trials']==10
            assert data['acquisition_id']==acquisition
            sources.append({'run':str(path),'sha256':sha(path),'process':str(root/'results/processes'/(label+'.json'))})
            count=data['allocation']['actual_sm_count']
            for case in data['cases']:
                assert case['correctness']=='passed' and len(case['trials'])==10
                assert case['stream_proof']['actual_sm_count']==count
                assert case['stream_proof']['green_context_id']==data['allocation']['green_context_id']
                assert 0<len(case['observed_smids'])<=count
                key=(case['id'],fraction)
                point=grouped.setdefault(key,{'condition':case['id'],'fraction':fraction,
                    'actual_sm_count':count,'metadata':case['metadata'],'event_ms_acquisitions':[],
                    'clocks':[],'producer_sources':[],'freeze_ids':[]})
                assert point['metadata']==case['metadata'] and point['actual_sm_count']==count
                point['event_ms_acquisitions'].append([t['gpu_event_ms']/t['repetitions'] for t in case['trials']])
                point['clocks'].append(clock_qualification({'cases':[case]},process['monitor']['source']))
                point['producer_sources'].append(data['source_sha256'])
                point['freeze_ids'].append(process['source_freeze_identity'])
    points=[]
    for point in grouped.values():
        clocks=point['clocks']
        valid=(len(point['event_ms_acquisitions'])==3 and all(q['passed'] for q in clocks)
               and len({json.dumps(x,sort_keys=True) for x in point['producer_sources']})==1
               and len(set(point['freeze_ids']))==1)
        ratios={m:max(q['clocks'][m]['maximum'] for q in clocks)/min(q['clocks'][m]['minimum'] for q in clocks)
                for m in ('sm_clock_mhz','memory_clock_mhz')} if all('clocks' in q for q in clocks) else {}
        valid=valid and len(ratios)==2 and max(ratios.values())<=1.05
        point.update(qualified_native_timing=valid,pooled_clock_ratios=ratios,
            raw_median_ms=statistics.median(statistics.median(a) for a in point['event_ms_acquisitions']),
            event_ms=hierarchical_interval(point['event_ms_acquisitions']) if valid else None,
            physical_service_bytes_per_second=None,
            scope='native pattern only; useful payload is not physical DRAM and does not establish decoder service')
        points.append(point)
    curves=[]
    for condition in sorted({p['condition'] for p in points}):
        selected=sorted((p for p in points if p['condition']==condition),key=lambda p:p['actual_sm_count'])
        clocks=[q for p in selected for q in p['clocks']]
        ratios={m:max(q['clocks'][m]['maximum'] for q in clocks)/min(q['clocks'][m]['minimum'] for q in clocks)
                for m in ('sm_clock_mhz','memory_clock_mhz')} if all('clocks' in q for q in clocks) else {}
        source_consistent=(len({json.dumps(s,sort_keys=True) for p in selected for s in p['producer_sources']})==1
                           and len({s for p in selected for s in p['freeze_ids']})==1)
        valid=len(selected)==6 and source_consistent and all(p['qualified_native_timing'] for p in selected) and len(ratios)==2 and max(ratios.values())<=1.05
        curves.append({'condition':condition,'points':selected,'qualified_across_sm_counts':valid,
                       'source_consistent_across_sm_counts':source_consistent,
                       'pooled_clock_ratios':ratios,'decoder_prediction':None})
        print('Native analyzed',condition,'qualified across counts=',valid,flush=True)
    result={'status':'complete_captures_analyzed','classification':'native single-device measurements',
            'gpu_uuid':load(root/'session.json')['gpu_uuid'],'sources':sources,'curves':curves,
            'acquisitions':3,'timed_trials_per_acquisition':10,'warmups':3,
            'qualified_curve_count':sum(c['qualified_across_sm_counts'] for c in curves),
            'physical_service':None,'physical_service_reason':'one full-device counter pilot; no3-acquisition counter joins per native point'}
    write(root/'results/native_priority_analysis.json',result)
    return result


def traffic(root):
    label='P2a_graph_b1_c4096_a0'
    original=load(root/'results'/(label+'.json'))
    process=load(root/'results/processes'/(label+'.json'))
    assert original['replays']['passed'] and process['status']=='complete'
    reference(root/'runs/graph_b1_c4096_stock_f100_a0_traffic_reference/run.json',
              root/'results/graph_b1_c4096_traffic_hf.json',load(root/'session.json')['gpu_uuid'])
    values=parse_csv((root/'logs'/(label+'_import.log')).read_text())
    assert len(values)==1
    metrics=next(iter(values.values()))
    assert metrics['dram__bytes_read.sum']>0 and metrics['dram__bytes_write.sum']>=0 and metrics['gpu__time_duration.sum']>0
    reviewed=copy.deepcopy(original)
    reviewed.update(status='single_acquisition_counter_gates_passed',ranges=values,
        reader_requalification_only=True,original_result_sha256=sha(root/'results'/(label+'.json')),
        reader_source_sha256=sha(Path(__file__).with_name('h200_ncu_read.py')),
        acquisition_count=1,required_acquisitions=3,
        physical_read_bytes_per_token=metrics['dram__bytes_read.sum']/3,
        physical_write_bytes_per_token=metrics['dram__bytes_write.sum']/3,
        achieved_clean_GB_s=None,primary_latency=None,
        limitation='allocation preempted before remaining acquisitions; pooled clock/source/clean-timing join remains unqualified')
    write(root/'results'/(label+'_reader_review.json'),reviewed)
    return reviewed


def plot_native(root,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['stream_512mib','stride17_512mib','gather_512mib','paged128_512mib']
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for axis,condition in zip(axes.flat,selected):
        curve=next(c for c in data['curves'] if c['condition']==condition)
        xs=[];ys=[]
        for point in curve['points']:
            meta=point['metadata'];useful=meta['useful_read_bytes_per_launch']+meta['useful_write_bytes_per_launch']
            xs.append(point['actual_sm_count']);ys.append(useful/point['raw_median_ms']/1e6)
            raw=[useful/statistics.median(a)/1e6 for a in point['event_ms_acquisitions']]
            axis.scatter([xs[-1]]*3,raw,color='0.65',s=14,alpha=.7)
        axis.plot(xs,ys,'o-',color='#167249' if curve['qualified_across_sm_counts'] else '#b66a13',markersize=4)
        axis.set_title(condition.replace('_512mib','')+' — '+('clock qualified' if curve['qualified_across_sm_counts'] else 'clock gate unresolved'))
        axis.set_ylabel('Useful payload GB/s')
        axis.grid(alpha=.2);axis.set_xticks(xs)
    for axis in axes[-1]:axis.set_xlabel('Driver-granted native SM count')
    fig.suptitle('H200 native512MiB controls: three acquisitions × ten trials')
    fig.text(.5,.015,'Useful payload rates are not physical HBM rates or decoder predictions. Gray dots: acquisition medians.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95))
    path=root/'analysis/native_512mib_controls.png';fig.savefig(path,dpi=170);plt.close(fig)
    return path


def main(root):
    allocation=load(root/'environment/allocated_devices.json')
    jid=allocation['slurm_job_id']
    accounting=list(csv.DictReader((root/'environment/final_slurm_accounting.psv').read_text().splitlines(),delimiter='|'))
    record=next(x for x in accounting if x['JobID']==jid)
    assert record['State']=='PREEMPTED'
    elapsed=int(record['ElapsedRaw']);gpu_hours=elapsed*len(allocation['devices'])/3600
    natives=native(root/'gpu1');physical=traffic(root/'gpu1')
    figure=plot_native(root,natives)
    records=[]
    for path in sorted((root/'gpu0/results').glob('*.json')):
        data=load(path)
        if isinstance(data,dict) and {'engine','fraction','batch','context','pilot','stock'}<=set(data):records.append(data)
    analyze_curve(root/'gpu0',records,'graph')
    stocks=[]
    for index,label in [(0,'graph_b1_c4096_stock_f100_a0_target_control'),(1,'graph_b1_c4096_stock_f100_a0_traffic_reference')]:
        data=load(root/f'gpu{index}/results'/(label+'.json'))
        stocks.append({'gpu':index,'gpu_uuid':data['gpu_uuid'],'raw_median_wall_ms_per_tick':statistics.median(data['step_ms']),
                       'trial_count':len(data['step_ms']),'qualified':data['qualified'],'clock':data['clock'],
                       'source':str(root/f'gpu{index}/results'/(label+'.json'))})
    interrupted={'status':'interrupted_by_allocation_preemption','case':'graph_b1_c4096_green_f100_a0_proof',
                 'allocation_record_present':(root/'gpu0/runs/graph_b1_c4096_green_f100_a0_proof/green_allocation.json').exists(),
                 'reason':'Slurm PREEMPTED; no completed decoder restriction proof; not evidence of unsupported optimized restriction',
                 'original_current_process_sha256':sha(root/'gpu0/results/current_process.json')}
    write(root/'gpu0/results/interrupted_cases.json',interrupted)
    reasons={'P0':'Inventory, exact optimized/eager references, stock decoder proof, native controls and real KV pilot saved; restricted decoder proof interrupted.',
             'P1a':'No restricted decoder count measured; full-green setup interrupted by preemption. All NVIDIA5/10/25percent thresholds unresolved.',
             'P1b':'Core SM curve unfinished; no wider-batch or long-context sweep.',
             'P2a':'One same-device full-GPU physical range requalified; two acquisitions, L2/kernel details and clock/timing joins missing.',
             'P5b':'Requires four GPUs and separate authorization.',
             'P5c':'Native service subset captured; remaining families not attempted.'}
    rows=[{'id':r,'status':'partial' if r in ('P0','P1a','P2a','P5c') else 'deferred',
           'reason':reasons.get(r,'Not attempted before allocation preemption.'),'reviewed_evidence':True} for r in ROWS]
    state={'status':'partial','termination':'PREEMPTED','slurm_job_id':jid,'elapsed_seconds':elapsed,
           'actual_allocated_gpu_hours':gpu_hours,'model_sm_curve_complete':False,
           'NVIDIA_thresholds':{'5percent':None,'10percent':None,'25percent':None},
           'native_analysis':'gpu1/results/native_priority_analysis.json',
           'physical_counter_review':'gpu1/results/P2a_graph_b1_c4096_a0_reader_review.json',
           'next_command_requires_new_allocation_authorization':True,
           'next_command':'sbatch --parsable batch/h200_priority_resume_20261126.sbatch'}
    write(root/'results/case_manifest.json',{'schema_version':1,'rows':rows})
    write(root/'STATUS.json',state)
    write(root/'results/closeout_summary.json',dict(state,stock_controls=stocks,interrupted=interrupted))
    write(root/'gpu0/STATUS.json',{'status':'partial','reason':'allocation_preempted','restriction':'unproved; interrupted'})
    write(root/'gpu1/STATUS.json',{'status':'partial','native_captures':'complete3x10','physical_counter_acquisitions':1,'termination':'PREEMPTED'})
    report=root/'analysis/report.md';archive=root/'analysis/report_during_allocation.md'
    if not archive.exists():archive.write_text(report.read_text())
    lines=['# H200 job'+jid+': preempted two-device measurement campaign','',
           '**The optimized32B B1/C4096 SM sweep did not complete. No NVIDIA5%,10% or25% SM threshold is established.**',
           '',f'Slurm preempted this Embers allocation at {record["End"]} Eastern after {elapsed} seconds: **{gpu_hours:.4f} allocated GPU-hours**, including staging and idle time. Requested wall time was8hours. See `../environment/final_slurm_accounting.psv`.',
           '', '## Decoder evidence and affected comparison','',
           'Pinned Qwen2.5-32B-Instruct revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, BF16/TP1, seed20261013. Optimized and eager C128 controls and optimized B1/C4096 controls matched independent HF dynamic-SDPA references exactly. The unrestricted target proof granted132SMs and mapped all2550 selected decoder kernel launches to the stock context;2499 were graph-node launches. This establishes stock graph execution, not restricted execution.',
           '', 'The full-green optimized proof was interrupted before an allocation record or decoder-placement result. Optimized SM restriction is **unproved**, not demonstrated broken. Restricted eager fallback was not reached. Do not pool these controls with another allocation.',
           '', '| Device | Unrestricted B1/C4096 raw median wall ms/tick | Timed trials | Clock gate |','|---|---:|---:|---|']
    for item in stocks:lines.append(f'| GPU{item["gpu"]} `{item["gpu_uuid"]}` | {item["raw_median_wall_ms_per_tick"]:.4f} | {item["trial_count"]} | {"passed within this one acquisition" if item["qualified"] else "rejected"} |')
    lines+=['','These are one-acquisition observations, not a qualified3-acquisition latency comparison. All trials are retained. The SM clock envelopes failed where marked; memory clocks and exact extrema are in `../results/closeout_summary.json`. No clock or power setting was changed. Primary J/token remains NA.','',
            '## Physical service evidence','',
            f'GPU1 captured all18 native allocations (six SM counts × three acquisitions), ten conditions each and ten timed trials per condition, with exact known-value checks. **{natives["qualified_curve_count"]} of {len(natives["curves"])} native condition curves pass the complete source/clock envelope across counts.** Native allocations are132/112/96/80/56/40SMs. See `../gpu1/results/native_priority_analysis.json`.',
            '', '![Native useful-payload service controls](native_512mib_controls.png)','',
            'The figure shows useful-payload service, not physical HBM bandwidth, architectural benefit or a decoder threshold. Same-pattern physical counters at every count and3-acquisition counter joins are still missing.','',
            'One optimized full-device B1/C4096 application range covering decode ticks4–6 passed replay/output gates. The corrected NCU2026 reader recovers **195243985664 DRAM read bytes and147042560 write bytes** for three generated tokens. The original failed legacy-reader result is preserved; see `../gpu1/results/P2a_graph_b1_c4096_a0_reader_review.json`. Only one of three acquisitions exists; clock/source/clean-timing joins and L2/kernel attribution remain unqualified. No achieved clean GB/s is inferred.','',
            '## Paired and native pilots','',
            'The synthetic128MiB peer-copy pilot passed. Real HF dynamic-SDPA KV handoff passed for64layers/128BF16 tensors at B1/C128:32MiB logical KV plus an8-byte first token, three warmups and three validation iterations, exact64-token outputs against both source continuation and independent generate. The receiver never prefills. These are functional pilots; vLLM serving connector, loaded serving, physical link bytes and primary transfer latency remain unvalidated. See `../results/P0_real_hf_kv_review.json` and `../results/P0_p2p_review.json`.','',
            '## Limitations and resume','',
            'Slurm exposed only one CUDA ordinal despite allocating two GPUs. A versioned explicit-UUID repair restored both132-SM H200s; both failed and repaired staging attempts are retained. Driver615.71.09, torch2.11.0+cu130, vLLM0.22.1 and Transformers4.57.6 were verified. The toolkit wrapper selects NCU2026.3.0 on compute and2025.3.1 on login; use a matching report reader and the validated wide/long CSV parser. Startup filesystem I/O delayed inventory. The limitations register retains each discovery and resolution.','',
            'GPU1 counter acquisitions1/2 were prepared with a reader-only continuation, but Slurm rejected that step as expired before it ran. No new job was submitted. All P2b/P3/P4 serving, contention and long-context cases remain unmeasured here. Four-GPU comparisons are deferred.','',
            'The queue disposition is `../results/case_manifest.json`. Raw artifact hashes are in `../results/checkpoint_raw_inventory.json`; raw profilers remain in persistent workspace and are excluded from public Git. Prior chronological notes are preserved in `report_during_allocation.md`.','',
            'Prepared next command, requiring authorization for another allocation:','',
            '```bash','sbatch --parsable batch/h200_priority_resume_20261126.sbatch','```','',
            'The resume launcher must obtain a fresh allocation and fresh UUID inventory. Do not reuse expired job16163464 or claim its partial curve complete.']
    report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'gpu_hours':gpu_hours,'native_qualified_curves':natives['qualified_curve_count'],
                      'stock_controls':stocks,'report':str(report),'figure':str(figure)},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root.resolve())
