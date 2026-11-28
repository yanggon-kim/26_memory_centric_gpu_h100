#!/usr/bin/env python3
"""Report observed H200 kernel counters without inventing full-step traffic."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
from h200_session import DEFAULT_ROOT,dump
from summarize_h200 import summarize
from fa3_metadata import decode as decode_scheduler,tile as attention_tile


def number(value):
    try:return float(value.replace(',',''))
    except (ValueError,TypeError,AttributeError):return None


def role(name):
    text=name.lower()
    if 'prepare_varlen_num_blocks' in text:return 'attention_scheduler'
    if 'combine' in text:return 'attention_combine'
    if 'reshape_and_cache' in text or 'kv_cache' in text:return 'kv_update'
    if 'flash' in text or 'attention' in text:return 'attention_forward'
    if any(x in text for x in ('nvjet','gemm','gemv','cublas')):return 'linear'
    return 'other'


def analyze(root):
    root=root.resolve()
    summarize(root)
    measurements=[]
    samples=[]
    for source in sorted((root/'profiles').glob('*_ncu_*.json')):
        profile=json.loads(source.read_text())
        by_kernel=collections.defaultdict(list)
        for row in profile.get('metrics',[]):by_kernel[row['ID']].append(row)
        for kernel_id,raw in by_kernel.items():
            first=raw[0]
            sample={'id':source.stem+'_'+kernel_id,'model':profile['model'],'batch':profile['batch'],
                    'context':profile['context'],'layer':profile.get('layer'),
                    'engine':profile['engine'],'kernel':first['Kernel Name'],'role':role(first['Kernel Name']),
                    'launch_grid':first.get('Grid Size'),'launch_block':first.get('Block Size'),
                    'replay_pass_count':None,'replay_pass_count_reason':'not emitted in retained CSV; exact command and profiler log preserved',
                    'nvtx_range':next((v for k,v in first.items() if k.startswith('thread Domain:Push/Pop')),first.get('NVTX Range','NA')),
                    'metrics':{},'source':str(source.relative_to(root)),
                    'scope':profile['scope'],'cache_control':profile['cache_control'],
                    'clock_control':profile['clock_control'],'replay_mode':profile['replay_mode']}
            sample['attention_tile']=attention_tile(first['Kernel Name'])
            for row in raw:
                key=row.get('Metric Name')
                if not key or not row.get('Metric Value'):continue
                value=number(row['Metric Value'])
                reason=None
                if row['Metric Unit']=='%' and value is not None and not 0<=value<=100:
                    reason='raw percentage outside [0,100]; inconsistent replay-derived metric excluded, never clamped'
                metric={'value':None if reason else value,'unit':row['Metric Unit'],
                        'raw_value':row['Metric Value'],'missing_reason':reason,
                        'classification':'measured-real-GPU','denominator':key,
                        'section':row.get('Section Name')}
                sample['metrics'][key]=metric
                if row.get('Section Name')=='Command line profiler metrics':
                    measurements.append({'sample_id':sample['id'],'model':sample['model'],
                         'batch':sample['batch'],'context':sample['context'],'layer':sample['layer'],
                         'kernel':sample['kernel'],'kernel_role':sample['role'],'metric':key,**metric,
                         'source':sample['source'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                         'scope':profile['scope']})
            samples.append(sample)
    decode_ranges=[]
    for source in sorted((root/'profiles').glob('*_range_result.json')):
        record=json.loads(source.read_text())
        item={'source':str(source.relative_to(root)),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),**record}
        if record['status']=='complete':
            item['hbm_read_bytes_per_generated_token']=record['dram_read_bytes']/record['timed_tokens']
            item['hbm_write_bytes_per_generated_token']=record['dram_write_bytes']/record['timed_tokens']
        decode_ranges.append(item)
    dump(root/'results/fixed_bandwidth_measurements.json',{'measurements':measurements,
         'decode_ranges':decode_ranges,
         'full_step_dram_read_bytes':None,'full_step_dram_write_bytes':None,
         'full_step_missing_reason':'no DRAM total for ordinary 63-step timing trials; any bounded three-step range has a separate identity and denominator in decode_ranges',
         'rate_policy':'bytes divided by the same kernel duration are descriptive, not an independent prediction',
         'invalid_percentage_policy':'raw values retained; out-of-range percentages become null with reason'})
    dump(root/'results/kernel_counter_samples.json',{'samples':samples})
    windows=[]
    for source in sorted((root/'profiles').glob('*.sqlite')):
        try:
            with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as connection:
                ranges=connection.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='decode' AND end IS NOT NULL").fetchall()
                for start,end in ranges:
                    intervals=connection.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start',(start,end)).fetchall()
                    total=0;left=right=None
                    for a,b in intervals:
                        if left is None:left,right=a,b
                        elif a<=right:right=max(right,b)
                        else:total+=right-left;left,right=a,b
                    if left is not None:total+=right-left
                    windows.append({'source':str(source.relative_to(root)),
                        'classification':'measured-real-GPU','external_instrumentation':True,
                        'decode_wall_ms':(end-start)/1e6,'cuda_kernel_count':len(intervals),
                        'kernel_sum_ms':sum(b-a for a,b in intervals)/1e6,'kernel_union_ms':total/1e6,
                        'kernel_union_fraction_of_instrumented_wall':total/(end-start),
                        'scope':'Nsight Systems decode NVTX interval; union removes kernel overlap; GPU copies excluded; not an uninstrumented wall-time Amdahl fraction'})
        except sqlite3.DatabaseError:
            continue # A newly exported database can still be under construction.
    dump(root/'results/timeline_windows.json',{'windows':windows})
    launches=[]
    schema_path=root/'environment/fa3_scheduler_schema.json'
    schema_valid=schema_path.exists() and json.loads(schema_path.read_text()).get('status')=='passed'
    for source in sorted((root/'runs').glob('*_metadata/worker_metadata.json')):
        data=json.loads(source.read_text())
        observations=source.parent/'attention_observations.json'
        observed=json.loads(observations.read_text()) if observations.exists() else []
        split_rows=[]
        if schema_valid and data['attention'] and data['attention'][0]['head_size']==128:
            for observation in observed:
                try:
                    decoded=decode_scheduler(observation['scheduler_metadata'],len(observation['seq_lens']),observation.get('causal',True))
                    observation['decoded_scheduler']=decoded
                    observation['per_request_dynamic_splits']=decoded['per_request_dynamic_splits']
                    observation['dynamic_split_reason']=None
                    observation['causal_provenance']='sampled' if 'causal' in observation else 'pinned decoder metadata default; original capture did not explicitly sample causal'
                    split_rows.append({'module':observation['module'],'seq_lens':observation['seq_lens'],
                                       'per_request_dynamic_splits':decoded['per_request_dynamic_splits']})
                except (ValueError,TypeError):
                    observation['dynamic_split_reason']='saved metadata differs from independently validated layout; raw values retained'
        launches.append({'run_id':source.parent.name,'source':str(source.relative_to(root)),
             'attention_implementations':data['attention'],'kv_groups':data['kv_storage']['groups'],
             'actual_attention_observations':observed,
             'request_page_source':str((source.parent/'request_kv_snapshots.json').relative_to(root)),
             'request_page_caveat':'input-batch counters/table precede the next engine update; executed sequence lengths and tables come from attention observations',
             'per_request_dynamic_splits':split_rows or None,
             'dynamic_split_missing_reason':None if split_rows else 'raw scheduler metadata captured, native decoder validation pending or unsupported layout',
             'productive_empty_intervals':None,'productive_empty_missing_reason':'not inferred from persistent launch grid',
             'tile_specialization':'observed full demangled kernel templates retained in counter and timeline inventories; source dimension interpretation remains separately scoped'})
    dump(root/'results/attention_launch_inventory.json',{'runs':launches,
         'scope':'actual runtime backend/KV layout/attention lengths plus observed launch metadata; allocations are not traffic'})
    def get(sample,key):
        return sample['metrics'].get(key,{}).get('value')
    lines=['# H200 decode bottleneck analysis','','## Measured kernel evidence','',
           'Counters below are bounded kernel replays of explicitly identified programs. Percentages use the exact hardware metric denominators in the machine-readable data. These samples cannot supply full-step DRAM totals or a removable compute fraction.','',
           '| Model and shape | Layer | Work | DRAM read KiB | DRAM sustained % | L2 sector hit % | Active occupancy % |',
           '|---|---:|---|---:|---:|---:|---:|']
    selected=[]
    seen=set()
    for sample in samples:
        key=(sample['model'],sample['batch'],sample['context'],sample['layer'],sample['engine'],sample['cache_control'],sample['role'],sample['kernel'])
        if key in seen:continue
        seen.add(key);selected.append(sample)
        values=[get(sample,'dram__bytes_read.sum'),get(sample,'dram__throughput.avg.pct_of_peak_sustained_elapsed'),
                get(sample,'lts__t_sector_hit_rate.pct'),get(sample,'sm__warps_active.avg.pct_of_peak_sustained_active')]
        values[0]=values[0]/1024 if values[0] is not None else None
        lines.append('| '+ ' | '.join([f"{sample['model']} B{sample['batch']}/C{sample['context']}",
                     str(sample['layer']),sample['role']]+['NA' if v is None else f'{v:.2f}' for v in values])+' |')
    lines += ['','## Interpretation','',
              'FlashAttention 3, paged BF16 KV, and NVJET Hopper linear kernels are observed in the bridge. They differ from the historical RTX GEMV inventory. Low tensor activity alone cannot establish removable arithmetic area.','',
              'Short-context attention samples with low achieved DRAM percentage and low occupancy do not establish HBM saturation. Bandwidth, latency/MLP, launch/issue, and resource limitations must be separated by kernel and workload. Linears require their own sampled counters.','',
              'A raw L2 sector-hit percentage outside 0–100 is excluded from interpretation and retained with an invalid-value reason. Uncontrolled-cache multi-pass replay can produce inconsistent derived values; [NVIDIA documents this limitation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html). Valid-looking ratios remain replay-scoped. Separately named cold-cache controls characterize this measurement sensitivity; they are not ordinary application cache behavior.','',
              'The Nsight Systems inventories retain the sum of instrumented kernel durations and their shares. These are GPU busy-time shares, not synchronized wall-time Amdahl fractions. Graph/eager full-step comparisons use separate uninstrumented runs.','',
              'Full-step counter coverage, complete tensor-specific traffic attribution, productive/empty attention-work counts, and simulator timing/power calibration retain explicit pending gates. Dynamic split values are interpreted only after native fixed-split schema probes pass.']
    if selected:
        lines += ['', '## Kernel resources', '',
                  '| Model and shape | Engine | Cache control | Work | Grid | Registers/thread | Dynamic shared KiB/block | Local-load sectors |',
                  '|---|---|---|---|---|---:|---:|---:|']
        for sample in selected:
            resources=[get(sample,'Registers Per Thread'),get(sample,'Dynamic Shared Memory Per Block'),
                       get(sample,'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum')]
            resources[1]=resources[1]/1024 if resources[1] is not None else None
            lines.append('| '+ ' | '.join([f"{sample['model']} B{sample['batch']}/C{sample['context']}",
                 sample['engine'],sample['cache_control'],sample['role'],str(sample.get('launch_grid'))]+
                 ['NA' if v is None else f'{v:.2f}' for v in resources])+' |')
        lines += ['', 'Allocated registers/shared memory and the observed grid jointly constrain concurrency. A high register count is not evidence of spilling; actual local-memory traffic is retained. More storage does not automatically increase productive work in an unchanged underfilled or persistent grid.']
    if windows:
        lines += ['', '## Instrumented decode windows', '',
                  '| Timeline | Decode wall ms | Kernel union ms | CUDA kernels |',
                  '|---|---:|---:|---:|']
        for window in windows:
            lines.append(f"| {Path(window['source']).stem} | {window['decode_wall_ms']:.3f} | {window['kernel_union_ms']:.3f} | {window['cuda_kernel_count']} |")
        lines += ['', 'These windows include profiling overhead. Kernel unions exclude overlapping kernel time but also exclude GPU copies. The difference from wall time cannot be assigned entirely to CPU scheduling, nor transferred as an uninstrumented Amdahl fraction.']
    (root/'analysis/bottleneck_analysis.md').write_text('\n'.join(lines)+'\n')
    traffic=['# H200 decode traffic attribution','',
             f'{len(samples)} sampled kernel launches currently provide {len(measurements)} raw counter observations. Reads and writes remain separate.','',
             'Kernel roles identify linear, attention-forward, combine, and KV-update work. A linear kernel can read weights, activations, and temporary data; its DRAM count is not automatically weight-only traffic. Allocation manifests identify possible owners without proving accessed or off-chip bytes.','',
             'No full-step traffic is assigned to the ordinary 63-step timing trials. Summing sampled layers or kernels would be an estimate requiring coverage and uncertainty; this report does not present such a sum as measurement. Bounded whole-engine three-step range attempts have separate scope and output-prefix proofs.','',
             'Executed attention query and KV lengths and allocated request/page maps are saved separately. Raw input-batch counters can lag execution. Persistent KV capacity includes unused pages and is not the active KV working set.','',
             'Raw percentages outside their physical range remain preserved but are excluded from derived interpretation. No sensor or counter is rescaled to force agreement.']
    if decode_ranges:
        traffic += ['', '## Whole-engine range attempts', '',
                    '| Model and shape | Status | HBM read bytes/token | HBM write bytes/token |',
                    '|---|---|---:|---:|']
        for item in decode_ranges:
            traffic.append('| '+' | '.join([f"{item['model']} B{item['batch']}/C{item['context']}",item['status'],
                str(item.get('hbm_read_bytes_per_generated_token','NA')),str(item.get('hbm_write_bytes_per_generated_token','NA'))])+' |')
        traffic += ['', 'Ranges cover three engine.step iterations after the one-token prefill barrier, including graph execution and LM head/sampling. Replay mode is recorded per attempt. Application-range replay re-executes the native program with a separate output directory per pass; native output-prefix proofs and raw metrics are required for acceptance. Instrumented range duration is not ordinary decode latency. Failed attempts retain NA with the raw failure. [Nsight Compute documents these replay scopes and limitations](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-range-replay).']
    (root/'analysis/traffic_attribution_report.md').write_text('\n'.join(traffic)+'\n')
    residency=['# H200 on chip residency opportunity','',
               'The current Phase 1 measurements distinguish software execution paths and sample the memory/resource behavior of actual decode kernels. No added-SRAM traffic reduction has been measured.','',
               'The next discriminating evidence is a calibrated full-model access stream across successive decode steps, charged against existing 60 MiB L2 behavior. Weight/KV pinning and ordinary replacement must preserve the intervening model traffic.','',
               'Traffic reduction, higher achieved bandwidth, and changed kernel work are different hypotheses. A controlled decode improvement alone cannot establish serving p99, area parity, or architectural energy benefit. SRAM banking, data-delivery costs, retained compute/control, and leakage remain outside the measured board-energy results.']
    (root/'analysis/residency_opportunity_report.md').write_text('\n'.join(residency)+'\n')
    return {'sampled_kernels':len(samples),'counter_observations':len(measurements),'attention_metadata_runs':len(launches)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    print(json.dumps(analyze(parser.parse_args().root),indent=2))
