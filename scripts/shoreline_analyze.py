#!/usr/bin/env python3
"""Consolidate independently scoped measurements without filling missing gates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def load(p):return json.loads(Path(p).read_text())
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root):
    from shoreline_statistics import hierarchical_interval,paired_ratio,latency_bounds,sensitivity
    from shoreline_accounting import useful
    baseline=[]
    for model,b,c in [('bridge',1,1024),('bridge',8,4096),('mistral7b',1,4096),('mistral7b',8,4096),('qwen14b',1,4096)]:
        modes={}
        for engine in ('graph','eager'):
            acquisitions=[];sources=[];clocks=[];hashes=[]
            for index in range(3):
                q=root/'results/qualifications'/f'{model}_{engine}_b{b}_c{c}_acq{index:02d}.json'
                if not q.exists():continue
                qualification=load(q);p=root/qualification['output']/'run.json'
                if not p.exists():continue
                d=load(p)
                if d['status']!='complete' or len(d['cases'])!=1:continue
                case=d['cases'][0]
                if not case['output_digests_identical']:continue
                acquisitions.append([t['wall_ms']/t['decode_iterations'] for t in case['trials']])
                hashes.append(case['trials'][0]['output_sha256'])
                sources.append({'path':str(p.relative_to(root)),'sha256':sha(p),'qualification':qualification['status']})
                clocks.append(qualification.get('clock_qualification',{}))
            qualified=(len(acquisitions)==3 and len(set(hashes))==1 and all(x.get('passed') for x in clocks))
            pooled={k:(max(x['clocks'][k]['maximum'] for x in clocks)/min(x['clocks'][k]['minimum'] for x in clocks))
                    for k in ('sm_clock_mhz','memory_clock_mhz')} if qualified else {}
            qualified=qualified and all(v<=1.05 for v in pooled.values())
            modes[engine]={'acquisition_count':len(acquisitions),'qualified_clock_matched_comparison':qualified,
                'raw_step_ms_acquisitions':acquisitions,'sources':sources,'pooled_clock_ratio':pooled,
                'raw_median_step_ms':statistics.median([statistics.median(a) for a in acquisitions]) if acquisitions else None,
                'statistical_interval':hierarchical_interval(acquisitions) if qualified else None,
                'missing_reason':None if qualified else 'insufficient independent acquisitions, output disagreement or clock-envelope failure',
                'output_hashes':hashes}
        match=(modes['graph']['output_hashes'] and modes['graph']['output_hashes']==modes['eager']['output_hashes'])
        baseline.append({'model':model,'batch':b,'context':c,'modes':modes,'exact_mode_match':bool(match)})
    dump(root/'results/shoreline_baseline_statistics.json',{'conditions':baseline,'policy':'rejected attempts retained; no clock-rejected CI/timing join'})
    traffic=[]
    for p in sorted((root/'profiles').glob('J2_*_reader_v3.json')):
        d=load(p)
        if d['status']!='complete':continue
        cfg=load(root/'configs'/f'{d["model"]}_smoke.json')
        ref=root/'results/qualifications'/f'{d["model"]}_{d["engine"]}_b{d["batch"]}_c{d["context"]}_acq00.json'
        tokens=None
        if ref.exists():
            outputs=load(root/load(ref)['output']/'run.json')['cases'][0]['trials'][0]['output_ids']
            tokens={tick:[row[tick-1] for row in outputs] for tick in (4,5,6)}
        inputs=None
        if d['layer'] is not None:
            # Runtime input ranges are collected outside counter intervals, with value versions per tick.
            metadata_paths=[Path(load(root/pub['publication'])['output'])/'shoreline_worker_metadata.json'
                            for pub in d['native_replay_publications']]
            all_inputs=[]
            for meta in metadata_paths:
                if meta.exists():
                    all_inputs += [x for w in load(meta) for x in w.get('layer_inputs',[]) if x['layer']==d['layer']]
            candidate={}
            for row in all_inputs:
                ranges=[]
                for t in row['tensors']:
                    if t['name'] not in ('hidden_states','residual'):continue
                    if not t['contiguous']:continue
                    ranges.append((t['data_ptr'],t['data_ptr']+t['logical_tensor_bytes']))
                if ranges:
                    from shoreline_timeline import union_ns
                    candidate[row['tick']]=union_ns(ranges)
            if set(candidate)=={4,5,6}:inputs=candidate
        u=useful(cfg['model_path'],d['batch'],d['context'],d['layer'],inputs,tokens)
        traffic.append({'source':str(p.relative_to(root)),'sha256':sha(p),'id':d['id'],
            'classification':'measured physical bytes plus separately analytical logical demand',
            'D_read':d['hbm_read_bytes'],'D_write':d['hbm_write_bytes'],'logical':u,
            'D_read_over_U_tick_sum':d['hbm_read_bytes']/u['U_tick_sum'] if u['U_tick_sum'] else None,
            'D_read_over_U_window_unique':d['hbm_read_bytes']/u['U_window_unique'] if u['U_window_unique'] else None})
    dump(root/'results/shoreline_traffic_accounting.json',{'cases':traffic})
    native=[]
    for p in sorted((root/'runs').glob('J4_native_v4_acq*/run.json')):
        d=load(p)
        native.append({'source':str(p.relative_to(root)),'sha256':sha(p),'status':d['status'],
                       'allocation':d['allocation'],'cases':d['cases']})
    dump(root/'results/shoreline_service_calibration.json',{'native_acquisitions':native,
        'physical_counter_calibration':None,'held_out_prediction':None,
        'missing_reason':'scope/clock-qualified physical native service and decoder joins must pass before predictions'})
    bounds=[];scenarios=[]
    # Missing physical service calibration remains missing, even when D/T can be computed.
    for case in baseline:
        bounds.append({'model':case['model'],'batch':case['batch'],'context':case['context'],
            'C':None,'M':None,'O_interval':None,'overlap':None,'prediction':None,
            'missing_reason':'independent, scope- and clock-qualified compute/memory service join pending; decode D/T is not beta'})
        for c in (1,.75,.5,.25):
            for r in (.7,1,1.2,1.4):
                for f in (0,.05,.1,.2):
                    scenarios.append({'model':case['model'],'batch':case['batch'],'context':case['context'],
                        'c':c,'r':r,'f':f,'C_multiplier':1/c,'M_multiplier':(1-f)/r,
                        'memory_service_improves':f>1-r,'predicted_time':None,
                        'classification':'conditional analytical multiplier; no architecture gain measured',
                        'missing_reason':'qualified independent service/latency bounds pending'})
    dump(root/'results/shoreline_latency_bounds.json',{'cases':bounds,'framework':'T=C+M-X+O; coupled O/X; no exact physical wall partition'})
    dump(root/'results/shoreline_joint_sensitivity.json',{'scenarios':scenarios,'physical_feasibility':None,'component_energy':None})
    lines=['# H200 shoreline evidence progress','',
           'Ordinary, profiled, analytical and historical evidence retain separate scopes. Clock-rejected rows have raw values and no qualified matched confidence interval.','',
           '| Model | B | C | Engine | Acquisitions | Raw median step ms | Clock-qualified |',
           '|---|---:|---:|---|---:|---:|---|']
    for case in baseline:
        for engine,m in case['modes'].items():
            v=m['raw_median_step_ms'];lines.append(f'| {case["model"]} | {case["batch"]} | {case["context"]} | {engine} | {m["acquisition_count"]} | {v:.4f}'+' | '+str(m['qualified_clock_matched_comparison'])+' |' if v is not None else f'| {case["model"]} | {case["batch"]} | {case["context"]} | {engine} | 0 | NA | False |')
    lines+=['','## Physical traffic versus logical demand','',
            '| Capture | Read bytes | Write bytes | D/U tick sum | D/U window unique |','|---|---:|---:|---:|---:|']
    for d in traffic:
        lines.append(f'| {d["id"]} | {d["D_read"]:.0f} | {d["D_write"]:.0f} | {d["D_read_over_U_tick_sum"]} | {d["D_read_over_U_window_unique"]} |')
    lines+=['','A near-three window ratio can reflect ordinary autoregressive recurrence. Aggregate ratios do not prove per-address exactly-once access or identify a KV hit rate.',
            '', '## Remaining gates','',
            'Independent physical-service and clock joins, actual resource restriction/concurrency proofs, retention attribution, bounded overlap, energy qualification and trace semantics/transfer remain individually indexed in the case ledger. No physical compute/SRAM/PHY/package benefit is established.']
    (root/'analysis/shoreline_opportunity_report.md').write_text('\n'.join(lines)+'\n')
    from shoreline_measurements import emit
    flat_records=emit(root)
    print(json.dumps({'baseline_conditions':len(baseline),'physical_traffic_captures':len(traffic),
                      'native_acquisitions':len(native),'conditional_scenarios':len(scenarios),'flat_measurements':flat_records}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
