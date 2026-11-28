#!/usr/bin/env python3
"""Summarize operation-verified samples and static loaded SASS, without work inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def static_opcodes(text):
    return dict(Counter(re.findall(r'^0x[0-9a-fA-F]+\s+(?:@\S+\s+)?([A-Z][A-Z0-9_.]*)\b',text,re.M)))


def analyze(root):
    def load(p):return json.loads(p.read_text())
    def source(p):return {'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    rows=[]
    names=['launch__grid_size','launch__block_size','launch__registers_per_thread',
           'launch__shared_mem_per_block_static','launch__shared_mem_per_block_dynamic',
           'launch__shared_mem_per_block_allocated','launch__shared_mem_config_size',
           'sm__warps_active.avg.pct_of_peak_sustained_active',
           'smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active',
           'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
    candidates={}
    for path in sorted((root/'profiles').glob('J3_*.json')):
        data=load(path)
        if 'role' not in data:continue
        prior=candidates.get(data['id'])
        if prior is None or '_reader_identity_v3.json' in path.name:candidates[data['id']]=path
    for path in candidates.values():
        d=load(path)
        if 'role' not in d:continue
        metrics={m['Metric Name']:{'value':m.get('Metric Value'),'unit':m.get('Metric Unit'),
            'definition':'exact Nsight metric name supplies numerator/denominator and active/elapsed scope'} for m in d.get('metrics',[])}
        process=root/'results/processes'/(d['id']+'.json')
        producer=load(process) if process.exists() else {}
        native=root/'runs'/d['id']/'run.json';config=root/'configs'/(d['id']+'.json')
        native_data=load(native) if native.exists() else {}
        config_match=config.exists() and native_data.get('config')==load(config)
        cases=native_data.get('cases',[])
        shape_match=(len(cases)==1 and cases[0]['batch_size']==d['batch'] and cases[0]['context_length']==d['context'])
        sass=root/'logs'/(d['id']+'_sass.log')
        opcodes=static_opcodes(sass.read_text()) if sass.exists() else {}
        useful=None;geometry=[]
        header=root/'environment/model_headers'/(d['model']+'_v2.json')
        if header.exists() and d['role'] in ('qkv','gate_up','down'):
            components={'qkv':['q_proj','k_proj','v_proj'],'gate_up':['gate_proj','up_proj'],'down':['down_proj']}[d['role']]
            tensors=[t for t in load(header)['safetensors_headers'] if t['name'].startswith(f'model.layers.{d["layer"]}.')
                and any('.'+part+'.weight' in t['name'] for part in components)]
            assert len(tensors)==len(components),'operator weight mapping incomplete'
            geometry=[{'name':t['name'],'shape':t['shape'],'dtype':t['dtype']} for t in tensors]
            useful=sum(2*d['batch']*t['shape'][0]*t['shape'][1] for t in tensors)
        rows.append({'id':d['id'],'status':d['status'],'model':d['model'],'batch':d['batch'],
            'context':d['context'],'role':d['role'],'layer':d['layer'],'acquisition_id':d['acquisition_id'],
            'source':source(path),'producer':source(process) if process.exists() else None,
            'reader_reprocessing':d.get('reader_reprocessing'),
            'producer_status':producer.get('status'),'source_freeze_identity':producer.get('source_freeze_identity'),
            'native_run_source':source(native) if native.exists() else None,
            'config_source':source(config) if config.exists() else None,
            'loaded_config_matches_saved_input':config_match,'native_shape_matches':shape_match,
            'isolation_failure':producer.get('monitor',{}).get('failure'),
            'exact_reference_ids':d.get('exact_reference_ids'),'symbols':d.get('symbols',[]),
            'metrics':{name:metrics.get(name,{'value':None,'missing_reason':'not present in this actual report'}) for name in names},
            'all_metric_source':source(path),'static_sass_opcodes':opcodes,
            'static_sass_instruction_rows':sum(opcodes.values()),'sass_source':source(sass) if sass.exists() else None,
            'operator_weight_geometry':geometry,'analytical_operator_flops':useful,
            'flop_scope':'complete logical linear operator; not automatically work of the first sampled kernel',
            'dynamic_issued_work':None,'productive_workers':None,
            'limitations':'Static SASS rows are not dynamic instruction counts; grid is not productive workers; kernel replay/pass metrics are not whole-model totals or a wall-time partition.'})
    result={'classification':'instrumented operation samples plus separately static/analytical evidence','cases':rows,
        'dynamic_work_calibration':None,'overlap_wall_decomposition':None}
    (root/'results/shoreline_kernel_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Operation resource and loaded-SASS evidence','',
        'Kernel replay samples retain their exact metric names/units and source/producer identities. Static SASS counts and logical operator FLOPs are separate from dynamic issued work.','',
        '| Case | Status | Static SASS rows | Logical operator FLOPs |', '|---|---|---:|---:|']
    for r in rows:lines.append(f'| {r["id"]} | {r["status"]} | {r["static_sass_instruction_rows"]} | {r["analytical_operator_flops"] if r["analytical_operator_flops"] is not None else "NA"} |')
    lines+=['','No productive/empty-worker counts, padded dynamic tensor work, whole-step traffic or exact DRAM-wait time are inferred from these samples.']
    (root/'analysis/kernel_evidence.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    print(json.dumps({'operation_samples':len(analyze(p.parse_args().root.resolve())['cases'])}))
