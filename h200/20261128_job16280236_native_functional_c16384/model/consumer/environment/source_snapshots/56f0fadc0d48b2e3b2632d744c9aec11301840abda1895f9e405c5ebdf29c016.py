#!/usr/bin/env python3
"""CPU extraction of measured launch geometry from saved eager placement proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import re
from h200_priority_campaign import load,write,sha
from h200_review_allocation import case_records


FIELDS=('gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread',
        'staticSharedMemory','dynamicSharedMemory','localMemoryPerThread')


def attribute(api,ranges):
    if api is None:return 'missing_unique_CPU_API_correlation'
    matches=[r for r in ranges if r['globalTid']==api['globalTid'] and r['start']<=api['start'] and r['end']>=api['end']]
    operations=[re.fullmatch(r'operation_layer_(\d+)_(self_attn_qkv_proj|self_attn_o_proj|mlp_gate_up_proj|mlp_down_proj)',r['text'] or '') for r in matches]
    operations=[m for m in operations if m]
    ticks=[re.fullmatch(r'tick_([456])',r['text'] or '') for r in matches]
    ticks=[m.group(1) for m in ticks if m]
    if len(operations)!=1 or len(ticks)!=1:return 'unattributed_or_ambiguous_CPU_NVTX'
    operation=operations[0]
    return f'tick{ticks[0]}/layer{operation.group(1)}/{operation.group(2)}'


def signature(row):return {k:row[k] for k in ('symbol',)+FIELDS+('launches',)}


def catalog(path,green_id,range_name='shoreline_ticks_4_6'):
    groups={}
    with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        apis={}
        for api in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
            apis.setdefault(api['correlationId'],[]).append(dict(api))
        host_ranges=[dict(r) for r in db.execute('SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL')]
        ranges=db.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL',(range_name,)).fetchall()
        assert len(ranges)==1,'one bounded three-tick range required per proof'
        for start,end in ranges:
            kernels=db.execute('SELECT k.*,s.value AS symbol FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=?',(start,end)).fetchall()
            assert kernels
            for kernel in kernels:
                row=dict(kernel)
                assert (row['greenContextId'] or 0)==green_id and not row['graphNodeId']
                assert all(isinstance(row[f],int) and row[f]>0 for f in FIELDS[:6])
                key=(row['symbol'],)+(tuple(row[f] for f in FIELDS))
                group=groups.setdefault(key,{'symbol':row['symbol'],**{f:row[f] for f in FIELDS},
                    'launches':0,'instrumented_duration_ns_sum':0,'CPU_NVTX_attribution':{}})
                group['launches']+=1;group['instrumented_duration_ns_sum']+=row['end']-row['start']
                correlated=apis.get(row['correlationId'],[])
                role=attribute(correlated[0] if len(correlated)==1 else None,host_ranges)
                group['CPU_NVTX_attribution'][role]=group['CPU_NVTX_attribution'].get(role,0)+1
    ordered=[groups[key] for key in sorted(groups)]
    signatures=[signature(row) for row in ordered]
    return ordered,hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()


def main(a):
    if getattr(a,'resident',False):
        rows=[load(p) for p in sorted((a.source/'results').glob('resident_b1_c4096_a0_proof_*.json'))]
    else:rows,issues,interrupted=case_records(a.source)
    proofs=[r for r in rows if r['pilot'] and r['engine']=='eager' and r['batch']==1 and r['context']==4096]
    assert len(proofs)==7 and all(r['qualified'] and r['exact_outputs'] for r in proofs)
    assert len({r['gpu_uuid'] for r in proofs})==len({r['source_identity'] for r in proofs})==1
    assert {r['actual_sm_count'] for r in proofs}=={132,112,96,80,56,40}
    cases=[]
    for proof in sorted(proofs,key=lambda r:(r['stock'],r['actual_sm_count'])):
        range_name='shoreline_ticks_4_6'
        if getattr(a,'resident',False):
            condition=proof['id'].split('_proof_',1)[1]
            range_name=f'resident_{condition}_ticks_4_6'
            matches=[]
            for candidate in (a.source/'profiles').glob(proof['producer_label']+'.*.sqlite'):
                with sqlite3.connect('file:'+str(candidate.resolve())+'?mode=ro',uri=True) as db:
                    if db.execute('SELECT count(*) FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL',(range_name,)).fetchone()[0]:matches.append(candidate)
            assert len(matches)==1,'one saved resident capture per condition required'
            path=matches[0]
        else:path=a.source/'profiles'/(proof['id']+'.sqlite')
        groups,signature_hash=catalog(path,proof['allocation']['green_context_id'] or 0,range_name)
        cases.append({'id':proof['id'],'stock':proof['stock'],'actual_sm_count':proof['actual_sm_count'],
            'database':str(path),'database_sha256':sha(path),'source_case':str(a.source/'results'/(proof['id']+'.json')),
            'source_case_sha256':sha(a.source/'results'/(proof['id']+'.json')),
            'launch_signature_sha256':signature_hash,'launches':sum(g['launches'] for g in groups),'groups':groups})
    green=[r for r in cases if not r['stock']]
    common=set.intersection(*[{json.dumps(signature(g),sort_keys=True) for g in r['groups']} for r in green])
    variants=[{'actual_sm_count':r['actual_sm_count'],'groups':[g for g in r['groups'] if json.dumps(signature(g),sort_keys=True) not in common]} for r in green]
    result={'classification':'CPU extraction from measured-real-GPU-instrumented NSYS captures',
        'new_GPU_measurements':False,'protocol':proofs[0].get('repair_route','original'),'gpu_uuid':proofs[0]['gpu_uuid'],'source_freeze_identity':proofs[0]['source_identity'],
        'cases':cases,'all_green_launch_signatures_identical':len({r['launch_signature_sha256'] for r in green})==1,
        'varying_launch_groups':variants,
        'scope':'Single three-tick placement capture per SM setting, complete eager decoder launches. Raw grids/register/shared allocations are launch metadata, not measured occupancy, issue rate, clean timing or physical HBM traffic. CPU NVTX operation associations do not establish dynamic instruction semantics or complete attribution. No three-acquisition timing interval is derived.'}
    path=a.destination/'results/eager_decoder_grid_catalog.json';write(path,result)
    lines=['# Saved eager decoder launch geometry','',result['scope'],'',
        '| Path | Actual SMs | Captured launches | Distinct symbol/resource signatures |',
        '|---|---:|---:|---:|']
    for case in cases:lines.append(f"| {'stock' if case['stock'] else 'Green'} | {case['actual_sm_count']} | {case['launches']} | {len(case['groups'])} |")
    lines+=['', 'Exact Green launch-signature equality across all six SM settings: **'+str(result['all_green_launch_signatures_identical'])+'**.',
        '', 'The comparison includes demangled symbol, grid/block dimensions, registers/thread, static/dynamic shared memory, local-memory/thread and launch counts; durations and Green IDs are excluded from that equality check. Equality applies only to these captured ranges and settings. It does not replace an explicit kernel-grid retuning experiment.',
        '', 'Limitations: one placement acquisition per setting and instrumented durations; ordinary timing qualification is recorded separately in the workload curve. Only uniquely correlated operation ranges are attributed; remaining attention/norm/head work, achieved occupancy/issue/stalls, dynamic instruction semantics and optimized-path traffic joins remain separate. Raw launch metadata is retained with exact SQLite and case checksums in `../results/eager_decoder_grid_catalog.json`.']
    lines+=['','## Observed varying launch groups','',
        '| Actual SMs | Symbol | Grid | Block | Launches | CPU-correlated NVTX operations |',
        '|---|---|---|---|---:|---|']
    for case in variants:
        for group in case['groups']:
            roles=sorted({key.rsplit('/',1)[-1] for key in group['CPU_NVTX_attribution']})
            lines.append(f"| {case['actual_sm_count']} | `{group['symbol']}` | {group['gridX']}×{group['gridY']}×{group['gridZ']} | {group['blockX']}×{group['blockY']}×{group['blockZ']} | {group['launches']} | {', '.join(roles)} |")
    lines+=['','Attribution uses the kernel CUDA correlation ID, the unique runtime API interval on its actual host thread, and containing versioned operation/tick NVTX ranges. Kernel-name substrings are not used to assign operation semantics. Missing or ambiguous joins remain explicitly unattributed.',
        '', 'Actual CUDA cluster dimensions are not provided by this SQLite schema. Do not infer cluster semantics from symbol names. Launch differences, where observed, do not establish causality, generality or latency thresholds. Resident-model protocols may establish different library/algorithm state from fresh-model-per-point initialization; retain their full setting-specific proofs and separate labels. Unavailable operation NVTX ranges remain unattributed.']
    (a.destination/'analysis/eager_decoder_grid_catalog.md').write_text('\n'.join(lines)+'\n')
    print({'cataloged_proofs':len(cases),'identical_green_signatures':result['all_green_launch_signatures_identical'],'output':str(path)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--resident',action='store_true',help='separate resident protocol proof files/ranges')
    main(p.parse_args())
