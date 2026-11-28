"""Read-only prospective connector capture review; no clean timing inference."""
import argparse
import json
import sqlite3
from pathlib import Path

from h200_priority_campaign import load,sha
from h200_review_native_model import verified_process
from h200_connector_timeline import read,union_ns

ENTRY='h200_native_connector_capture.py'
PROTOCOL='host_API_NVTX_metadata_range_v2'
LABEL='native_model_KV_functional'


def checked_gate(gate,run):
    assert gate['protocol']==PROTOCOL and type(gate['actor_pid']) is int and gate['actor_pid']>0
    assert gate['capture_count']==1 and not gate['active'] and not gate.get('diagnostic_close_errors')
    assert len(gate['events'])==1
    event=gate['events'][0]
    assert event['mode']=='metadata_control' and event['trial']==3
    assert event['status']=='range_closed_pending_GPU_coverage_review' and not event['errors']
    for key in ('profiler_start_returned','profiler_stop_attempted','profiler_stop_returned',
                'nvtx_push_returned','nvtx_pop_returned','request_functional_phase_passed'):
        assert event[key] is True
    assert event['host_end_unix_ns']>=event['host_begin_unix_ns']
    expected=[(mode,trial) for mode in ('metadata_control','payload_proof') for trial in range(6)]
    assert [(r['mode'],r['trial']) for r in run['trials']]==expected
    assert all(r['status']=='functional_phase_passed' and r['warmup']==(r['trial']<3)
        and r['payload_proof']==(r['mode']=='payload_proof') for r in run['trials'])
    selected=run['trials'][3]
    assert event['request_id']==selected['request_id'] and not selected['warmup'] and not selected['payload_proof']
    return selected


def method_associations(timeline,spans):
    events=spans['events'];assert [r['index'] for r in events]==list(range(len(events)))
    methods={};errors=[]
    for row in timeline['selected_kernels']:
        label=row['innermost_connector_NVTX']
        if label is None:continue
        parts=label.split('/')
        if len(parts)!=3 or parts[0]!='h200_connector' or not parts[2].isdigit():
            errors.append({'label':label,'reason':'unsupported_marker'});continue
        index=int(parts[2])
        if index>=len(events):errors.append({'label':label,'reason':'missing_sidecar_event'});continue
        event=events[index];method=event['owner']+'.'+event['method']
        if parts[1]!=method or event['status']=='running':
            errors.append({'label':label,'reason':'sidecar_event_mismatch_or_incomplete'});continue
        method_row=methods.setdefault(method,{'kernel_intervals':[],'event_indices':[],
            'explicit_request_ids':set(),'unassigned_request_event_indices':set()})
        method_row['kernel_intervals'].append((row['start'],row['end']))
        method_row['event_indices'].append(index)
        if event['request_id'] is None:method_row['unassigned_request_event_indices'].add(index)
        else:method_row['explicit_request_ids'].add(event['request_id'])
    summary={name:{'kernel_records':len(row['kernel_intervals']),
        'kernel_duration_sum_ns':sum(b-a for a,b in row['kernel_intervals']),
        'kernel_active_union_ns':union_ns(row['kernel_intervals']),
        'source_event_indices':sorted(set(row['event_indices'])),
        'explicit_request_ids':sorted(row['explicit_request_ids']),
        'unassigned_request_event_indices':sorted(row['unassigned_request_event_indices'])}
        for name,row in methods.items()}
    return summary,errors


def review(root):
    root=Path(root).resolve();pair=load(root/'pair.json');traces=load(root/'results/traces.json')
    assert pair['connector_attribution_protocol']==PROTOCOL
    cert_path=root/'results/native_model_CPU_review.json';cert=load(cert_path)
    assert cert['status']=='native_model_functional_review_passed' and not cert['errors'] and cert['job_id']==pair['job_id']
    assert len(cert['roles'])==2 and {r['role'] for r in cert['roles']}=={'producer','consumer'}
    for item in cert['roles']:
        for key in ('session','run','process','monitor'):assert sha(item[key])==item[key+'_sha256']
    rows=[];errors=[]
    for role in ('producer','consumer'):
        try:
            base=root/role;session=load(base/'session.json');run=load(base/'runs/model/run.json')
            provenance=verified_process(base,LABEL,ENTRY,session)
            process=load(provenance['process'])
            expected=[str(Path(session['cuda_home'])/'bin/nsys'),'profile','--sample=none','--cpuctxsw=none',
                '--trace=cuda,nvtx','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                '--output='+str(base/'profiles/connector'),session['python'],str(Path(session['frozen_source_root'])/'scripts'/ENTRY),
                '--pair-config',str(root/'pair.json'),'--role',role,'--gpu-uuid',session['gpu_uuid'],'--output',str(base/'runs/model')]
            assert process['command']==expected,'profiled command differs from declared wrapper/flags/UUID'
            assert run['source_identity']==session['frozen_source_identity'] and run['gpu_uuid']==pair['roles'][role]['gpu_uuid']==session['gpu_uuid']
            gate_path=base/'runs/model/profiler_capture_gate.json';gate=load(gate_path);selected=checked_gate(gate,run)
            monitor=[json.loads(line) for line in Path(provenance['monitor']).read_text().splitlines()]
            assert any(row.get('owned_ancestry',{}).get(str(gate['actor_pid'])) is True for row in monitor),'profile actor not in monitored owned ancestry'
            spans_path=base/'runs/model/connector_host_spans.json';spans=load(spans_path)
            assert spans['protocol']==PROTOCOL and spans['actor_pid']==gate['actor_pid'] and not spans.get('diagnostic_close_errors')
            export=load(base/'results/processes/connector_nsys_export.json')
            assert export['status']=='complete' and export['exit_code']==0 and export['source_freeze_identity']==session['frozen_source_identity']
            assert sha(base/'logs/connector_nsys_export.log')==export['log_sha256']
            trace=traces[role];assert sha(trace['report'])==trace['report_sha256'] and sha(trace['database'])==trace['database_sha256']
            assert Path(trace['report'])==base/'profiles/connector.nsys-rep' and Path(trace['database'])==base/'profiles/connector.sqlite'
            expected_export=[str(Path(session['cuda_home'])/'bin/nsys'),'export','--type=sqlite','--output='+trace['database'],trace['report']]
            assert export['command']==expected_export
            timeline=read(trace['database'],gate['actor_pid'],'h200_connector_metadata_request/'+selected['request_id'])
            associations,association_errors=method_associations(timeline,spans)
            structure=timeline['nonempty_selected_launch_coverage'] and not timeline['foreign_kernel_globalPids'] and not timeline['missing_or_ambiguous_API_kernel_records'] and not timeline['outside_selected_launch_window_kernel_rowids'] and not association_errors
            rows.append({'role':role,'gpu_uuid':session['gpu_uuid'],'source_identity':session['frozen_source_identity'],
                'provenance':provenance,'gate':str(gate_path),'gate_sha256':sha(gate_path),
                'spans':str(spans_path),'spans_sha256':sha(spans_path),'trace':trace,
                'selected_request_id':selected['request_id'],'actor_pid':gate['actor_pid'],
                'structure_passed':bool(structure),'timeline':timeline,'method_associations':associations,
                'association_errors':association_errors})
        except (OSError,ValueError,KeyError,AssertionError,sqlite3.Error) as error:errors.append({'role':role,'error':str(error),'error_type':type(error).__name__})
    complete=len(rows)==2 and not errors and all(r['structure_passed'] for r in rows)
    return {'classification':'Independent structural/source/functional review of instrumented connector traces; no clean performance qualification',
        'status':'instrumented_capture_structure_passed' if complete else 'partial_or_rejected_connector_capture',
        'functional_CPU_certificate':str(cert_path),'functional_CPU_certificate_sha256':sha(cert_path),
        'rows':rows,'errors':errors,'structure_passed':complete,'scientific_qualified':False,
        'GPU_layer_compute_attributed':False,'primary_producer_penalty_ratio':None,'deployed_TTFT_ms':None,
        'serving_goodput':None,'physical_HBM_bytes':None,'physical_link_bytes':None,'primary_J_per_token':None,
        'scope':'CUDA activity is associated with observed host NVTX API scopes, not semantic transformer compute. Unmatched kernels, unassigned requests and overlapping execution remain explicit; per-method unions are not additive. Original twelve-request correctness proof is mandatory. Serialized functional handoff and profiler intrusion prevent direct deployed-latency/goodput or clean producer-ratio inference. Independent matched instrumentation calibration and compute attribution remain unresolved.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();result=review(args.root)
    if args.output:
        with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
