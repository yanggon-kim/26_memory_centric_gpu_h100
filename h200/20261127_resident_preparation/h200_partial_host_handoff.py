#!/usr/bin/env python3
"""Reproduce raw host handoff intervals; never qualify a performance ratio."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_review_producer_quant import read_capture
from h200_review_producer_timing import check_pair


def derive(root, acquisition=0):
    root=Path(root).resolve()
    manifest=json.loads((root/'results/acquisitions.json').read_text())
    captures=[r for r in manifest['captures'] if r['acquisition']==acquisition and r['condition']=='native']
    assert len(captures)==2 and {r['role'] for r in captures}=={'producer','consumer'}
    runs={};provenance={}
    for capture in captures:
        checked,run=read_capture(capture)
        runs[capture['role']]=run
        provenance[capture['role']]={'run':capture['run'],'run_sha256':capture['run_sha256'],
            'clock':checked['clock'],'process':checked['provenance']}
    check_pair(runs['producer'],runs['consumer'])
    rows=[]
    for p,c in zip(runs['producer']['trials'],runs['consumer']['trials']):
        assert (p['request_id'],p['trial'],p['warmup'])==(c['request_id'],c['trial'],c['warmup'])
        if p['warmup']:continue
        assert p['status']==c['status']=='exact_output_passed' and p['native_sent_tensor_count']==64
        spans=c['host_step_intervals'];assert len(spans)==63
        rows.append({'trial':p['trial'],'request_id':p['request_id'],
            'producer_first_step_host_ms':p['submit_to_final_step_return_ns']/1e6,
            'p0_return_to_all_sends_observation_host_ms':(p['all_native_sends_observed_unix_ns']-p['final_step_return_unix_ns'])/1e6,
            'p0_return_to_consumer_first_file_read_host_ms':(c['first_token_file_read_unix_ns']-p['final_step_return_unix_ns'])/1e6,
            'p0_return_to_consumer_submission_host_ms':(c['submit_unix_ns']-p['final_step_return_unix_ns'])/1e6,
            'consumer_first_handoff_call_host_ms':spans[0]['elapsed_ns']/1e6,
            'consumer_following62_calls_host_ms':sum(s['elapsed_ns'] for s in spans[1:])/1e6,
            'consumer_whole63_output_request_host_ms':c['submit_to_final_step_return_ns']/1e6,
            'p0_return_to_first_consumer_output_host_ms':(spans[0]['end_unix_ns']-p['final_step_return_unix_ns'])/1e6,
            'send_observation_relative_to_first_consumer_output_host_ms':(p['all_native_sends_observed_unix_ns']-spans[0]['end_unix_ns'])/1e6,
            'two_process_harness_producer_submit_to_consumer_last_return_host_ms':(c['final_step_return_unix_ns']-p['submit_unix_ns'])/1e6})
    assert len(rows)==10
    summary={k:{'minimum':min(r[k] for r in rows),'median':statistics.median(r[k] for r in rows),
        'maximum':max(r[k] for r in rows)} for k in rows[0] if k.endswith('_ms')}
    return {'classification':'Derived raw host intervals of one complete paired capture, not an independent-acquisition performance comparison',
        'acquisition':acquisition,'timed_requests':10,'rows':rows,'raw_host_summary_ms':summary,
        'provenance':provenance,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'performance_qualified':False,'qualified_producer_ratio':None,'serving_goodput':None,
        'physical_HBM_or_link_bytes':None,'primary_J_per_token':None,
        'scope':'Host polling and file IPC are included. Intervals overlap and are not additive. Send-readiness observation is not isolated physical transfer time; consumer call spans are not pure GPU kernel latency. Preserve original cohort completion/clock/source/isolation gates and all partial captures.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--acquisition',type=int,default=0)
    parser.add_argument('--output',type=Path,help='Optional fresh derived file; existing evidence is never overwritten')
    args=parser.parse_args();result=derive(args.root,args.acquisition)
    if args.output:
        with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result['raw_host_summary_ms'],indent=2))
