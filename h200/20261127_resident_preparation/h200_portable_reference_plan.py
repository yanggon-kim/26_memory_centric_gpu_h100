#!/usr/bin/env python3
"""CPU-only work manifest for full-length portable serving references.

The AMD generator preserves request content across rates and arrival types.
Enumerate identical input contracts once while retaining every offered-request
occurrence. A content key is not a correctness certificate or permission to
reuse timings, source/clock/isolation gates, devices or policy decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_portable_serving_trace import validate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plan(inputs):
    assert inputs, 'at least one sealed trace required'
    traces=[]
    items={}
    seen=set()
    for path in inputs:
        path=Path(path).resolve()
        raw=path.read_bytes()
        trace=json.loads(raw)
        validate(trace)
        assert trace['sha256'] not in seen, 'duplicate trace is not an independent input'
        seen.add(trace['sha256'])
        traces.append({'path':str(path), 'file_sha256':hashlib.sha256(raw).hexdigest(),
            'trace_sha256':trace['sha256'], 'split':trace['split'],
            'request_count':len(trace['requests']), 'arrival':trace['arrival'],
            'configured_rate_requests_s':trace['configured_rate_requests_s']})
        for index, request in enumerate(trace['requests']):
            contract={'model':trace['model'], 'seed':20261013,
                'prompt_ids':request['prompt_ids'], 'max_tokens':request['max_tokens'],
                'greedy':True, 'ignore_eos':True, 'repetition_penalty':1.0,
                'prefix_cache':False, 'split':trace['split']}
            key=digest(contract)
            if key not in items:
                items[key]={'input_contract_sha256':key, 'contract':contract,
                    'prompt_tokens':len(request['prompt_ids']),
                    'output_tokens':request['max_tokens'], 'occurrences':[],
                    'status':'pending_actual_full_length_references',
                    'qualified_reference':None}
            assert items[key]['contract']==contract, 'content-key collision'
            items[key]['occurrences'].append({'trace_sha256':trace['sha256'],
                'request_id':request['id'], 'offered_index':index,
                'arrival_s':request['arrival_s'],
                'trace_warmup_request':index<trace['warmup_requests']})
        assert hashlib.sha256(path.read_bytes()).hexdigest()==traces[-1]['file_sha256'], 'trace changed during read'
    rows=[items[key] for key in sorted(items)]
    occurrences=sum(len(row['occurrences']) for row in rows)
    assert occurrences==sum(row['request_count'] for row in traces)
    result={'version':'H200_portable_reference_work_plan_v1',
        'classification':'CPU-only pending reference work; no GPU evidence or runtime estimate',
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'traces':traces, 'items':rows,
        'offered_request_occurrences':occurrences,
        'unique_input_contracts':len(rows),
        'repeated_input_contract_occurrences':occurrences-len(rows),
        'required_per_contract':{
            'vLLM_source':'Actual full-length EOS-ignored greedy serial outputs; original model/revision/BF16/seed/settings and owned process provenance',
            'HF_dynamic':'Unchanged default dynamic-cache full output;3warmups and3trials, all identical and exact source agreement',
            'HF_StaticCache':'Unchanged strict StaticCache full output;3warmups and3trials, all exact source/dynamic agreement',
            'preservation':'Retain every wrong output, timeout, partial and unsupported case; no prefix-only substitution or tolerance',
            'reuse':'Input identity is necessary only. Verify source/runtime/model/settings, exact lengths and original certificate provenance before any output-reference reuse; fresh measurement gates remain required.'},
        'performance_qualified':False, 'GPU_execution':False,
        'scope':'No serving request, timestamp, length or warmup occurrence is removed. Development/judging reference contracts are separate. No policy tuning from judging outputs, timing pooling, capacity prediction or automatic acceptance of any reference.'}
    result['sha256']=digest(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=plan(args.trace)
    with args.output.open('x') as stream:
        json.dump(result,stream,separators=(',', ':'),allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('sha256','offered_request_occurrences',
        'unique_input_contracts','repeated_input_contract_occurrences','GPU_execution')}))


if __name__=='__main__':main()
