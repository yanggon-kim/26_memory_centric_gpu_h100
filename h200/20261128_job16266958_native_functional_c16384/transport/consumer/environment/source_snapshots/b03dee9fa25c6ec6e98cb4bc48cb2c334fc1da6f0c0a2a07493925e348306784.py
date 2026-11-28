#!/usr/bin/env python3
"""Resume GPU1 counter acquisitions1/2 after a reader-only failure of acquisition0.

Runs within the existing allocation, outside an outer h200_session supervisor.
Reuses the original immutable numerical tree and HF correctness reference.
It never submits a Slurm job or upgrades scientific clock/timing qualifications.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from h200_priority_campaign import Worker, load, write, sha, valid_tokens, publish, limitation
from h200_ncu_read import parse_csv, resolve_report


def reference(source_path, hf_path, uuid):
    source, hf = load(source_path), load(hf_path)
    assert source['hardware']['uuid']==uuid==hf['gpu_uuid']
    assert hf['source_sha256']==sha(source_path) and hf['status']=='passed'
    assert hf['reference_outputs_identical'] is True
    expected=hf['trials'][0]['output_ids']
    assert len(hf['warmups'])>=3 and len(hf['trials'])>=3
    assert all(r['output_ids']==expected for r in hf['warmups']+hf['trials'])
    assert valid_tokens(source,expected)
    case=source['cases'][0]
    assert (case['batch_size'],case['context_length'],case['generated_tokens'])==(1,4096,64)
    assert len(case['trials'])==10 and case['prompt_sha256']==hf['prompt_sha256']
    assert case['prompt_ids']==hf['prompt_ids']
    assert source['config']['model_revision']==hf['revision']
    assert source['config']['config_sha256']==hf['config_sha256']
    assert source['engine_settings']['dtype']=='bfloat16' and hf['precision']=='BF16'
    assert source['versions']['torch']==hf['versions']['torch']
    assert source['versions']['cuda_runtime']==hf['versions']['cuda']
    return source,case


def main(args):
    old=args.original.resolve();root=args.output.resolve()
    session=load(old/'session.json')
    assert os.environ.get('SLURM_JOB_ID')==session['slurm_job_id']
    label='graph_b1_c4096_stock_f100_a0_traffic_reference'
    source_path=old/'runs'/label/'run.json'
    hf_path=old/'results/graph_b1_c4096_traffic_hf.json'
    source,expected=reference(source_path,hf_path,session['gpu_uuid'])
    original=load(old/'results/P2a_graph_b1_c4096_a0.json')
    process=load(old/'results/processes/P2a_graph_b1_c4096_a0.json')
    assert original['replays']['passed'] and process['status']=='complete'
    assert process['source_freeze_identity']==session['frozen_source_identity']
    values=parse_csv((old/'logs/P2a_graph_b1_c4096_a0_import.log').read_text())
    assert len(values)==1 and next(iter(values.values()))['dram__bytes_read.sum']>0
    root.mkdir(exist_ok=False)
    for folder in ('environment','results','logs','configs','runs','profiles','analysis'):
        (root/folder).mkdir()
    (root/'analysis/report.md').write_text('# GPU1 physical-counter reader repair and continuation\n\nOriginal failed reader result is preserved. No clock, clean timing or bandwidth join is inferred.\n')
    session['resumed_from']=str(old/'session.json')
    session['reader_repair']='NCU2026 wide/legacy long CSV parser; numerical source tree unchanged'
    write(root/'session.json',session)
    write(root/'results/reused_reference.json',{'source':str(source_path),'source_sha256':sha(source_path),
          'hf':str(hf_path),'hf_sha256':sha(hf_path),'gpu_uuid':session['gpu_uuid'],
          'source_freeze_identity':session['frozen_source_identity'],
          'collector_sha256':sha(Path(__file__)),'reader_sha256':sha(Path(__file__).with_name('h200_ncu_read.py'))})
    requalified=copy.deepcopy(original)
    requalified.update(status='complete',ranges=values,reader_requalification_only=True,
                       original_status=original['status'],original_result=str(old/'results/P2a_graph_b1_c4096_a0.json'),
                       original_result_sha256=sha(old/'results/P2a_graph_b1_c4096_a0.json'),
                       counter_artifact=str(resolve_report(old/'profiles/P2a_graph_b1_c4096_a0.ncu-rep')))
    write(root/'results/P2a_graph_b1_c4096_a0_reader_review.json',requalified)
    worker=Worker(root,root/'session.json');worker.expected=expected['trials'][0]['output_ids']
    worker.hs.verify_allocation()
    publish(root,'H200 GPU1: requalify existing physical capture with corrected reader; no recapture')
    clean=[str(old/'results'/(label+'.json'))]
    for acquisition in (1,2):
        control=worker.case('graph',1.,acquisition,stock=True,suffix='_traffic_reference')
        clean.append(str(root/'results'/(control['id']+'.json')))
        assert control['status']=='complete','new exact stock reference failed'
        actual=load(control['run'])
        assert actual['cases'][0]['prompt_sha256']==expected['prompt_sha256']
        assert actual['engine_settings']==source['engine_settings'],'effective engine settings differ'
        worker.traffic(acquisitions=[acquisition],reference_case=expected)
    captures=[requalified]+[load(root/'results'/f'P2a_graph_b1_c4096_a{i}.json') for i in (1,2)]
    status='three_counter_acquisitions_captured' if all(x['status']=='complete' for x in captures) else 'partial'
    write(root/'results/comparison.json',{'status':status,'gpu_uuid':session['gpu_uuid'],
          'acquisitions':captures,'clean_reference_rows':clean,'ordinary_latency':None,
          'qualification':'Exact replay/output and physical counter range gates only; pooled clock/source/timing joins remain pending'})
    write(root/'STATUS.json',{'status':status})
    with (root/'analysis/report.md').open('a') as stream:
        stream.write('\nCapture disposition: '+status+'. See `results/comparison.json`. Acq0 reused its original GPU execution; acq1/2 use fresh producers and paired fresh3-warmup/10-trial full-device controls. All numerical producers use the original frozen tree. Clock/source joins remain pending.\n')
    publish(root,'H200 GPU1: complete reader repair and remaining physical-counter acquisitions')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():
            write(args.output/'STATUS.json',{'status':'partial','error':str(error)})
            limitation(args.output,'P2a_reader_resume','Counter continuation failed: '+str(error),
                       str(args.output),'Preserved original capture and each new attempt; no overwritten producer data',
                       'Inspect exact reference/reader/producer failure in this allocation')
            publish(args.output,'H200 GPU1: preserve partial counter reader continuation')
        raise
