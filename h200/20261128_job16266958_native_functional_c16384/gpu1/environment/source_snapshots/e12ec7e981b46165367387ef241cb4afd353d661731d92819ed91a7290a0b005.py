#!/usr/bin/env python3
"""Reconcile saved32B physical traffic and logical demand without GPU execution."""
import argparse
from pathlib import Path
import shutil
from h200_priority_campaign import load,write,sha,valid_tokens
from shoreline_accounting import archive_headers,useful


def main(a):
    source=a.source.resolve();destination=a.destination.resolve()
    session=load(source/'session.json')
    config_path=Path('h200/20261125_priority_preparation/qwen32b_smoke.json').resolve()
    cfg=load(config_path);snapshot=Path(cfg['model_path'])
    assert sha(snapshot/'config.json')==cfg['config_sha256']
    proof=config_path.with_suffix('.download_manifest.json')
    target=destination/'configs/qwen32b_smoke.download_manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==sha(proof)
    else:shutil.copyfile(proof,target)
    header=archive_headers(destination,'qwen32b',cfg)
    reviewed_path=source/'results/P2a_L2_triplicate_review.json';reviewed=load(reviewed_path)
    assert reviewed['status']=='three_instrumented_acquisitions_audited'
    assert reviewed['gpu_uuid']==session['gpu_uuid'] and reviewed['source_freeze_identity']==session['frozen_source_identity']
    reference=load(Path(load(Path(reviewed['reference']))['run']))
    expected=reference['cases'][0]['trials'][0]['output_ids']
    rows=[]
    for item in reviewed['records']:
        path=Path(item['result']);assert sha(path)==item['result_sha256']
        result=load(path);assert result['ticks']==[4,5,6] and result['token_denominator']==3
        artifact=Path(item['counter_artifact']);assert sha(artifact)==item['counter_sha256']
        publications=[];logical=None
        for replay in result['replays']['passes']:
            publication=source/replay['publication'];assert sha(publication)==replay['sha256']
            native_path=Path(load(publication)['output'])/'run.json'
            native=load(native_path);assert valid_tokens(native,expected)
            assert native['config']['model_revision']==cfg['model_revision'] and native['config']['config_sha256']==cfg['config_sha256']
            case=native['cases'][0];assert case['batch_size']==1 and case['context_length']==4096
            interval=native_path.parent/'selected_interval.json';selected=load(interval)
            assert selected['ticks']==[4,5,6] and selected['timed_tokens']==3
            assert all(r['output_token_counts']==[r['tick']+1] for r in selected['ledger'])
            ids=case['trials'][0]['output_ids']
            inputs={tick:[request[tick-1] for request in ids] for tick in (4,5,6)}
            current=useful(header,1,4096,input_tokens=inputs)
            if logical is None:logical=current
            else:assert logical==current
            publications.append({'publication':str(publication),'publication_sha256':sha(publication),
                'run':str(native_path),'run_sha256':sha(native_path),'selected_interval':str(interval),
                'selected_interval_sha256':sha(interval)})
        assert logical is not None and logical['U_tick_sum']>0
        metrics=next(iter(result['ranges'].values()))
        reads=metrics['dram__bytes_read.sum'];writes=metrics['dram__bytes_write.sum']
        flops=sum(t['useful_linear_flops']+t['useful_attention_flops'] for t in logical['ticks'])
        rows.append({'acquisition':result['acquisition'],'physical_result':str(path),'physical_result_sha256':sha(path),
            'physical_DRAM_read_bytes':reads,'physical_DRAM_write_bytes':writes,
            'physical_L2_total_requested_bytes':metrics['lts__t_bytes.sum'],'replays':publications,
            'analytical_logical':logical,'analytical_useful_FLOPs':flops,
            'physical_read_over_U_tick_sum':reads/logical['U_tick_sum'],
            'physical_read_over_U_window_unique':reads/logical['U_window_unique'],
            'useful_FLOP_per_physical_DRAM_read_write_byte':flops/(reads+writes),
            'signed_read_minus_three_weight_streams_bytes':reads-3*logical['ticks'][0]['required_weight_bytes'],
            'signed_residual_scope':'not measured KV traffic; cache hits, rereads, writes/overfetch and omitted demand can offset',
            'clean_bandwidth_GB_s':None,'issued_FLOPs':None})
    result={'classification':'CPU analytical reprocessing of separately measured physical counters',
        'new_GPU_measurements':False,'source_root':str(source),'gpu_uuid':session['gpu_uuid'],
        'source_freeze_identity':session['frozen_source_identity'],'model_revision':cfg['model_revision'],
        'model_headers':str(header),'model_headers_sha256':sha(header),
        'model_download_manifest_source':str(proof),'model_download_manifest_sha256':sha(proof),
        'counter_review':str(reviewed_path),'counter_review_sha256':sha(reviewed_path),'cases':rows,
        'scope':'Analytical useful inputs/FLOPs are separate from measured DRAM/L2. Aggregate near-one reads/logical demand does not prove address-level exactly-once reads or absence of reuse. Clock-rejected clean timing is not joined to these counters; no architectural benefit or clean bandwidth inferred.'}
    path=destination/'results/qwen32b_physical_logical_accounting.json';write(path,result)
    lines=['# Saved Qwen32B traffic versus analytical demand','',result['scope'],'',
        f"GPU `{session['gpu_uuid']}`; model revision `{cfg['model_revision']}`.",'',
        '| Acquisition | DRAM read / sum of per-tick logical inputs | DRAM read / window-unique inputs | Useful FLOP / DRAM R+W byte |',
        '|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['acquisition']} | {row['physical_read_over_U_tick_sum']:.6f} | {row['physical_read_over_U_window_unique']:.6f} | {row['useful_FLOP_per_physical_DRAM_read_write_byte']:.6f} |")
    logical=rows[0]['analytical_logical'];tick=logical['ticks'][0]
    lines+=['',f"The analytical decoder weight input is {tick['required_weight_bytes']:,} bytes per tick, including norms/biases and the untied output head. The untied input embedding contributes only the actual selected row. Historical KV is approximately1.075GB per tick at these lengths. Geometry comes from the pinned checkpoint:64layers,40query heads,8KV heads,128head dimension; sliding windows are disabled.",
        '',f"Three-tick logical input sum: {logical['U_tick_sum']:,} bytes. Window-unique logical inputs: {logical['U_window_unique']:,} bytes. Shared weights are counted once in the latter, explaining why a repeated weight stream can yield a ratio near3without address-level rereading beyond the normal token recurrence.",
        '', 'Limitations: useful FLOPs omit non-main operations and actual issued/padded work. Logical inputs are not a compulsory-DRAM oracle. L2 requested bytes are not a hit rate. The signed residual after subtracting three weight streams is not measured KV traffic. No confidence interval, clean GB/s or latency threshold is derived from these clock-unqualified joins. Kernel attribution, restricted-side counters and grid retuning remain pending.',
        '', 'All exact model/header, physical report, replay output and selected-interval checksums are retained in `../results/qwen32b_physical_logical_accounting.json`. This analysis adds no GPU measurement and does not alter the archived original run.']
    (destination/'analysis/qwen32b_physical_logical_accounting.md').write_text('\n'.join(lines)+'\n')
    print({'physical_acquisitions_reaccounted':len(rows),'output':str(path)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);main(p.parse_args())
