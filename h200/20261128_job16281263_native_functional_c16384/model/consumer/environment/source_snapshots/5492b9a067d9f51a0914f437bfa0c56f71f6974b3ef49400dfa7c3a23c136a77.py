#!/usr/bin/env python3
"""CPU audit of the saved same-cohort DRAM/L2 and scheduler-observer bundles."""
import argparse
import math
from pathlib import Path
import statistics
from h200_priority_campaign import load,write,sha,valid_tokens


def process(root,label,session,gpu=True):
    path=root/'results/processes'/(label+'.json');data=load(path)
    assert data['status']=='complete' and data['exit_code']==0
    if gpu:assert data['monitor']['failure'] is None and Path(data['monitor']['source']).is_file()
    assert data['source_freeze_identity']==session['frozen_source_identity']
    return data,{'path':str(path),'sha256':sha(path)}


def review(root):
    session=load(root/'session.json');uuid=session['gpu_uuid'];source=session['frozen_source_identity']
    traffic=[];signatures=[]
    for acquisition in range(3):
        label=f'P2a_graph_b1_c4096_a{acquisition}';path=root/'results'/(label+'.json');data=load(path)
        assert data['status']=='complete' and data['gpu_uuid']==uuid and data['acquisition']==acquisition
        assert data['replays']['passed'] and data['ticks']==[4,5,6] and data['token_denominator']==3
        assert len(data['ranges'])==1 and data['ordinary_latency'] is None
        proc,provenance=process(root,label,session);signatures.append(proc['script_sha256'])
        process(root,label+'_import',session,gpu=False)
        for publication in data['replays']['passes']:
            target=root/publication['publication']
            assert publication['passed'] and sha(target)==publication['sha256'] and load(target)['status']=='complete'
        artifact=Path(data['counter_artifact']);assert artifact.is_file() and artifact.stat().st_size>0
        metrics=next(iter(data['ranges'].values()))
        assert all(math.isfinite(v) and v>=0 for v in metrics.values())
        assert all(metrics[k]>0 for k in ('dram__bytes_read.sum','gpu__time_duration.sum','lts__t_bytes.sum'))
        traffic.append({'acquisition':acquisition,'result':str(path),'result_sha256':sha(path),
            'producer':provenance,'counter_artifact':str(artifact),'counter_sha256':sha(artifact),
            'physical_DRAM_read_bytes_per_token':metrics['dram__bytes_read.sum']/3,
            'physical_DRAM_write_bytes_per_token':metrics['dram__bytes_write.sum']/3,
            'L2_total_requested_bytes_per_token':metrics['lts__t_bytes.sum']/3,
            'instrumented_three_step_duration_ms':metrics['gpu__time_duration.sum']/1e6})
    assert all(x==signatures[0] for x in signatures)
    reference_path=root/'results/graph_b1_c4096_stock_f100_a0_traffic_reference.json'
    reference=load(reference_path)
    assert reference['gpu_uuid']==uuid and reference['source_identity']==source
    assert reference['exact_outputs'] and reference['run_sha256']==sha(Path(reference['run']))
    hf=root/'results/P2a_reference_hf.json'
    # The original driver records its independent HF reference under its label.
    if not hf.exists():
        candidates=[p for p in (root/'results').glob('*hf.json')
                    if load(p).get('source_run')==reference['run'] and load(p).get('status')=='passed']
        assert len(candidates)==1,'independent traffic reference missing or ambiguous'
        hf=candidates[0]
    assert load(hf)['status']=='passed' and load(hf)['source_sha256']==sha(Path(reference['run']))
    physical={'status':'three_instrumented_acquisitions_audited','gpu_uuid':uuid,'source_freeze_identity':source,
        'records':traffic,'reference':str(reference_path),'independent_HF':str(hf),
        'clean_timing_clock_passed':reference['clock']['passed'],'clean_bandwidth':None,'L2_hit_rate':None,
        'scope':'Three physical DRAM/L2 totals for full optimized decode, three ticks each. L2 metric is total requested bytes. No clean bandwidth, address-level exactly-once claim, architectural SRAM saving or restricted-arm explanation is established.'}
    write(root/'results/P2a_L2_triplicate_review.json',physical)
    pilot_path=root/'results/P3a_observer_pilot.json';pilot=load(pilot_path)
    assert pilot['status']=='six_producers_completed_exact_outputs' and len(pilot['records'])==6
    assert {(r['acquisition'],r['observed']) for r in pilot['records']}=={(a,o) for a in range(3) for o in (False,True)}
    hfpath=root/'results/P3a_clean_hf.json';hfdata=load(hfpath)
    assert hfdata['status']=='passed' and hfdata['gpu_uuid']==uuid
    expected=load(Path(pilot['records'][0]['run']))['cases'][0]['trials'][0]['output_ids']
    records=[];scheduler_identities=[];observer_signatures=[]
    for row in pilot['records']:
        assert row['producer_complete'] and row['exact_outputs']
        assert row['gpu_uuid']==uuid and row['source_freeze_identity']==source
        path=Path(row['run']);assert sha(path)==row['run_sha256']
        run=load(path);assert valid_tokens(run,expected)
        assert len(run['cases'][0]['trials'])==10 and len(run['cases'][0]['warmup_output_ids'])==3
        proc,provenance=process(root,path.parent.name,session);observer_signatures.append(proc['script_sha256'])
        record={'acquisition':row['acquisition'],'observed':row['observed'],'run':str(path),
                'run_sha256':sha(path),'producer':provenance,'clock_passed':row['clock']['passed'],
                'raw_unqualified_median_ms_per_tick':statistics.median(row['raw_step_ms'])}
        if row['observed']:
            assert run['classification']=='measured-real-GPU-instrumented'
            assert run['scheduler_observation']['clean_timing_eligible'] is False
            files=list(path.parent.glob('scheduler_*.json'));assert len(files)==13
            for file in files:
                data=load(file);scheduler_identities.append(data['identity'])
                rows=data['rows'];assert len(rows)==64
                assert sum(r['prompt_input_queries'] for r in rows)==4096
                assert sum(r['post_prompt_input_queries'] for r in rows)==63
                assert sum(r['composition']=='prefill_only' for r in rows)==1
                assert sum(r['composition']=='decode_only' for r in rows)==63
                assert not any(r['any_preempted_request'] or r['composition']=='mixed' for r in rows)
            record['scheduler_files']=[{'path':str(p),'sha256':sha(p)} for p in sorted(files)]
            record.update(scheduler_rows=832,prompt_input_queries=53248,post_prompt_input_queries=819)
        records.append(record)
    assert all(x==observer_signatures[0] for x in observer_signatures)
    assert all(x==scheduler_identities[0] for x in scheduler_identities)
    pooled={m:max(r['clock']['clocks'][m]['maximum'] for r in pilot['records'])/
              min(r['clock']['clocks'][m]['minimum'] for r in pilot['records'])
            for m in ('sm_clock_mhz','memory_clock_mhz')}
    observed={'status':'three_functional_observer_pairs_audited','gpu_uuid':uuid,'source_freeze_identity':source,
        'records':records,'pooled_clock_ratios':pooled,'qualified_clean_latency_ratio':None,
        'scheduler_identity':scheduler_identities[0],
        'scope':'B1/C4096 decode-only observer validation with one full prefill step and63decode steps per request. Not loaded serving, tuned policies, mixed arrivals, held-out SLOs or demonstrated negligible observer intrusion. Clock gates failed; all raw trials retained.'}
    write(root/'results/P3a_observer_review.json',observed)
    lines=['# Full-device physical traffic and scheduler-observer review','',
        'These are saved measurements from this UUID/source cohort, audited on CPU after preemption. No new GPU execution occurred.',
        '',f'GPU `{uuid}`; source freeze `{source}`.','',
        '## Three DRAM/L2 acquisitions','',
        '| Acquisition | DRAM read B/token | DRAM write B/token | L2 requested B/token | Instrumented 3-step ms |',
        '|---|---:|---:|---:|---:|']
    for row in traffic:
        lines.append(f"| {row['acquisition']} | {row['physical_DRAM_read_bytes_per_token']:.1f} | {row['physical_DRAM_write_bytes_per_token']:.1f} | {row['L2_total_requested_bytes_per_token']:.1f} | {row['instrumented_three_step_duration_ms']:.4f} |")
    lines+=['',physical['scope'],'',
        'Limitation: the clean timing reference failed its clock envelope. Physical totals cannot be joined to qualified clean latency or GB/s. The native-linker repair made collection functional; restricted-side counters, kernel-grid retuning and complete attribution remain unfinished. Resume with those separately qualified comparisons; retain all captures and original failed linker attempt.',
        '', '## Three scheduler-observer pairs','',
        'All six independent producers completed exact outputs with three warmups and ten timed trials each. Each observed producer retained13scheduler files,832step records,53,248prompt input queries and819post-prompt input queries, with no mixed steps or preemptions. Checksums and source identities are in `../results/P3a_observer_review.json`.',
        '',observed['scope'],'',
        'Limitation: pooled SM clock ratio is '+str(pooled['sm_clock_mhz'])+'; the5%clock gate fails. No observer latency ratio is qualified. Resume with explicit prefill-only/mixed/load adapters and matched intrusion review before serving conclusions.']
    (root/'analysis/counter_observer_review.md').write_text('\n'.join(lines)+'\n')
    return {'physical_acquisitions':3,'scheduler_pairs':3,'clean_timing_qualified':False,'report':str(root/'analysis/counter_observer_review.md')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();print(review(a.root))
