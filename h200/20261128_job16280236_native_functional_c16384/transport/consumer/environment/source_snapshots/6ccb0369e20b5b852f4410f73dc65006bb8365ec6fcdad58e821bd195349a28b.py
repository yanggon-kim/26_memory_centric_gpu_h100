#!/usr/bin/env python3
"""CPU provenance review of a finalized resident curve, including rejected clocks."""
import argparse
import datetime as dt
import statistics
from pathlib import Path
from h200_priority_campaign import load,write,sha,valid_tokens,strict_reference_record
from h200_review_allocation import case_records


def review(root,batch=1,context=4096):
    rows,issues,interrupted=case_records(root)
    assert not issues and not interrupted,(issues,interrupted)
    rows=[r for r in rows if (r['batch'],r['context'])==(batch,context)]
    clean=[r for r in rows if not r['pilot']]
    assert len(clean)==27 and len({r['repair_route'] for r in clean})==1
    hf=root/'results/resident_pilot_hf.json';reference=load(hf)
    assert strict_reference_record(reference,sha(reference['source_run']))
    expected=load(reference['source_run'])['cases'][0]['trials'][0]['output_ids']
    for row in clean:
        native=load(row['run'])
        assert len(native['cases'][0]['trials'])==10 and valid_tokens(native,expected)
        assert row['conditioning_scope_verified'] and row['sm_grant_matches_requested_scope']
    for acquisition in range(3):
        selected=[r for r in clean if r['acquisition']==acquisition]
        assert len(selected)==9 and sum(r['stock'] for r in selected)==1
        assert {r['suffix'] for r in selected}=={'','_interleaved','_end'}
        assert {r['actual_sm_count'] for r in selected}=={132,112,96,80,56,40}
    proofs=[r for r in rows if r['pilot'] and '_proof_' in r['id']]
    assert len(proofs)==7 and all(r['qualified'] and r['placement_passed'] and r['exact_outputs'] for r in proofs)
    curve_path=root/'results'/f'curve_eager_b{batch}_c{context}.json';curve=load(curve_path)
    result={'classification':'CPU provenance review of measured resident eager curve',
        'reviewed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'qualified' if curve['complete_qualified_curve'] else 'complete_capture_quality_unresolved',
        'complete_acquisition_set':True,'independent_acquisitions':3,'conditions_per_acquisition':9,
        'warmups_per_condition':3,'timed_trials_per_condition':10,'batch':batch,'context':context,
        'protocol':clean[0]['repair_route'],'qualified_timing_curve':curve['complete_qualified_curve'],
        'thresholds':curve['thresholds'],'case_gate_issues':issues,
        'source_session':str(root/'session.json'),'source_session_sha256':sha(root/'session.json'),
        'strict_HF_reference':str(hf),'strict_HF_reference_sha256':sha(hf),
        'curve':str(curve_path),'curve_sha256':sha(curve_path),
        'cases':[{'id':r['id'],'run':r['run'],'run_sha256':r['run_sha256'],
            'raw_median_ms_per_tick':statistics.median(r['step_ms']),'clock_passed':r['clock']['passed'],
            'actual_sm_count':r['actual_sm_count'],'acquisition':r['acquisition']} for r in clean],
        'scope':'Same physical UUID and frozen source; this resident protocol only. Optimized graph restriction remains separately blocked. No thresholds, clean bandwidth or architecture savings from rejected controls.'}
    write(root/'results/resident_curve_CPU_review.json',result)
    lines=['# Finalized resident eager curve: CPU review','',result['scope'],'',
        'All3independent acquisitions captured9conditions each, with3warmups and10timed trials. All7actual-decoder placement proofs and both matching64-output HF reference routes passed. Producer source/run checksums and isolation records passed CPU review. Finalized JSON artifacts are parseable. Quantitative qualification remains as recorded in the curve; raw observations do not override clock/source gates.','',
        '| Condition | Acquisition0raw median ms/tick | Acquisition1 | Acquisition2 |',
        '|---|---:|---:|---:|']
    for name in ['stock','green132','green112','green96','green132_interleaved','green80','green56','green40','green132_end']:
        values=[next(r for r in clean if r['id']==f'resident_b{batch}_c{context}_a{a}_{name}') for a in range(3)]
        lines.append('| '+name+' | '+' | '.join(f"{statistics.median(r['step_ms']):.4f}" for r in values)+' |')
    lines+=['','Evidence: `../results/resident_curve_CPU_review.json`. These are retained acquisition medians; confidence intervals and thresholds require all quantitative gates.']
    (root/'analysis/resident_curve_CPU_review.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1);p.add_argument('--context',type=int,default=4096)
    a=p.parse_args();result=review(a.root,a.batch,a.context)
    print({k:result[k] for k in ('status','complete_acquisition_set','qualified_timing_curve','protocol')})
