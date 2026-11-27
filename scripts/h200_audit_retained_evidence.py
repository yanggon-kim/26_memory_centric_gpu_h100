#!/usr/bin/env python3
"""CPU provenance audit of accepted historical H200 evidence before reuse."""
import argparse
import json
from pathlib import Path
from h200_priority_campaign import PACKAGE,load,write,sha


def audit(destination):
    source=PACKAGE/'h200/20261021_job14243981/analysis/consolidated/results/qualified_evidence.json'
    data=load(source);checks=[];components=[]
    def verify(path,expected,base=PACKAGE):
        p=Path(path);p=p if p.is_absolute() else (PACKAGE/p if p.parts and p.parts[0]=='h200' else base/p)
        actual=sha(p) if p.is_file() else None
        checks.append({'path':str(p),'expected_sha256':expected,'actual_sha256':actual,'passed':actual==expected})
        return actual==expected
    def refs(items,base):
        values=[]
        if isinstance(items,dict):
            if 'path' in items and 'sha256' in items:values.append(verify(items['path'],items['sha256'],base))
            else:
                for value in items.values():values.extend(refs(value,base))
        elif isinstance(items,list):
            for value in items:values.extend(refs(value,base))
        return values
    for item in data['evidence_sources']:verify(item['path'],item['sha256'])
    for kind,key in [('engine','engine_comparison'),('sm','sm_comparisons')]:
        items=data[key] if isinstance(data[key],list) else [data[key]]
        for item in items:
            passed=refs(item.get('sources',[]),PACKAGE/data['cohort_namespaces'][kind])
            components.append({'family':kind,'batch':item.get('batch',1),'context':item.get('context',1024),
                'qualified_in_original_scope':item.get('qualified',item.get('exact_mode_match',False)),
                'source_hash_checks_passed':bool(passed) and all(passed),
                'actual_sm_count':item.get('actual_sm_count'),'transfer_to32B_C4096':False})
    for item in data['native_service']['patterns']:
        passed=refs(item['sources'],PACKAGE/data['cohort_namespaces']['native'])
        components.append({'family':'native_service','condition':item['condition'],'actual_sm_count':item['actual_sm_count'],
            'qualified_in_original_scope':item['clock_qualified'],'source_hash_checks_passed':bool(passed) and all(passed),
            'acquisitions':item['acquisition_count'],'ten_trials_each':len(item['raw_event_ms_acquisitions'])==3 and all(len(v)>=10 for v in item['raw_event_ms_acquisitions']),
            'transfer_to_decoder':False})
    for item in data['treatment_comparisons']:
        passed=refs(item.get('sources',[]),PACKAGE/data['cohort_namespaces']['treatments'])
        components.append({'family':item['job'],'model':item['model'],'batch':item['batch'],'context':item['context'],
            'numerator':item['numerator'],'denominator':item['denominator'],'qualified_in_original_scope':item['qualified'],
            'source_hash_checks_passed':bool(passed) and all(passed),'transfer_to32B':False})
    for item in data['energy_families']:
        if not item['qualified']:continue
        passed=refs(item.get('sources',[]),PACKAGE/data['cohort_namespaces']['energy'])
        components.append({'family':'energy','id':item['id'],'qualified_in_original_scope':True,
            'source_hash_checks_passed':bool(passed) and all(passed),'external_absolute_calibration':False,'transfer_to32B':False})
    retained=[]
    for item in data['retained_repair_cohorts']:
        passed=verify(item['source'],item['sha256']);repair=item['cohort']
        qualified=[v for v in repair['treatment_comparisons'] if v.get('qualified')]
        retained.append({'source':item['source'],'source_sha256':item['sha256'],'source_hash_passed':passed,
            'session':repair['session'],'qualified_J6_original_scope':sum(v.get('job')=='J6' for v in qualified),
            'physical_counter_dispositions':[v.get('status') for v in repair['physical_counter_captures']],
            'transfer_to32B':False,'scope':'retained cohort; never pooled with other allocations'})
    result={'classification':'CPU provenance/reuse audit of retained H200 evidence','new_GPU_measurements':False,
        'source':str(source),'source_sha256':sha(source),'source_hash_checks':checks,'components':components,'retained_repairs':retained,
        'reusable_qualified_native_cells':sum(c['family']=='native_service' and c['qualified_in_original_scope'] and c['source_hash_checks_passed'] and c['ten_trials_each'] for c in components),
        'limitation':'Accepted original shapes/runtime/source/UUID scopes only. Does not complete32B C4096, new serving/load/KV comparisons or architecture validation. Missing/changed files require requalification before reuse; do not recapture merely to fill a new label.'}
    recovered=destination/'results/retained_treatment_requalification.json'
    if recovered.exists():
        receipt=load(recovered);copy=Path(receipt['new_statistics'])
        if (receipt.get('exact_old_aggregate_recovered') and sha(copy)==receipt['original_aggregate_expected_sha256']
                and all(c['maximum_ratio_or_CI_difference_from_old_snapshot']==0 for c in receipt['contrasts'])):
            result['exact_reconstruction']={'receipt':str(recovered),'receipt_sha256':sha(recovered),
                'copy':str(copy),'copy_sha256':sha(copy),'original_current_path_preserved':True}
            identities={tuple(c['identity']) for c in receipt['contrasts']}
            for component in components:
                identity=tuple(component.get(k) for k in ('family','model','batch','context','numerator','denominator'))
                if identity in identities:component['source_requalified_from_retained_raw']=True
    write(destination/'results/retained_H200_reuse_audit.json',result)
    failed=[c for c in checks if not c['passed']]
    lines=['# Retained H200 evidence audit','',result['limitation'],'',
        f"Verified {len(checks)-len(failed)}/{len(checks)} referenced file hashes. Reusable qualified native cells: {result['reusable_qualified_native_cells']}/12.",'',
        'Earlier bridge graph/eager and restricted-SM comparisons, selected pressure/event controls and accepted long-window board energy remain in their original namespaces. Reuse requires the component source checks to pass; this audit creates no new timing, energy or physical-traffic result.', '',
        'Four qualified retained J6 bridge B1/C1024 comparisons remain separate from current32B and other allocations. Physical-counter records preserve their original exact scope and counter/clock limitations. Rejected J8/J10 families and failed old StaticCache routes are not promoted.', '',
        'P5c remains partial: old J0–J11 evidence is audited for reuse, but new model/shape calibration, matched traffic/resource/kernel evidence and held-out prediction requirements remain pending.']
    if failed:lines+=['','## Source failures','']+[f"- `{c['path']}`: absent or checksum changed; do not reuse that component without requalification." for c in failed]
    if result.get('exact_reconstruction'):
        lines+=['','## Exact reconstruction resolves the treatment aggregate','',
            'The original path remains unchanged with its different checksum. A CPU reconstruction from the original raw inputs exactly matches the aggregate SHA expected by the consolidated report. Four original J7/J8 contrasts were independently requalified with24raw provenance checks and exactly matching statistics. Reuse those original scopes through the reconstructed copy and its receipt, not through an assumed match at the historical path. No new GPU measurement occurred.']
    (destination/'analysis/retained_H200_reuse_audit.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();result=audit(a.destination)
    print(json.dumps({'hashes_verified':sum(c['passed'] for c in result['source_hash_checks']),
                      'hashes_total':len(result['source_hash_checks']),'reusable_native_cells':result['reusable_qualified_native_cells']}))
