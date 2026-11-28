#!/usr/bin/env python3
"""CPU closeout of a terminal priority allocation; never launch GPU work.

Original producers and chronological reports remain intact. Missing process
completion after Slurm termination is interruption evidence, not an unsupported
engine/restriction diagnosis.
"""
import argparse
import copy
import csv
import datetime as dt
import json
from pathlib import Path
import re

import h200_priority_campaign as campaign

TERMINAL={'COMPLETED','PREEMPTED','TIMEOUT','FAILED','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED'}


def accounting(path, job_id, expected_devices):
    rows=list(csv.DictReader(path.read_text().splitlines(),delimiter='|'))
    row=next(r for r in rows if r['JobID']==str(job_id))
    state=row['State'].split()[0].rstrip('+')
    assert state in TERMINAL, 'closeout requires terminal Slurm accounting'
    tres=dict(x.split('=',1) for x in row['AllocTRES'].split(',') if '=' in x)
    count=int(tres['gres/gpu'])
    assert count==expected_devices and count>0
    elapsed=int(row['ElapsedRaw']);assert elapsed>=0
    return {'job_id':str(job_id),'state':state,'start':row['Start'],'end':row['End'],
            'elapsed_seconds':elapsed,'allocated_gpus':count,
            'actual_allocated_gpu_hours':elapsed*count/3600,
            'scope':'allocated GPU-hours include staging, idle and instrumented work',
            'source':str(path),'source_sha256':campaign.sha(path),'raw_row':row}


def case_records(root):
    session=campaign.load(root/'session.json')
    records=[];issues=[]
    for path in sorted((root/'results').glob('*.json')):
        row=campaign.load(path)
        if not isinstance(row,dict) or not {'id','engine','fraction','batch','context','pilot','stock'}<=row.keys():
            continue
        row=copy.deepcopy(row);errors=[]
        process_path=root/'results/processes'/(row.get('producer_label',row['id'])+'.json')
        process=campaign.load(process_path) if process_path.exists() else {}
        run=Path(row['run'])
        if row.get('gpu_uuid')!=session['gpu_uuid']:errors.append('case UUID differs from its session')
        if row.get('source_identity')!=session['frozen_source_identity']:errors.append('case source differs from its session')
        if process.get('source_freeze_identity')!=session['frozen_source_identity']:errors.append('producer source differs from its session')
        if process.get('status')!='complete':errors.append('producer not complete')
        if not process.get('monitor') or process['monitor'].get('failure'):errors.append('isolation monitor missing or failed')
        if not run.is_file() or row.get('run_sha256')!=campaign.sha(run):errors.append('producer output absent or checksum changed')
        if errors:
            row['qualified']=False
            issues.append({'case':row['id'],'source':str(path),'errors':errors})
        records.append(row)
    interrupted=[]
    for path in sorted((root/'results/processes').glob('*.json')):
        process=campaign.load(path)
        if process.get('status') not in ('complete','failed'):
            interrupted.append({'label':process.get('label',path.stem),'original_status':process.get('status'),
                                'process':str(path),'sha256':campaign.sha(path),
                                'classification':'incomplete at terminal allocation; cause requires logs and Slurm state'})
    return records,issues,interrupted


def physical_records(root):
    rows=[]
    for path in sorted((root/'results').glob('P2a_*.json')):
        if not re.fullmatch(r'P2a_(graph|eager)_b\d+_c\d+_a[012]',path.stem):continue
        row=campaign.load(path)
        rows.append({'source':str(path),'sha256':campaign.sha(path),'status':row.get('status'),
                     'acquisition':row.get('acquisition'),'gpu_uuid':row.get('gpu_uuid'),
                     'replay_passed':row.get('replays',{}).get('passed'),
                     'ranges':row.get('ranges',{}),'token_denominator':row.get('token_denominator'),
                     'clock_join':row.get('clock_join'),'ordinary_latency':None})
        reviewed=path.with_name(path.stem+'_reader_review.json')
        if reviewed.exists():
            data=campaign.load(reviewed)
            assert data['original_result_sha256']==campaign.sha(path)
            rows[-1]['separate_reader_review']={'source':str(reviewed),'status':data['status'],
                                                'sha256':campaign.sha(reviewed),'ranges':data.get('ranges')}
    return rows


def closeout(root, accounting_path):
    allocation=campaign.load(root/'environment/allocated_devices.json')
    result={'schema_version':1,'reviewed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'classification':'CPU reconciliation of saved measured evidence',
            'allocation':accounting(accounting_path,allocation['slurm_job_id'],len(allocation['devices'])),
            'devices':[],'new_GPU_measurements':False,'NVIDIA_thresholds_derived_from_AMD':False}
    for index,device in enumerate(allocation['devices']):
        sub=root/f'gpu{index}'
        if not (sub/'session.json').exists():
            result['devices'].append({'gpu_uuid':device['uuid'],'status':'no worker session'});continue
        session=campaign.load(sub/'session.json');assert session['gpu_uuid']==device['uuid']
        records,issues,interrupted=case_records(sub)
        curves=[]
        for engine,batch,context in sorted({(r['engine'],r['batch'],r['context']) for r in records}):
            curve=campaign.analyze_curve(sub,records,engine,batch,context)
            attempts=[r for r in records if (r['engine'],r['batch'],r['context'])==(engine,batch,context)]
            expected={(f,a) for f in campaign.FRACTIONS for a in range(3)}
            actual={(r['fraction'],r['acquisition']) for r in attempts if not r['stock'] and not r['pilot'] and not r.get('suffix') and r.get('status')=='complete' and len(r.get('step_ms',[]))==10}
            stock={r['acquisition'] for r in attempts if r['stock'] and not r['pilot'] and not r.get('suffix') and r.get('status')=='complete' and len(r.get('step_ms',[]))==10}
            drift=all(sorted(r['acquisition'] for r in attempts if not r['stock'] and not r['pilot'] and r.get('suffix')==s and r.get('status')=='complete' and len(r.get('step_ms',[]))==10)==[0,1,2] for s in ('_interleaved','_end'))
            captured=expected<=actual and stock=={0,1,2} and drift
            proofs={r['fraction'] for r in attempts if r['pilot'] and not r['stock'] and r.get('qualified')}
            curves.append({'engine':engine,'batch':batch,'context':context,
                'complete_acquisition_set':captured,'all_requested_placement_pilots_passed':set(campaign.FRACTIONS)<=proofs,
                'complete_qualified_curve':curve['complete_qualified_curve'],'thresholds':curve['thresholds'],
                'source':str(sub/'results'/f'curve_{engine}_b{batch}_c{context}.json'),
                'disposition':'qualified' if curve['complete_qualified_curve'] else 'captured_quality_unresolved' if captured else 'incomplete',
                'resume':'Do not rerun merely for favorable clocks; inspect specific failed gates and advance eligible work' if captured and not curve['complete_qualified_curve'] else 'Review saved placement/quality evidence and actual case budget before a fresh cohort'})
        native=None
        native_paths=[sub/'runs'/f'native_f{round(f*100)}_a{a}'/'run.json' for f in campaign.FRACTIONS for a in range(3)]
        if all(p.is_file() for p in native_paths):
            try:
                from h200_analyze_priority_partial import native as analyze_native
                analyzed=analyze_native(sub)
                native={'status':'analyzed','qualified_condition_curves':analyzed['qualified_curve_count'],
                        'source':str(sub/'results/native_priority_analysis.json'),
                        'scope':'native timing/useful payload only; no decoder threshold or physical HBM service inferred'}
            except (AssertionError,KeyError,ValueError,OSError) as error:
                native={'status':'unqualified','error':str(error),'scope':'all originals retained; inspect failure before a new claim'}
        result['devices'].append({'gpu_uuid':session['gpu_uuid'],'source_freeze_identity':session['frozen_source_identity'],
            'curves':curves,'case_gate_issues':issues,'interrupted_producers':interrupted,
            'physical_captures':physical_records(sub),'native':native,
            'physical_join_scope':'instrumented capture status only; pooled source/clock/clean timing, L2 and kernel attribution remain separate gates'})
    campaign.write(root/'results/allocation_closeout.json',result)
    manifest_path=root/'results/case_manifest.json'
    manifest=copy.deepcopy(campaign.load(manifest_path)) if manifest_path.exists() else {'schema_version':1,'rows':[{'id':r,'status':'pending'} for r in campaign.ROWS]}
    core=[c for d in result['devices'] for c in d.get('curves',[]) if c['batch']==1 and c['context']==4096]
    physical=[r for d in result['devices'] for r in d.get('physical_captures',[])]
    for row in manifest['rows']:
        if row['id']=='P1a':
            qualified=any(c['engine']=='graph' and c['complete_qualified_curve'] for c in core)
            row.update(status='complete' if qualified else 'partial' if core else 'pending',
                       reason='Optimized core curve qualified' if qualified else 'See allocation closeout for incomplete or quality-rejected curves; eager remains separately labeled',
                       reviewed_evidence=True,evidence='results/allocation_closeout.json')
        elif row['id']=='P2a' and physical:
            row.update(status='partial',reason='Physical captures retained; complete source/clock/clean-timing, L2 and kernel-attribution comparison remains separate',
                       reviewed_evidence=True,evidence='results/allocation_closeout.json')
    campaign.write(root/'results/case_manifest_closeout.json',manifest)
    lines=[f"# Allocation {allocation['slurm_job_id']} closeout",'',
           f"Slurm state: **{result['allocation']['state']}**. Allocated GPU-hours: **{result['allocation']['actual_allocated_gpu_hours']:.4f}**, including staging and idle.",'',
           'CPU reconciliation only. Original producers, partial runs and failures are retained. Different physical devices are not pooled.','',
           '| GPU UUID | Engine | Batch/context | Acquisition set captured | Qualified curve | Disposition |','|---|---|---|---|---|---|']
    for d in result['devices']:
        for c in d.get('curves',[]):
            lines.append(f"| {d['gpu_uuid']} | {c['engine']} | {c['batch']}/{c['context']} | {c['complete_acquisition_set']} | {c['complete_qualified_curve']} | {c['disposition']} |")
    lines+=['','See `../results/allocation_closeout.json` for exact accounting, source identities, thresholds, interrupted producers and physical capture records.',
            'A completed acquisition set with failed clock/source gates remains unqualified. Do not resample solely for favorable clocks. An interrupted proof does not establish that SM restriction is unsupported.',
            'Physical counters remain instrumented evidence; this review does not establish a clean GB/s join or a serving result.',
            'Before another authorized two-H200 submission, select the highest-priority unfinished eligible comparison and validate its launcher. Reuse completed results only within their recorded scope.','']
    (root/'analysis/allocation_closeout.md').write_text('\n'.join(lines))
    original=campaign.load(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
    original.update(status='partial',closeout_reviewed=True,termination=result['allocation']['state'],
                    actual_allocated_gpu_hours=result['allocation']['actual_allocated_gpu_hours'],
                    elapsed_seconds=result['allocation']['elapsed_seconds'],closeout='results/allocation_closeout.json',
                    reviewed_case_manifest='results/case_manifest_closeout.json')
    authorization=campaign.PACKAGE/'h200/20261125_priority_preparation/results/continuation_authorization.json'
    if authorization.exists():
        original.update(next_command_requires_new_allocation_authorization=False,
                        continuation_authorization=str(authorization),continuation_scope='sequential two-H200 jobs only')
    campaign.write(root/'STATUS.json',original)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--accounting',type=Path,help='sacct -P output with JobID,State,Start,End,ElapsedRaw,AllocTRES')
    args=parser.parse_args()
    path=args.accounting or args.root/'environment/final_slurm_accounting.psv'
    result=closeout(args.root,path)
    print(json.dumps({'job_id':result['allocation']['job_id'],'state':result['allocation']['state'],
                      'gpu_hours':result['allocation']['actual_allocated_gpu_hours'],'report':str(args.root/'analysis/allocation_closeout.md')}))


if __name__=='__main__':main()
