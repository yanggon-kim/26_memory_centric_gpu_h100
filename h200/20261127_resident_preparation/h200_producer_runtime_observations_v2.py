"""Prospective archive/provenance adapter for the startup-budget correction.

No Slurm query, CUDA, GPU launch, certificate rewrite or performance pooling.
Promote/version only at a future safe source boundary; current job unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_priority_campaign import PACKAGE,load,sha
from h200_producer_quant_budget import budget
from h200_review_producer_quant import read_capture
from h200_producer_startup_budget_v2 import propose


def archived_reader(root):
    root=Path(root).resolve()
    assert root.is_relative_to(PACKAGE/'h200')
    receipts=list(root.glob('results/archive_transfer_*.json'));assert len(receipts)==1
    receipt=load(receipts[0]);assert receipt['status']=='archive_transfer_verified'
    assert receipt['every_archive_member_verified'] is True
    assert receipt['archive_sha256']==receipt['destination_sha256']
    assert Path(receipt['source_root']).resolve()==root
    manifest_path=Path(receipt['source_manifest']);assert sha(manifest_path)==receipt['manifest_sha256']
    manifest=load(manifest_path);members={r['path']:r for r in manifest['files']}
    assert len(members)==len(manifest['files'])
    def checked(path):
        path=Path(path).resolve();relative=str(path.relative_to(root))
        assert sha(path)==members[relative]['sha256'], 'runtime evidence differs from original archive'
        return path
    return checked,{'receipt':str(receipts[0]),'archive_sha256':receipt['archive_sha256']}


def collect(pilot_root,failed_root):
    pilot_root=Path(pilot_root).resolve();failed_root=Path(failed_root).resolve()
    pilot=budget(pilot_root) # Original complete/correct pilot/source reader.
    pilot_check,pilot_archive=archived_reader(pilot_root)
    failed_check,failed_archive=archived_reader(failed_root)
    outcome=load(failed_check(failed_root/'results/worker_outcome.json'))
    assert outcome['status']=='blocked_or_partial_fixed3x10' and not outcome.get('review_error')
    review_path=failed_check(outcome['review']);assert sha(review_path)==outcome['review_sha256']
    reviewed=load(review_path)
    assert reviewed['complete'] is False and reviewed['performance_qualified'] is False
    pilot_captures=load(pilot_check(pilot_root/'results/captures.json'))['captures']
    failed_captures=load(failed_check(failed_root/'results/acquisitions.json'))['captures']
    assert len(pilot_captures)==4 and len(failed_captures)==3
    partial_base=failed_root/'a0/stock_after'
    partial={'condition':'stock_after','role':'stock','base':str(partial_base),
        'run':str(partial_base/'runs/timing/run.json'),
        'process':str(partial_base/'results/processes/matched_producer_timing.json')}
    observations=[];provenance=[];versions=None;partial_count=0
    for captures,checked,archive in ((pilot_captures,pilot_check,pilot_archive),
            (failed_captures+[partial],failed_check,failed_archive)):
        for capture in captures:
            base=Path(capture['base']).resolve()
            run_path=checked(capture['run']);process_path=checked(capture['process'])
            run=load(run_path);process=load(process_path);session_path=checked(base/'session.json');session=load(session_path)
            assert process['status'] in ('complete','timeout') and not process['monitor']['failure']
            assert process['script_sha256']['h200_producer_timing_role.py']==pilot['role_source_sha256']
            assert process['source_freeze_identity']==run['source_identity']==session['frozen_source_identity']
            assert run['role']==capture['role'] and run['gpu_uuid']==session['gpu_uuid']
            assert run['engine_settings_without_KV_object']==pilot['engine_settings']
            if versions is None:versions=run['versions']
            assert run['versions']==versions
            snapshot=checked(base/'environment/source_snapshots'/(pilot['role_source_sha256']+'.py'))
            assert sha(snapshot)==pilot['role_source_sha256']
            log=checked(base/'logs/matched_producer_timing.log');assert sha(log)==process['log_sha256']
            monitor_path=checked(process['monitor']['source']);samples=[json.loads(line) for line in monitor_path.read_text().splitlines()]
            assert samples and any(r['pids'] for r in samples)
            assert all(r['gpu_uuid']==session['gpu_uuid'] and not r.get('rejection') and
                all(v is True for v in r['owned_ancestry'].values()) for r in samples)
            pair_path=base.parent/'pair.json' if capture['condition']=='native' else base/'pair.json'
            pair_path=checked(pair_path);pair=load(pair_path);assert sha(pair_path)==run['pair_sha256']
            assert pair['context']==pilot['context'] and pair['generated_tokens']==64
            assert pair['job_id']==session['slurm_job_id']
            done=[r for r in run['trials'] if r['status']=='exact_output_passed']
            if process['status']=='timeout':
                partial_count+=1
                assert capture==partial and process['exit_code']==-15
                assert pair['mode']=='comparison' and pair['acquisition']==0 and pair['timed_trials']==10
                assert len(done)==6 and len(run['trials'])==7 and run['trials'][-1]['status']=='running'
                assert sum(not r['warmup'] for r in done)==3
                reference=load(pair['reference_run']);assert sha(pair['reference_run'])==pair['reference_run_sha256']==run['reference_run_sha256']
                expected=reference['cases'][0]['trials'][0]['output_ids'][0][:1]
                assert all(len(r['outputs'])==1 and r['outputs'][0]['output_ids']==expected for r in done)
            elif pair['mode']=='comparison':read_capture(capture)
            times=[(r['completed_unix_ns']-r['raw_energy_counter']['before']['query_start_unix_ns'])/1e9 for r in done]
            observations.append({'condition':capture['condition'],'role':capture['role'],
                'source_role_sha256':pilot['role_source_sha256'],'context':pair['context'],
                'engine_settings':run['engine_settings_without_KV_object'],'elapsed_s':process['elapsed_s'],
                'completed_lifecycle_s':times,'process_status':process['status']})
            provenance.append({'run':str(run_path),'run_sha256':sha(run_path),'process':str(process_path),
                'process_sha256':sha(process_path),'session_sha256':sha(session_path),
                'monitor_sha256':sha(monitor_path),'archive':archive})
    assert len(observations)==8 and partial_count==1
    return {'pilot':pilot,'observations':observations,'provenance':provenance,
        'reader_sha256':sha(Path(__file__)),'GPU_execution':False,'performance_samples_used':0}


def revised_budget(pilot_root,failed_root):
    inputs=collect(pilot_root,failed_root)
    result=propose(inputs['pilot'],inputs['observations'])
    result['runtime_provenance']=inputs['provenance']
    result['runtime_reader_sha256']=inputs['reader_sha256']
    result['runtime_budget_source_sha256']=sha(Path(__file__).with_name('h200_producer_startup_budget_v2.py'))
    return result
