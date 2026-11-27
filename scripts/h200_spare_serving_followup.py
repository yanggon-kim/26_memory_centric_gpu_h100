#!/usr/bin/env python3
"""Bounded serving pilot after an owned spare's completed prefill comparison."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import traceback
from h200_priority_campaign import (PACKAGE, Worker, load, write, sha, publish,
                                    config, valid_tokens, strict_reference_record, limitation)


def validate_idle(outcome, current, gpu_pids, review, session_sha):
    assert outcome.get('status') == 'prefill_acquisition_disposition'
    assert outcome.get('complete_acquisition_set') is True
    assert current.get('status') == 'idle' and not gpu_pids
    assert review.get('complete_acquisition_set') is True and not review.get('errors')
    assert len(review.get('rows', [])) == 15
    assert review.get('session_sha256') == session_sha
    # Rejected clock gates on the preceding *independent* comparison transfer
    # no timing qualification and do not prohibit a new functional pilot.


def capture(worker, protected_cutoff):
    from h200_resident_campaign import build_environment
    available = min(protected_cutoff,
                    worker.hs.verify_allocation()['effective_launch_cutoff_unix_s'])-time.time()
    budget = {'available_s': available, 'producer_limit_s': 660, 'reference_limit_s': 240,
              'publication_allowance_s': 60, 'margin_s': 60,
              'required_s': 1020, 'admitted': available >= 1020,
              'basis': 'prior three-policy202.79s and strict71.29s plus18development arrival traces; conservative new-source bounded producer/reference limits',
              'scope': 'analytical admission only; aim to finish2minutes before first-hour preemption eligibility, never a guaranteed lease'}
    write(worker.root/'results/comparison_budget.json', budget)
    if not budget['admitted']:
        return {'status': 'deferred_budget', 'budget': budget}
    build_environment(worker)
    inventory = worker.execute('inventory', [worker.py, worker.script/'inventory.py',
        '--gpu-uuid', worker.uuid, '--check-cuda', '--output', worker.root/'environment/inventory.json'], timeout=180)
    assert inventory['status'] == 'complete'
    available = min(protected_cutoff,
                    worker.hs.verify_allocation()['effective_launch_cutoff_unix_s'])-time.time()
    budget['available_after_inventory_s'] = available
    budget['admitted_after_inventory'] = available >= budget['required_s']
    write(worker.root/'results/comparison_budget.json', budget)
    if not budget['admitted_after_inventory']:
        return {'status': 'deferred_budget_after_inventory', 'budget': budget}
    cfg = config(worker.root, 'graph', 4096, 3, 1)
    label = 'P3a_openloop_development_pilot'
    output = worker.root/'runs'/label
    proc = worker.execute(label, [worker.py, worker.script/'h200_serving_load.py',
        '--config', cfg, '--gpu-uuid', worker.uuid, '--output', output,
        '--pilot', '--rate', '2', '--requests', '12', '--acquisition', '0'], timeout=660)
    source = output/'run.json';manifest_path = output/'serving_load.json'
    passed = proc['status'] == 'complete' and source.exists() and manifest_path.exists()
    if passed:
        manifest = load(manifest_path)
        passed = bool(valid_tokens(load(source)))
        cleanup_path = output/'cleanup.json'
        passed = passed and cleanup_path.exists() and load(cleanup_path).get('status') == 'complete'
        passed = passed and manifest['status'] == 'captured_pending_CPU_and_intrusion_review'
        passed = passed and manifest['protocol'] == 'finite_open_loop_policy_v2'
        passed = passed and manifest['fresh_functional_policy_gate']['status'] == 'three_functional_policies_passed'
        passed = passed and len(manifest['captures']) == 18
        passed = passed and all(r['status'] == 'finite_trace_exact_outputs_and_composition_passed'
                                and sha(Path(r['path'])) == r['sha256'] for r in manifest['captures'])
    reference_path = worker.root/'results'/f'{label}_hf.json'
    if passed:
        reference = worker.execute(label+'_hf', [worker.py, worker.script/'hf_full_reference.py',
            '--source-run', source, '--gpu-uuid', worker.uuid, '--output', reference_path], timeout=240)
        passed = reference['status'] == 'complete' and reference_path.exists()
        passed = passed and strict_reference_record(load(reference_path), sha(source))
    row = {'status': 'functional_captured_pending_CPU_review' if passed else 'blocked_functional_or_reference',
           'gpu_uuid': worker.uuid, 'source_identity': worker.settings['frozen_source_identity'],
           'run': str(source), 'run_sha256': sha(source) if source.exists() else None,
           'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path) if manifest_path.exists() else None,
           'strict_HF_reference': str(reference_path),
           'strict_HF_reference_sha256': sha(reference_path) if reference_path.exists() else None,
           'quantitative_qualified': False,
           'scope': 'Fresh three-policy functional gate, then3warmup/3development traces per policy with12requests and64outputs each, one model/process. Not3independent acquisitions, tuning, steady-state serving, held-out SLOs, overhead-neutral timing or energy.'}
    write(worker.root/'results/serving_pilot_disposition.json', row)
    with (worker.root/'analysis/report.md').open('a') as stream:
        stream.write('\n## Bounded development pilot\n\n'+row['status']+'. '+row['scope']+'\n')
    if not passed:
        limitation(worker.root, label, 'Bounded functional open-loop producer or strict independent reference failed',
                    'results/serving_pilot_disposition.json',
                    'Same-model fresh three-policy gate before development traces;660second producer/240second reference limits; all partial records retained',
                    'Inspect the first failure before a separate bounded repair; no quantitative serving result is inferred')
    return row


def main(args):
    old = load(args.source_session);prior = args.source_session.resolve().parent
    assert os.environ.get('SLURM_JOB_ID') == old['slurm_job_id']
    assert socket.gethostname() == old['expected_hostname']
    state = subprocess.check_output(['scontrol','show','job','-o',old['slurm_job_id']], text=True)
    assert 'JobState=RUNNING' in state and f'({os.getuid()})' in state and 'gres/gpu:h200=2' in state
    started_at = dt.datetime.fromisoformat(re.search(r'\bStartTime=(\S+)',state).group(1))
    if started_at.tzinfo is None:
        timezone = dt.datetime.fromisoformat(old['launch_cutoff']).tzinfo
        assert timezone is not None
        started_at = started_at.replace(tzinfo=timezone)
    started = started_at.timestamp()
    protected_cutoff = started+3600-120
    import pynvml as nv
    nv.nvmlInit()
    try:
        gpu_pids = [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByUUID(old['gpu_uuid']))]
    finally:
        nv.nvmlShutdown()
    review_path = prior/'results/prefill_CPU_review.json';review = load(review_path)
    outcome = load(prior/'results/worker_outcome.json');current = load(prior/'results/current_process.json')
    validate_idle(outcome, current, gpu_pids, review, sha(args.source_session))
    for row in review['rows']:
        assert row['source_verified'] and row['gpu_uuid'] == old['gpu_uuid']
        assert row['source_identity'] == old['frozen_source_identity']
        assert sha(Path(row['run'])) == row['run_sha256']
        assert sha(Path(row['process'])) == row['process_sha256']
        assert sha(Path(row['monitor'])) == row['monitor_sha256']
    from h200_resident_campaign import verify_published_core
    previous = old['published_core_gate']
    assert sha(Path(previous['review'])) == previous['review_sha256']
    verify_published_core(Path(previous['review']))
    lock = (Path(old['workspace'])/'producer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.root = args.root.resolve();args.root.mkdir(exist_ok=False)
    for folder in ('environment','results','logs','configs','runs','analysis','profiles'):
        (args.root/folder).mkdir()
    (args.root/'analysis/report.md').write_text('# Same-allocation spare: bounded serving development pilot\n\nOriginal spare prefill worker returned; its complete source/output review was reverified. Own GPU UUID, no active GPU PIDs and original producer lock held. This is a new frozen source/workspace, not a timing extension or pooling of the prefill cohort. The other allocated GPU may continue independent work.\n')
    write(args.root/'results/idle_transition.json', {'source_session': str(args.source_session),
        'source_session_sha256': sha(args.source_session), 'prior_review': str(review_path),
        'prior_review_sha256': sha(review_path), 'gpu_pids': gpu_pids,
        'outcome': outcome, 'current_process': current, 'protected_cutoff_unix_s': protected_cutoff})
    session = dict(old);session['workspace'] = old['workspace']+'_serving_'+args.root.name
    Path(session['workspace']).mkdir(exist_ok=False)
    for key in ('frozen_source_root','frozen_source_identity'):
        session.pop(key, None)
    session['protocol'] = 'spare_finite_openloop_development_pilot_v2'
    session_path = args.root/'session.json';write(session_path, session)
    publish(args.root, 'H200 serving follow-up: publish reviewed idle admission and versioned source')
    session['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip()
    write(session_path, session)
    subprocess.run([session['python'],PACKAGE/'scripts/freeze_shoreline.py','--session',session_path,'--activate'],check=True)
    worker = Worker(args.root,session_path)
    try:
        result = capture(worker,protected_cutoff)
        write(args.root/'results/worker_outcome.json',result)
    except BaseException as error:
        write(args.root/'results/worker_outcome.json',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        limitation(args.root,'serving_followup_controller','Spare serving controller failed: '+str(error),
                    'results/worker_outcome.json','Owned completed/idle transition; separate frozen source cohort',
                    'Retain evidence and inspect before a bounded repair; no scientific result inferred')
        raise
    finally:
        publish(args.root,'H200 spare serving pilot: publish final functional disposition')
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
