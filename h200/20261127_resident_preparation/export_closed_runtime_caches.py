"""CPU-only cache export after every timed pilot worker has returned.

Run inside the still-owned allocation. Never initializes CUDA, submits jobs,
edits session settings or copies model weights, credentials or GPU state.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import datetime as dt

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from h200_priority_campaign import load,write,sha
from h200_native_model_followup import copy_runtime_cache


def main(args):
    core=args.core.resolve();pilot=core.with_name(core.name+'_producer_timing_pilot')
    job=os.environ['SLURM_JOB_ID'];assert job.isdigit()
    assert core.is_relative_to(REPO/'h200')
    sessions=[load(core/f'gpu{i}/session.json') for i in (0,1)]
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
    fields=dict(v.split('=',1) for v in raw.split() if '=' in v)
    assert fields['JobState']=='RUNNING' and fields['JobId']==job
    assert fields['UserId'].endswith('('+str(os.getuid())+')')
    assert fields['NodeList']==socket.gethostname().split('.')[0]
    assert 'gres/gpu:h200=2' in fields['AllocTRES'].split(',')
    assert all(s['slurm_job_id']==job and s['expected_hostname']==socket.gethostname() for s in sessions)
    assert load(core/'model/results/native_model_CPU_review.json')['status']=='native_model_functional_review_passed'
    closed=[core/'model'/r for r in ('producer','consumer')]
    closed += [pilot/'stock_before',pilot/'native/producer',pilot/'native/consumer',pilot/'stock_after']
    for root in closed:
        assert load(root/'results/current_process.json')['status']=='idle',str(root)+' still active'
    for condition,role in (('stock_before',None),('native','producer'),('native','consumer'),('stock_after',None)):
        root=pilot/condition if role is None else pilot/condition/role
        run=load(root/'runs/timing/run.json')
        assert run['status']=='complete' and len(run['trials'])==6
    destination=args.destination
    assert not destination.is_symlink() and not destination.exists()
    assert destination.parent.resolve()==Path('/storage/project/r-dmahajan7-0/jkim4112/h200_priority_runtime_caches')
    destination.mkdir(parents=True,exist_ok=False)
    receipt={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'job_id':job,
        'classification':'Verified closed compilation/registry cache copy only; no GPU measurement',
        'slurm':raw,'exporter_sha256':sha(Path(__file__)),
        'copy_helper_sha256':sha(REPO/'scripts/h200_native_model_followup.py'),
        'source_numerical_identity':sessions[0]['frozen_source_identity'],
        'copies':[],'status':'copying',
        'scope':'Compiler cache hints only; fresh model/source/defaultHF/exact-output/UUID/isolation/clock gates remain mandatory. No timing or correctness qualification is transferred. Original caches and evidence are retained.'}
    receipt_path=core/'results/runtime_cache_export.json'
    write(receipt_path,receipt)
    try:
        with contextlib.ExitStack() as stack:
            for i,session in enumerate(sessions):
                source=Path(session['workspace'])
                assert not source.is_symlink() and source.parent==Path('/tmp')
                assert source.stat().st_uid==os.getuid()
                lock=stack.enter_context((source/'producer.lock').open('a'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                target=destination/f'gpu{i}';target.mkdir(exist_ok=False)
                row=copy_runtime_cache(source,target)
                row.update(source_session=str(core/f'gpu{i}/session.json'),
                    source_session_sha256=sha(core/f'gpu{i}/session.json'))
                receipt['copies'].append(row)
                write(receipt_path,receipt)
        receipt['status']='closed_runtime_caches_export_verified'
    finally:
        write(receipt_path,receipt)
        write(destination/'manifest.json',receipt)
    print(json.dumps({'status':receipt['status'],'destination':str(destination),
        'bytes':sum(r['bytes'] for r in receipt['copies']),
        'files':sum(len(r['files']) for r in receipt['copies'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    main(p.parse_args())
