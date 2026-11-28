"""Prospective CPU cache export at a serial entrypoint's functional boundary.

Not wired into the current queued job. A future parent must call this before
creating any follow-up root and bound this CPU subprocess to120seconds. Never
race an independently running parent to seize its idle gap. No CUDA context,
Slurm submission, model weights, credentials or correctness/timing transfer.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.append(str(REPO/'scripts'))
from h200_priority_campaign import load,write,sha
from h200_native_transport_pair import validate_sources
from h200_native_model_followup import copy_runtime_cache

DESTINATION=Path('/storage/project/r-dmahajan7-0/jkim4112/h200_priority_runtime_caches')
MAX_TOTAL_BYTES=512*1024**2


def admission(sessions,currents,job,hostname,uid,slurm,outcome,certificate,
              sibling_roots,all_processes,now):
    validate_sources(sessions,currents,job,hostname,uid,slurm)
    assert outcome['status']=='functional_prerequisites_passed'
    assert certificate['status']=='native_model_functional_review_passed'
    assert certificate['job_id']==job and not certificate['errors']
    assert len({s['frozen_source_identity'] for s in sessions})==1
    assert not sibling_roots, 'export only before any follow-up root is created'
    assert all_processes and all(r.get('status')=='idle' for r in all_processes)
    fields=dict(v.split('=',1) for v in slurm.split() if '=' in v)
    cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
        *(dt.datetime.fromisoformat(s['launch_cutoff']).timestamp() for s in sessions))
    assert cutoff-now>=180, '120second CPU bound plus60second margin required'
    return {'job_id':job,'available_s':cutoff-now,'required_s':180,
        'CPU_subprocess_bound_s':120,'source_identity':sessions[0]['frozen_source_identity']}


def idle_devices(nv,sessions):
    for session in sessions:
        handle=nv.nvmlDeviceGetHandleByUUID(session['gpu_uuid'])
        assert not nv.nvmlDeviceGetComputeRunningProcesses(handle), 'owned GPU still has compute processes'
        assert not nv.nvmlDeviceGetGraphicsRunningProcesses(handle), 'owned GPU still has graphics processes'


def archived_certificate(archive,manifest,core,certificate_sha):
    assert archive['status']=='archive_transfer_verified' and archive['every_archive_member_verified'] is True
    assert archive['archive_sha256']==archive['destination_sha256']
    assert Path(archive['source_root']).resolve()==core.resolve()
    assert any(row['path']=='model/results/native_model_CPU_review.json' and
        row['sha256']==certificate_sha for row in manifest['files']), 'functional certificate absent from verified archive'


def main(args):
    core=args.core.resolve();job=os.environ.get('SLURM_JOB_ID','')
    assert job.isdigit() and core.is_relative_to(REPO/'h200')
    assert args.parent_boundary is True, 'future serial parent boundary invocation required'
    sessions=[load(core/f'gpu{i}/session.json') for i in (0,1)]
    outcome=load(core/'results/worker_outcome.json')
    certificate=Path(outcome['model_review'])
    assert certificate.resolve()==core/'model/results/native_model_CPU_review.json'
    assert sha(certificate)==outcome['model_review_sha256']
    archives=list(core.glob('results/archive_transfer_*.json'));assert len(archives)==1
    archive=load(archives[0]);manifest_path=Path(archive['source_manifest'])
    assert sha(manifest_path)==archive['manifest_sha256']
    archived_certificate(archive,load(manifest_path),core,sha(certificate))
    assert sha(Path(archive['archive']))==archive['archive_sha256']
    processes=[load(p) for p in core.rglob('results/current_process.json')]
    slurm=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
    gate=admission(sessions,[load(core/f'gpu{i}/results/current_process.json') for i in (0,1)],
        job,socket.gethostname(),os.getuid(),slurm,outcome,load(certificate),
        list(core.parent.glob(core.name+'_*')),processes,__import__('time').time())
    destination=args.destination
    assert destination.parent.resolve()==DESTINATION.resolve() and not destination.is_symlink() and not destination.exists()
    workspaces=[Path(s['workspace']) for s in sessions]
    assert len(set(workspaces))==2
    total=0
    for source in workspaces:
        assert source.parent==Path('/tmp') and not source.is_symlink()
        assert source.stat().st_uid==os.getuid()
        assert not (source/'cache').is_symlink()
        for name in ('vllm','torchinductor','triton','cuda'):
            path=source/'cache'/name
            assert not path.is_symlink()
            for entry in path.rglob('*'):
                assert not entry.is_symlink() and entry.stat().st_uid==os.getuid()
                if entry.is_file():total+=entry.stat().st_size
    assert 0<total<=MAX_TOTAL_BYTES, 'bounded nonempty pair cache payload required'
    receipt_path=core/'results/functional_boundary_cache_export.json'
    assert not receipt_path.exists(), 'preserve earlier attempt'
    record={'status':'checking_idle_pair','classification':'CPU compilation-cache hints only; no measurement qualification',
        'admission':gate,'slurm':slurm,'exporter_sha256':sha(Path(__file__)),
        'copy_helper_sha256':sha(REPO/'scripts/h200_native_model_followup.py'),
        'original_archive_receipt':str(archives[0]),'original_archive_sha256':archive['archive_sha256'],
        'source_sessions':[{'path':str(core/f'gpu{i}/session.json'),'sha256':sha(core/f'gpu{i}/session.json')} for i in (0,1)],
        'planned_bytes':total,'copies':[],'destination':str(destination),
        'cache_bundle_available':False,'performance_qualified':False,
        'scope':'Future serial parent must finish this CPU-only export before launching any next GPU case. Cache hints alone transfer no outputs, calibration, clocks, UUID qualification or timing; importer compatibility and fresh full gates remain mandatory.'}
    with contextlib.ExitStack() as stack:
        for source in workspaces:
            lock=stack.enter_context((source/'producer.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import pynvml as nv
        nv.nvmlInit()
        try:idle_devices(nv,sessions)
        finally:nv.nvmlShutdown()
        # This only exports the original functional workspaces. The future
        # importer must independently qualify platform/model/runtime/source
        # compatibility before trying hints on another allocation.
        destination.mkdir(parents=True,exist_ok=False)
        write(receipt_path,record)
        try:
            record['status']='copying_closed_compilation_caches'
            for i,source in enumerate(workspaces):
                target=destination/f'gpu{i}';target.mkdir(exist_ok=False)
                row=copy_runtime_cache(source,target)
                record['copies'].append(row);write(receipt_path,record)
            assert sum(row['bytes'] for row in record['copies'])==total
            record.update(status='functional_boundary_cache_export_verified',cache_bundle_available=True)
        except BaseException as error:
            record.update(status='partial_or_failed_cache_export',error_type=type(error).__name__,error=str(error))
            raise
        finally:
            write(receipt_path,record);write(destination/'manifest.json',record)
    print(json.dumps({'status':record['status'],'bytes':total,'cache_bundle_available':record['cache_bundle_available']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--parent-boundary',action='store_true')
    main(parser.parse_args())
