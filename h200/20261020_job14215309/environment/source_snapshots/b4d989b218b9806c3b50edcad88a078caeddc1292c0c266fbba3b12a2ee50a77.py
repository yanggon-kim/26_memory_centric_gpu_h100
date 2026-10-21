#!/usr/bin/env python3
"""Resume remaining jobs in a fresh, explicitly allocated session namespace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    args=parser.parse_args();session=args.session.resolve();root=session.parent
    package=Path(__file__).resolve().parents[1]
    os.environ['H200_SESSION_CONFIG']=str(session)
    from h200_session import verify_allocation,run,python_command,dump
    verify_allocation()
    steps=[('runtime_dependency_check_v1',[sys.executable,'-m','pip','check']),
           ('batch_recovery_contracts_v1',[sys.executable,str(package/'scripts/test_h200_batch.py')]),
           ('gap_repair_contracts_v2',[sys.executable,str(package/'scripts/test_h200_gaps.py')]),
           ('failure_recovery_contracts_v3',[sys.executable,str(package/'scripts/test_failure_recovery.py')]),
           ('decoder_calibration_contracts_v4',[sys.executable,str(package/'scripts/test_decoder_calibration.py')]),
           ('continuation_contracts_v1',[sys.executable,str(package/'scripts/test_shoreline_contracts.py')]),
           ('J0_inventory',[sys.executable,str(package/'scripts/shoreline_campaign.py'),'--session',str(session),'--job','J0']),
           ('J1_functional_smokes',[sys.executable,str(package/'scripts/shoreline_campaign.py'),'--session',str(session),'--job','J1'])]
    for name,command in steps:
        log=root/'logs'/(name+'.log')
        marker=root/'results'/(name+'.json')
        if marker.exists():
            assert json.loads(marker.read_text())['exit_code']==0, 'failed setup requires a versioned diagnostic, not an automatic retry'
            continue
        with log.open('x') as stream:
            code=subprocess.call(command,cwd=package,stdout=stream,stderr=subprocess.STDOUT)
        dump(marker,{'exit_code':code,'command':command,'log':str(log.relative_to(root))})
        if code:raise SystemExit(code)
    import shoreline_campaign as campaign
    for key in ('PACKAGE','PYTHON','UUID','dump','run','python_command'):
        if key=='PACKAGE':value=package
        else:
            import h200_session
            value=getattr(h200_session,key)
        setattr(campaign,key,value)
    registry=root/'results/prior_capture_evidence.json'
    prior=json.loads(registry.read_text())
    for row in prior['evidence']:
        assert hashlib.sha256((package/row['path']).read_bytes()).hexdigest()==row['sha256']
    for job in ('J2','J3','J4'):
        campaign.update_job(root,job,'partial','Prior completed captures retained with hashes; no new acquisition and no quantitative pooling across sessions. See prior_capture_evidence.json and previous completion reviews.')
    for step,proof in json.loads(session.read_text()).get('queue_reused_jobs',{}).items():
        assert hashlib.sha256((package/proof['path']).read_bytes()).hexdigest()==proof['sha256']
        campaign.update_job(root,'J11' if step=='J11_gpu' else step,'partial',
            'Prior command-finished disposition retained; no new measurement or scientific pass inferred. See session.queue_reused_jobs.')
    campaign.checkpoint(root,'Fresh allocation ready: exact functional controls and prior evidence verified; resume remaining jobs')
    if json.loads(session.read_text()).get('decoder_calibration_version')=='decoder_calibration_v4':
        raise SystemExit(subprocess.call([sys.executable,str(package/'scripts/calibrate_h200_decoder.py'),
            '--session',str(session)],cwd=package))
    if json.loads(session.read_text()).get('failure_diagnostics_version')=='failure_recovery_v3':
        raise SystemExit(subprocess.call([sys.executable,str(package/'scripts/recover_h200_failures.py'),
            '--session',str(session)],cwd=package))
    schedule=json.loads(session.read_text()).get('queue_schedule','legacy')
    command=[sys.executable,str(package/'scripts/shoreline_queue.py'),'--session',str(session),'--start-at','J5','--schedule',schedule]
    with (root/'logs/remaining_campaign_v1.log').open('x') as stream:
        code=subprocess.call(command,cwd=package,stdout=stream,stderr=subprocess.STDOUT)
    raise SystemExit(code)


if __name__=='__main__':main()
