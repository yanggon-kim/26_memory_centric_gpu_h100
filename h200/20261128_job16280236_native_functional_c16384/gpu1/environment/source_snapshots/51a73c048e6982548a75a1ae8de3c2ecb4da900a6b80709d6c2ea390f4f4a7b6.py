#!/usr/bin/env python3
"""One bounded same-input reference diagnostic after an owned B8 worker returns."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from h200_priority_campaign import PACKAGE, Worker, load, write, sha, publish, limitation, valid_tokens, strict_reference_record


def validate_idle(old, outcome, current, gpu_pids, reference, source):
    assert outcome.get('status') == 'blocked_resident_pilot'
    assert current.get('status') == 'idle' and not gpu_pids
    assert reference['status'] == 'failed' and reference['gpu_uuid'] == old['gpu_uuid']
    assert (reference['batch'],reference['context'],reference['generated_tokens']) == (8,4096,64)
    assert valid_tokens(source)
    assert reference['prompt_ids'] == source['cases'][0]['prompt_ids']
    assert reference['model_id'] == source['config']['model_id']
    assert reference['revision'] == source['config']['model_revision']
    assert reference['precision'] == source['config']['precision'] == 'BF16'


def main(args):
    old=load(args.source_session);prior=args.source_session.resolve().parent
    assert os.environ.get('SLURM_JOB_ID') == old['slurm_job_id']
    assert socket.gethostname() == old['expected_hostname']
    raw=subprocess.check_output(['scontrol','show','job','-o',old['slurm_job_id']],text=True,timeout=25)
    fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
    assert fields['JobState']=='RUNNING' and fields['NodeList']==socket.gethostname().split('.')[0]
    assert fields['UserId'].endswith('('+str(os.getuid())+')')
    assert 'gres/gpu:h200=2' in fields['AllocTRES'].split(',')
    cutoff=min(dt.datetime.fromisoformat(fields['EndTime']).timestamp()-900,
               dt.datetime.fromisoformat(old['launch_cutoff']).timestamp())
    assert cutoff-time.time()>=420, 'complete bounded diagnostic/publication budget does not fit'
    reference_path=prior/'results/resident_pilot_hf.json';reference=load(reference_path)
    source_path=Path(reference['source_run']);source=load(source_path)
    assert sha(source_path)==reference['source_sha256']
    import pynvml as nv
    nv.nvmlInit()
    try:
        gpu_pids=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByUUID(old['gpu_uuid']))]
    finally: nv.nvmlShutdown()
    outcome=load(prior/'results/worker_outcome.json');current=load(prior/'results/current_process.json')
    validate_idle(old,outcome,current,gpu_pids,reference,source)
    lock=(Path(old['workspace'])/'producer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.root=args.root.resolve();args.root.mkdir(exist_ok=False)
    for name in ('environment','results','logs','configs','runs','analysis','profiles'):
        (args.root/name).mkdir()
    (args.root/'analysis/report.md').write_text('# B8 deterministic HF diagnostic\n\nThe original B8worker returned after an exact reference failure. This is one separate same-input algorithm-control diagnostic, not an accepted tie or tolerance, a changed prompt/precision, a repaired default reference or a latency result. Preserve all dynamic and unchanged controlled StaticCache outputs.\n')
    write(args.root/'environment/idle_transition.json',{'source_session':str(args.source_session),
        'source_session_sha256':sha(args.source_session),'old_outcome':outcome,'old_current':current,
        'gpu_pids':gpu_pids,'failed_reference':str(reference_path),'failed_reference_sha256':sha(reference_path),
        'source_run':str(source_path),'source_run_sha256':sha(source_path),'slurm':raw,
        'cutoff_unix_s':cutoff,'scope':'Original producer lock held and no active selected-GPU compute process; other allocated worker continues independently. No process terminated.'})
    session=dict(old);session['workspace']=old['workspace']+'_detHF_'+args.root.name
    Path(session['workspace']).mkdir(exist_ok=False)
    for key in ('frozen_source_root','frozen_source_identity'): session.pop(key,None)
    session['protocol']='HF_deterministic_diagnostic_v1'
    session_path=args.root/'session.json';write(session_path,session)
    publish(args.root,'H200 B8 reference: publish one bounded deterministic diagnostic before GPU execution')
    session['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip()
    write(session_path,session)
    subprocess.run([session['python'],PACKAGE/'scripts/freeze_shoreline.py','--session',session_path,'--activate'],check=True,timeout=120)
    worker=Worker(args.root,session_path)
    result={'status':'preparing','promotes_original_default_reference':False,'quantitative_qualified':False}
    try:
        assert worker.hs.verify_allocation()['effective_launch_cutoff_unix_s']-time.time()>=360
        output=args.root/'results/deterministic_HF_reference.json'
        proc=worker.execute('B8_deterministic_HF',[worker.py,worker.script/'h200_deterministic_hf_reference.py',
            '--source-run',source_path,'--gpu-uuid',worker.uuid,'--output',output],timeout=300)
        passed=proc['status']=='complete' and output.exists()
        if passed:
            record=load(output);settings=record['deterministic_reference_diagnostic']
            passed=strict_reference_record(record,sha(source_path))
            passed=passed and settings['torch_deterministic_algorithms'] is True and settings['warn_only'] is False
            passed=passed and settings['cublas_workspace_config']==':4096:8'
            passed=passed and record['promotes_original_default_reference'] is False
            passed=passed and len(record['warmups'])==len(record['trials'])==3
            passed=passed and len(record['static_cache_gate']['warmups'])==len(record['static_cache_gate']['trials'])==3
        result.update(status='deterministic_reference_matches_source' if passed else 'blocked_deterministic_reference',
            reference=str(output),reference_sha256=sha(output) if output.exists() else None,
            source_run=str(source_path),source_run_sha256=sha(source_path),
            scope='One same-input diagnostic with stricter process-local algorithm settings; original default oracle remains failed. No SM curve, latency, favorable-output selection or serving qualification follows.')
        if not passed:
            limitation(args.root,'B8_deterministic_reference','One deterministic algorithm/workspace control failed or remained incomplete on the original exact B8inputs',
                'results/deterministic_HF_reference.json; results/processes/B8_deterministic_HF.json',
                'Unchanged HF generate and controlled StaticCache/SDPA functions; deterministic Torch algorithms and explicit cuBLAS workspace,300second bound; all outputs retained',
                'Retain B8as blocked and advance eligible full-GPU traffic/serving; investigate preserved scores separately without selecting favorable outputs/prompts or relaxing the default oracle')
    except BaseException as error:
        result.update(status='failed_deterministic_diagnostic',error=str(error),traceback=traceback.format_exc())
        limitation(args.root,'B8_deterministic_controller',str(error),'results/worker_outcome.json',
            'Same-job returned/idle guard, one new frozen cohort and bounded unchanged-reference call',
            'Preserve this attempt and advance eligible work; do not infer reference repair')
        raise
    finally:
        write(args.root/'results/worker_outcome.json',result)
        with (args.root/'analysis/report.md').open('a') as stream:stream.write('\nDisposition: '+result['status']+'. Original default reference remains failed; no quantitative comparison is promoted.\n')
        publish(args.root,'H200 B8 reference: preserve bounded deterministic control and exact disposition')
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
