#!/usr/bin/env python3
"""Fresh same-allocation cohort for the import-preflight capture repair.

Launch directly in an owned Slurm step only after the prior worker has exited.
This orchestrator uses the ordinary per-producer supervisor; do not nest it
inside h200_session.py. No allocation is requested or changed here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from h200_priority_campaign import Worker, PACKAGE, load, write, publish, limitation

SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',type=Path,required=True)
    p.add_argument('--prior-worker-outcome',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    a=p.parse_args()
    old=load(a.source_session)
    assert old['slurm_job_id']==os.environ['SLURM_JOB_ID']
    assert old['gpu_uuid']==a.gpu_uuid
    outcome=load(a.prior_worker_outcome)
    assert outcome['status']=='captures_saved_scientific_gates_separate'
    assert old['frozen_source_identity']
    a.root=a.root.resolve();a.root.mkdir(exist_ok=False)
    for folder in ('environment','results','logs','configs','runs','profiles','analysis'):
        (a.root/folder).mkdir()
    (a.root/'analysis/report.md').write_text('# Optimized32B import-preflight repair cohort\n\nFresh source cohort on the same allocated UUID. No pooling with the original mixed-context or failed import attempts. Exact HF outputs, actual SM grants, real NSYS graph-kernel placement and3x10 timing gates remain binding.\n')
    session=dict(old)
    session['workspace']=str(Path(old['workspace']).with_name(Path(old['workspace']).name+'_import_preflight'))
    Path(session['workspace']).mkdir(exist_ok=False)
    session['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name in ('frozen_source_root','frozen_source_identity'):session.pop(name,None)
    session_path=a.root/'session.json';write(session_path,session)
    subprocess.run([session['python'],PACKAGE/'scripts/freeze_shoreline.py','--session',session_path,'--activate'],check=True)
    worker=Worker(a.root,session_path)
    worker.hs.verify_allocation()
    worker.rebind_step=True;worker.capture_bind=True
    start=time.monotonic()
    stock=worker.case('graph',1.,0,stock=True,pilot=True)
    if not stock['qualified'] or not worker.reference(stock['run'],'graph_target'):
        write(a.root/'results/followup_outcome.json',{'status':'blocked','reason':'fresh stock NSYS/HF gate'})
        publish(a.root,'H200 fresh repair cohort: retain blocked stock gate')
        return
    worker.expected=load(stock['run'])['cases'][0]['trials'][0]['output_ids']
    # One corrected full-green pilot bounds the integration repair. Do not
    # start expensive cuts or acquire a curve unless it actually passes.
    proof=worker.case('graph',1.,0,pilot=True,suffix='_integration_check')
    write(a.root/'results/integration_pilot.json',{'status':'passed' if proof['qualified'] else 'blocked',
          'elapsed_s':time.monotonic()-start,'stock':stock['id'],'full_green':proof['id'],
          'scope':'fresh import-preflight integration pilot; no latency threshold inferred'})
    if not proof['qualified']:
        limitation(a.root,'import_preflight_integration','Corrected capture adapter still failed full-green measured proof',
                    'results/integration_pilot.json','One fresh full-green NSYS pilot after identity-checked import preflight',
                    'Keep optimized restriction blocked; retain separately collected eager/full-GPU cases; no further repair cycle in this follow-up')
        publish(a.root,'H200 corrected integration pilot: preserve blocked placement')
        return
    qualified=worker.curve('graph',stock_proof=stock)
    write(a.root/'results/followup_outcome.json',{'status':'curve_attempted','complete_qualified_curve':qualified,
          'stop':worker.last_curve_stop,'scope':'fresh source/UUID cohort; no cross-allocation pooling'})
    publish(a.root,'H200 optimized32B import-preflight cohort: curve disposition')


if __name__=='__main__':main()
