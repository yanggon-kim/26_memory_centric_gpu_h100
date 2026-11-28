#!/usr/bin/env python3
"""Prepared allocation entrypoint: native functional gates then producer pilot.

No Slurm submission. Never changes or cancels an existing allocation. The
functional driver owns both devices first; its successful return is required
before the new controller considers an idle pair and remaining-time budget.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from h200_priority_campaign import PACKAGE,load,sha


def main(args):
    assert os.environ.get('SLURM_JOB_ID','').isdigit()
    job=os.environ['SLURM_JOB_ID']
    subprocess.run([sys.executable,PACKAGE/'scripts/h200_native_functional_campaign.py',
        '--context',str(args.context),'--preceding-job',args.preceding_job,'--spare-native-counter'],check=True)
    roots=list((PACKAGE/'h200').glob('*_job'+job+'_native_functional_c'+str(args.context)))
    assert len(roots)==1
    source=roots[0];outcome=load(source/'results/worker_outcome.json')
    assert outcome['status']=='functional_prerequisites_passed'
    assert sha(Path(outcome['model_review']))==outcome['model_review_sha256']
    subprocess.run([sys.executable,PACKAGE/'scripts/h200_producer_timing_followup.py',
        '--functional-root',source,'--root',source.with_name(source.name+'_producer_timing_pilot')],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=int,choices=(16384,32704),required=True)
    parser.add_argument('--preceding-job',required=True)
    main(parser.parse_args())
