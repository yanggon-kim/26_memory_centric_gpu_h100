#!/usr/bin/env python3
"""Prepared fresh allocation: functional gates then fixed3x10producer cohort.

The previous pilot is reused only for runtime bounds. Current functional
references/physical-pair gates and all three new acquisitions remain mandatory.
No Slurm submission and no use of pilot samples in the performance estimate.
An exact-job plan may admit a separate instrumented connector pilot only after
the original fixed cohort is closed, reviewed, published and archived.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from h200_priority_campaign import PACKAGE,load,sha


def main(args):
    assert os.environ.get('SLURM_JOB_ID','').isdigit()
    job=os.environ['SLURM_JOB_ID'];pilot=args.pilot_root.resolve()
    assert pilot.is_relative_to(PACKAGE/'h200')
    subprocess.run([sys.executable,PACKAGE/'scripts/h200_native_functional_campaign.py',
        '--context',str(args.context),'--preceding-job',args.preceding_job,'--spare-native-counter'],check=True)
    roots=list((PACKAGE/'h200').glob('*_job'+job+'_native_functional_c'+str(args.context)));assert len(roots)==1
    source=roots[0];outcome=load(source/'results/worker_outcome.json')
    assert outcome['status']=='functional_prerequisites_passed'
    assert sha(Path(outcome['model_review']))==outcome['model_review_sha256']
    quant=source.with_name(source.name+'_producer_quant')
    subprocess.run([sys.executable,PACKAGE/'scripts/h200_producer_quant_campaign.py','--functional-root',source,
        '--pilot-root',pilot,'--root',quant],check=True)
    from h200_planned_connector_followup import run_if_planned
    return run_if_planned(source,quant)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=int,choices=(16384,32704),required=True)
    parser.add_argument('--preceding-job',required=True);parser.add_argument('--pilot-root',type=Path,required=True)
    main(parser.parse_args())
