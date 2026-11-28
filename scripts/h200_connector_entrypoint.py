#!/usr/bin/env python3
"""Fresh allocation: strict native functional gates, then connector attribution.

Dedicated instrumented diagnostic after the preserved producer cohort failed.
No producer timing resampling, Slurm submission, deadline extension or claim of
clean latency. The controller independently enforces source, idle-pair, current
allocation and complete capture/export/review budget gates.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from h200_priority_campaign import PACKAGE, load, sha


def main(args):
    job = os.environ.get('SLURM_JOB_ID', '')
    assert job.isdigit() and args.preceding_job.isdigit()
    assert args.context in (16384, 32704)
    subprocess.run([sys.executable, PACKAGE/'scripts/h200_native_functional_campaign.py',
        '--context', str(args.context), '--preceding-job', args.preceding_job,
        '--spare-native-counter'], check=True)
    roots = list((PACKAGE/'h200').glob('*_job'+job+'_native_functional_c'+str(args.context)))
    assert len(roots) == 1, 'one fresh current-job functional root required'
    core = roots[0]
    outcome = load(core/'results/worker_outcome.json')
    assert outcome['status'] == 'functional_prerequisites_passed'
    certificate = core/'model/results/native_model_CPU_review.json'
    assert Path(outcome['model_review']).resolve() == certificate.resolve()
    assert sha(certificate) == outcome['model_review_sha256']
    reviewed = load(certificate)
    assert reviewed['status'] == 'native_model_functional_review_passed'
    assert reviewed['job_id'] == job and not reviewed['errors']
    target = core.with_name(core.name+'_connector_capture')
    assert not target.exists(), 'preserve every earlier capture root'
    subprocess.run([sys.executable, PACKAGE/'scripts/h200_connector_capture_campaign.py',
        '--functional-root', core, '--root', target], check=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, choices=(16384, 32704), required=True)
    parser.add_argument('--preceding-job', required=True)
    main(parser.parse_args())
