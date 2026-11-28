#!/usr/bin/env python3
"""Owned same-allocation, separately scoped conditioned eager continuation."""
import argparse
from pathlib import Path
import traceback
SOURCE_AT_IMPORT=Path(__file__).read_bytes()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--session',type=Path,required=True);a=p.parse_args()
    from h200_priority_campaign import Worker,write,publish,limitation
    worker=Worker(a.root,a.session)
    try:
        from h200_resident_campaign import primary
        outcome=primary(worker,conditioned=True)
        write(a.root/'results/worker_outcome.json',outcome)
    except BaseException as error:
        write(a.root/'results/worker_outcome.json',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        limitation(a.root,'conditioned_worker','Conditioned worker stopped; all outputs retained',
            'results/worker_outcome.json','Separate fixed-conditioning cohort with unchanged scientific gates',
            'Inspect the failure; no unconditioned or failed trial promotion')
        raise
    finally:publish(a.root,'H200 fixed conditioned eager follow-up: preserve final complete/partial disposition')
