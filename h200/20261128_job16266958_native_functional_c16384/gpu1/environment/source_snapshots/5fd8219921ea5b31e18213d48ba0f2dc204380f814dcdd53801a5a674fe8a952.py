#!/usr/bin/env python3
"""Publish every Nsight app-range prefill pass before native execution."""
import argparse
import os
from pathlib import Path
import time
import traceback
from software_decode_experiments import dump


def reserve(base,condition):
    base.mkdir(parents=True,exist_ok=True)
    for index in range(100):
        output=base/f'pass{index:03d}';publication=base/f'pass{index:03d}.json'
        if output.exists():continue
        try:fd=os.open(publication,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        except FileExistsError:continue
        os.close(fd)
        dump(publication,{'status':'running','pass':index,'output':str(output),
            'profile_condition':condition,'started_unix_ns':time.time_ns()})
        return output,publication,index
    raise RuntimeError('unexpected application replay count')


def main(a):
    base=a.output;output,publication,index=reserve(base,a.profile_condition)
    a.output=output;a.pilot=not a.full_conditions
    try:
        output.mkdir(parents=True,exist_ok=False)
        from h200_prefill_resident import main as native
        native(a)
        dump(publication,{'status':'complete','pass':index,'output':str(output),
            'profile_condition':a.profile_condition,'finished_unix_ns':time.time_ns()})
    except BaseException as error:
        dump(publication,{'status':'failed','pass':index,'output':str(output),
            'profile_condition':a.profile_condition,'error':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--acquisition',type=int,required=True)
    p.add_argument('--profile-condition',choices=('whole','chunk256','chunk1024','chunk4096','whole_end'),default='whole')
    p.add_argument('--full-conditions',action='store_true');p.add_argument('--external-profiler',action='store_true',required=True)
    p.add_argument('--mode',choices=('timing',),default='timing');p.add_argument('--generated-tokens',type=int,default=64)
    p.add_argument('--batch',type=int);p.add_argument('--context',type=int);p.add_argument('--collect-worker-metadata',action='store_true')
    main(p.parse_args())
