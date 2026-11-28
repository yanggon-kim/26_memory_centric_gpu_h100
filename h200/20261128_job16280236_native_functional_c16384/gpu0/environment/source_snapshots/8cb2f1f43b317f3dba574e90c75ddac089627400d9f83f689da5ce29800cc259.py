#!/usr/bin/env python3
"""Optional bounded read-only runtime page-cache diagnostic inside allocation.

Never imports torch, changes installed files or substitutes for runtime checks.
Only counts/paths are published; binary contents are not logged.
"""
import argparse
import json
from pathlib import Path
import time


def candidates(environment):
    environment=environment.resolve()
    sites=sorted((environment/'lib').glob('python*/site-packages'))
    found=[];seen=set()
    for site in sites:
        paths=[]
        for pattern in ('torch/_C*.so','torch/lib/libtorch_global_deps.so','torch/lib/libc10.so',
                        'torch/lib/libtorch_python.so','torch/lib/libtorch_cpu.so','torch/lib/libtorch_cuda.so',
                        'torch/lib/*.so','nvidia/*/lib/*.so*'):
            paths.extend(sorted(site.glob(pattern)))
        for path in paths:
            target=path.resolve()
            if target in seen or not target.is_relative_to(environment) or not target.is_file():continue
            seen.add(target);found.append(target)
    return found


def warm(paths,max_bytes,seconds):
    assert max_bytes>0 and seconds>0
    began=time.monotonic();total=0;rows=[];reason=None
    for path in paths:
        row={'path':str(path),'file_bytes':path.stat().st_size,'bytes_read':0}
        try:
            with path.open('rb') as stream:
                while True:
                    if total>=max_bytes:reason='byte_budget';break
                    if time.monotonic()-began>=seconds:reason='time_budget';break
                    data=stream.read(min(8*1024**2,max_bytes-total))
                    if not data:break
                    row['bytes_read']+=len(data);total+=len(data)
        except OSError as error:row['read_error']=str(error)
        rows.append(row)
        if reason:break
    return {'status':'partial_budget' if reason else 'read_attempts_finished','stop_reason':reason,
            'elapsed_s':time.monotonic()-began,'bytes_read':total,'files':rows,
            'classification':'optional CPU-only read-only page-cache diagnostic; no GPU measurement',
            'scope':'No installed bytes changed; success does not establish runtime correctness or improve scientific qualification; mandatory version/device/model checks still run'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-bytes',type=int,default=8*1024**3)
    p.add_argument('--seconds',type=float,default=60)
    a=p.parse_args()
    import os
    assert os.environ.get('SLURM_JOB_ID','').isdigit(),'runtime prewarm is allocation-only'
    result=warm(candidates(a.environment),a.max_bytes,a.seconds)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'bytes_read':result['bytes_read'],'elapsed_s':result['elapsed_s']}))
