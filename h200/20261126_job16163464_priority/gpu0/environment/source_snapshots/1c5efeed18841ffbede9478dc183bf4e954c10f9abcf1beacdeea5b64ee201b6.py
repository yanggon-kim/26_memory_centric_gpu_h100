#!/usr/bin/env python3
"""CPU-only private archive transfer/readback after GPU collection ends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();package=Path(__file__).resolve().parents[1]
    assert root.is_relative_to(package/'h200')
    os.environ['H200_SESSION_CONFIG']=str(root/'session.json')
    from h200_session import environment,dump
    session=json.loads((root/'session.json').read_text())
    candidates=list((root/'environment').glob('native_trace_micro_*.tar'))+list((root/'profiles').glob('bridge_*.tar'))
    retained=session.get('retained_trace_blocker')
    if retained:
        path=(package/retained['archive']).resolve()
        assert path.is_relative_to(package/'h200')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==retained['archive_sha256']
        candidates.append(path)
    results=[]
    for index,path in enumerate(sorted(set(candidates))):
        label=f'cpu_archive_{index:02d}_v1';record=root/'results'/(label+'.json')
        if record.exists():results.append(json.loads(record.read_text()));continue
        proof=root/'results'/(label+'_readback.json');log=root/'logs'/(label+'.log')
        original_session=path.relative_to(package/'h200').parts[0]
        command=[sys.executable,str(package/'scripts/shoreline_archive.py'),'--file',str(path),
                 '--proof',str(proof),'--remote-path',f'archives/{original_session}/{path.name}']
        with log.open('x') as stream:
            try:code=subprocess.call(command,env=environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
            except subprocess.TimeoutExpired:code=124
        row={'id':label,'exit_code':code,'status':'complete' if code==0 else 'unavailable',
             'archive':str(path.relative_to(package)),'log':str(log.relative_to(root)),
             'readback':str(proof.relative_to(root)) if proof.exists() else None,
             'scope':'private remote upload/readback only; no GPU or scheduler requirement'}
        dump(record,row);results.append(row)
        subprocess.run([sys.executable,str(package/'scripts/checkpoint_h200.py'),'--root',str(root),
                        '--message',f'CPU archive checkpoint: {label} {row["status"]}','--push'],check=True)
    dump(root/'results/shoreline_archive_closeout.json',{'attempts':results,
         'all_selected_readbacks_verified':bool(results) and all(x['status']=='complete' for x in results)})


if __name__=='__main__':main()
