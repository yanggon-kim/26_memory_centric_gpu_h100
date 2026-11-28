#!/usr/bin/env python3
"""Launch an allocation-bound campaign independently of the terminal connection."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    args=parser.parse_args();session=args.session.resolve();root=session.parent
    os.environ['H200_SESSION_CONFIG']=str(session)
    from h200_session import verify_allocation,environment,dump
    allocation=verify_allocation();settings=json.loads(session.read_text())
    package=Path(__file__).resolve().parents[1]
    command=[settings['python'],str(package/'scripts/continue_shoreline.py'),'--session',str(session)]
    log=root/'logs/detached_continuation_v1.log'
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=package,env=environment(),stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    dump(root/'results/detached_launcher_v1.json',{'pid':process.pid,'command':command,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation':allocation,
        'log':str(log.relative_to(root)),
        'scope':'terminal-independent process within existing allocation; scheduler preemption or cgroup cleanup can still terminate it'})
    print(json.dumps({'pid':process.pid,'log':str(log)}))


if __name__=='__main__':main()
