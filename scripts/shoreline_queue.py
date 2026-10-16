#!/usr/bin/env python3
"""Serial rental campaign driver with source-family boundaries and closeout reserve."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main(a):
    root=a.session.resolve().parent;package=Path(__file__).resolve().parents[1]
    state_path=root/'results/campaign_queue.json';state={'status':'running','steps':[]}
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    def execute(name,command):
        session=json.loads(a.session.read_text())
        if time.time()>=datetime.datetime.fromisoformat(session['launch_cutoff']).timestamp():
            state['steps'].append({'id':name,'status':'pending','reason':'launch cutoff preserves closeout buffer'});save();return False
        row={'id':name,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command}
        state['steps'].append(row);save();print('QUEUE_START',name,flush=True)
        with (root/'logs'/('queue_'+name+'.log')).open('w') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,cwd=package)
        row.update(exit_code=code,status='command_complete' if code==0 else 'command_failed',
                   ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print('QUEUE_END',name,code,flush=True)
        if code!=0:
            state['status']='needs_diagnosis';save();return False
        return True
    python=str(Path(sys.executable));jobs=str(package/'scripts/shoreline_jobs.py')
    base=[python,jobs,'--session',str(a.session.resolve())]
    # J1 is completed/supervised by the caller before this queue begins.
    steps=[('J1_saved_text',base+['--job','J1','--representative-only']),('J2',base+['--job','J2']),('J3',base+['--job','J3']),
           ('freeze_new_treatments',[python,str(package/'scripts/freeze_shoreline.py'),'--session',str(a.session.resolve()),'--activate']),
           ('J2_metadata',base+['--job','J2','--metadata-only']),
           ('J4_extensions',base+['--job','J4','--extensions']),('J11_semantic_pilot',base+['--job','J11']),('J5',base+['--job','J5']),
           ('J8',base+['--job','J8']),('J9_analysis',[python,str(package/'scripts/shoreline_analyze.py'),'--root',str(root)]),
           ('J7',base+['--job','J7']),('J10',base+['--job','J10']),('J6',base+['--job','J6']),
           ('J11_closeout',base+['--job','J11']),
           ('final_analysis',[python,str(package/'scripts/shoreline_analyze.py'),'--root',str(root)]),
           ('final_figures',[python,str(package/'scripts/plot_shoreline.py'),'--root',str(root)])]
    if a.start_at:steps=steps[next(i for i,(name,_) in enumerate(steps) if name==a.start_at):]
    for name,command in steps:
        if not execute(name,command):return
    state['status']='commands_finished_scientific_gates_separate';save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--start-at')
    main(p.parse_args())
